"""Stockout / overflow anomaly detection.

Distinguishes "predictably empty/full" (matches a station's own hour/day-of-
week pattern) from "anomalous" (an unusual deviation from that pattern) using
an IsolationForest over rate-of-change and deviation-from-own-history
features, rather than a fixed bikes-available threshold. Outputs a severity
score per flagged station-snapshot, not just a binary.
"""
from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sqlalchemy import Engine, text

from ml.features import expanding_hour_dow_mean

MODEL_DIR = os.environ.get("ML_MODEL_DIR", "/opt/airflow/ml/models")

FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "occupancy_ratio",
    "rate_of_change",
    "deviation_from_hist_avg",
]


@dataclass
class AnomalyDetectionResult:
    model_version: str
    model_path: str
    num_flags: int
    flags: pd.DataFrame  # columns: station_id, ts, risk_type, severity_score


def load_recent_snapshots(engine: Engine, lookback_days: int = 30) -> pd.DataFrame:
    query = text(
        """
        SELECT sss.station_id, sss.ts, sss.num_bikes_available, sss.num_docks_available, s.capacity
        FROM station_status_snapshots sss
        JOIN stations s ON s.station_id = sss.station_id
        WHERE sss.ts >= now() - (:lookback_days * INTERVAL '1 day')
        ORDER BY sss.station_id, sss.ts
        """
    )
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"lookback_days": lookback_days})


def build_anomaly_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rate-of-change and own-history-deviation features, one row per snapshot.

    hist_avg_hour_dow (and therefore deviation_from_hist_avg) uses only prior
    snapshots for that station/hour/day-of-week - a freshly-deployed system
    with only a few days of snapshots will have mostly-NaN deviations until
    ingest_live_status has been running long enough to build up history.
    """
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values(["station_id", "ts"]).reset_index(drop=True)

    df["hour"] = df["ts"].dt.hour
    df["day_of_week"] = df["ts"].dt.dayofweek
    df["occupancy_ratio"] = df["num_bikes_available"] / df["capacity"].replace(0, np.nan)

    df["rate_of_change"] = df.groupby("station_id")["num_bikes_available"].diff()

    df["hist_avg_hour_dow"] = expanding_hour_dow_mean(
        df, ["station_id", "hour", "day_of_week"], "num_bikes_available"
    )
    df["deviation_from_hist_avg"] = df["num_bikes_available"] - df["hist_avg_hour_dow"]

    return df


def train_and_score(
    engine: Engine, model_version: str | None = None, lookback_days: int = 30
) -> AnomalyDetectionResult:
    model_version = model_version or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    raw = load_recent_snapshots(engine, lookback_days=lookback_days)
    if raw.empty:
        raise ValueError("station_status_snapshots is empty - run ingest_live_status first")

    df = build_anomaly_features(raw)
    scored = df.dropna(subset=FEATURE_COLUMNS)
    if scored.empty:
        raise ValueError("no rows with complete rate-of-change/history features yet")

    model = IsolationForest(n_estimators=200, contamination="auto", random_state=0)
    model.fit(scored[FEATURE_COLUMNS])

    decision_scores = model.decision_function(scored[FEATURE_COLUMNS])  # higher = more normal
    is_outlier = model.predict(scored[FEATURE_COLUMNS]) == -1
    score_range = decision_scores.max() - decision_scores.min()
    # .astype(float) - plain Python floats, not np.float64, for the DB layer.
    severity = ((decision_scores.max() - decision_scores) / (score_range + 1e-9)).astype(float)

    scored = scored.assign(is_outlier=is_outlier, severity_score=severity)
    flagged = scored[scored["is_outlier"]].copy()
    # Unusually few bikes (docks unusually full) -> stockout risk; unusually
    # many bikes (docks unusually empty) -> overflow (can't return) risk.
    flagged["risk_type"] = flagged["deviation_from_hist_avg"].apply(
        lambda d: "stockout" if d < 0 else "overflow"
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"anomaly_detection_{model_version}.joblib")
    joblib.dump(model, model_path)

    flags = flagged[["station_id", "ts", "risk_type", "severity_score"]].reset_index(drop=True)

    return AnomalyDetectionResult(
        model_version=model_version,
        model_path=model_path,
        num_flags=len(flags),
        flags=flags,
    )
