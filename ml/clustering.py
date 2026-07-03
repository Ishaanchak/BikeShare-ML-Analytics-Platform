"""Station usage-archetype clustering.

KMeans over per-station usage features (trip volume, weekday/weekend split,
peak hour, trip duration, net-flow variance). k is chosen by silhouette
score over a small candidate range rather than fixed a priori, and clusters
are labeled post-hoc by inspecting their (inverse-scaled) centers.
"""
from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sqlalchemy import Engine, text

MODEL_DIR = os.environ.get("ML_MODEL_DIR", "/opt/airflow/ml/models")

FEATURE_COLUMNS = [
    "avg_trips_per_day",
    "weekday_weekend_ratio",
    "peak_hour",
    "avg_trip_duration_min",
    "net_flow_variance",
]

_USAGE_QUERY = text(
    """
    -- Pre-aggregating to daily grain before the weekday/weekend split avoids
    -- COUNT(DISTINCT ...) FILTER, which forces Postgres to track a separate
    -- distinct set per filter branch and is dramatically slower than a plain
    -- COUNT(*) over rows already grouped to one-per-station-per-day.
    WITH daily AS (
        SELECT station_id,
               date,
               sum(departures) AS departures,
               (extract(dow FROM date) IN (0, 6)) AS is_weekend
        FROM hourly_station_demand
        GROUP BY station_id, date
    ),
    daily_agg AS (
        SELECT station_id,
               count(*) AS num_days,
               sum(departures) FILTER (WHERE NOT is_weekend) AS weekday_departures,
               sum(departures) FILTER (WHERE is_weekend) AS weekend_departures,
               count(*) FILTER (WHERE NOT is_weekend) AS num_weekdays,
               count(*) FILTER (WHERE is_weekend) AS num_weekend_days
        FROM daily
        GROUP BY station_id
    ),
    hourly_agg AS (
        SELECT station_id,
               sum(departures + arrivals) AS total_activity,
               variance(net_flow) AS net_flow_variance
        FROM hourly_station_demand
        GROUP BY station_id
    )
    SELECT d.station_id, d.num_days, h.total_activity, d.weekday_departures, d.weekend_departures,
           d.num_weekdays, d.num_weekend_days, h.net_flow_variance
    FROM daily_agg d
    JOIN hourly_agg h ON h.station_id = d.station_id
    """
)

_PEAK_HOUR_QUERY = text(
    """
    SELECT station_id, hour, sum(departures) AS total_departures
    FROM hourly_station_demand
    GROUP BY station_id, hour
    """
)

_DURATION_QUERY = text(
    """
    SELECT s.station_id,
           avg(EXTRACT(EPOCH FROM (t.ended_at - t.started_at)) / 60.0) AS avg_trip_duration_min
    FROM trips t
    JOIN stations s ON s.short_name = t.start_station_id
    WHERE t.ended_at > t.started_at
    GROUP BY s.station_id
    """
)


def load_station_usage_features(engine: Engine) -> pd.DataFrame:
    """One row per station with the raw aggregates FEATURE_COLUMNS is derived from."""
    with engine.connect() as conn:
        usage = pd.read_sql(_USAGE_QUERY, conn)
        peak = pd.read_sql(_PEAK_HOUR_QUERY, conn)
        duration = pd.read_sql(_DURATION_QUERY, conn)

    peak_hour = (
        peak.sort_values("total_departures", ascending=False)
        .drop_duplicates("station_id")[["station_id", "hour"]]
        .rename(columns={"hour": "peak_hour"})
    )

    df = usage.merge(peak_hour, on="station_id", how="left").merge(
        duration, on="station_id", how="left"
    )

    df["avg_trips_per_day"] = df["total_activity"] / df["num_days"].replace(0, np.nan)
    weekday_avg = df["weekday_departures"] / df["num_weekdays"].replace(0, np.nan)
    weekend_avg = df["weekend_departures"] / df["num_weekend_days"].replace(0, np.nan)
    df["weekday_weekend_ratio"] = weekday_avg / (weekend_avg.fillna(0) + 1e-3)

    return df


def _label_cluster(center: pd.Series) -> str:
    """Heuristic post-hoc label from a cluster's (inverse-scaled) center."""
    if center["weekday_weekend_ratio"] > 1.3 and center["avg_trips_per_day"] > 0:
        return "commuter hub"
    if center["weekday_weekend_ratio"] < 0.8:
        return "leisure / weekend"
    return "balanced / mixed"


@dataclass
class ClusteringResult:
    model_version: str
    model_path: str
    chosen_k: int
    silhouette: float
    assignments: pd.DataFrame  # columns: station_id, cluster_id, cluster_label


def train_and_score(
    engine: Engine, model_version: str | None = None, k_candidates: range = range(2, 8)
) -> ClusteringResult:
    model_version = model_version or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    raw = load_station_usage_features(engine)
    df = raw.dropna(subset=FEATURE_COLUMNS)
    if len(df) < max(k_candidates) + 1:
        raise ValueError(
            f"only {len(df)} stations have complete usage features - need more history/stations"
        )

    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURE_COLUMNS])

    best_k, best_score, best_model = None, -1.0, None
    for k in k_candidates:
        model = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = model.fit_predict(X)
        score = float(silhouette_score(X, labels))
        if score > best_score:
            best_k, best_score, best_model = k, score, model

    labels = best_model.predict(X).astype(int)
    centers_scaled = best_model.cluster_centers_
    centers = pd.DataFrame(scaler.inverse_transform(centers_scaled), columns=FEATURE_COLUMNS)
    cluster_labels = {i: _label_cluster(centers.iloc[i]) for i in range(best_k)}

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"clustering_{model_version}.joblib")
    joblib.dump({"scaler": scaler, "kmeans": best_model, "cluster_labels": cluster_labels}, model_path)

    assignments = pd.DataFrame(
        {
            "station_id": df["station_id"].values,
            "cluster_id": labels,
            "cluster_label": [cluster_labels[c] for c in labels],
        }
    )

    return ClusteringResult(
        model_version=model_version,
        model_path=model_path,
        chosen_k=best_k,
        silhouette=best_score,
        assignments=assignments,
    )
