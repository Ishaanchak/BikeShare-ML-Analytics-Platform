"""Train and score the next-hour demand forecasting model.

Predicts next-hour net bike flow (departures - arrivals) per station and
evaluates against a naive "same hour, last week" baseline (ml.features.
BASELINE_COLUMN), so the improvement over that baseline is demonstrable.
"""
from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sqlalchemy import Engine, text

from ml.features import (
    BASELINE_COLUMN,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_demand_features,
    build_live_features,
    load_hourly_demand,
    train_test_split_chronological,
)

MODEL_DIR = os.environ.get("ML_MODEL_DIR", "/opt/airflow/ml/models")


@dataclass
class DemandForecastResult:
    model_version: str
    model_path: str
    model_mae: float
    model_rmse: float
    baseline_mae: float
    baseline_rmse: float
    predictions: pd.DataFrame  # columns: station_id, target_ts, predicted_value


def train_and_score(
    engine: Engine, model_version: str | None = None, lookback_days: int | None = 90
) -> DemandForecastResult:
    """`lookback_days` bounds training to a rolling recent window rather than
    all-time history: ridership patterns drift over months (new stations,
    seasonal shifts), so a recent window is arguably more representative of
    current demand than the full history anyway - and it keeps the
    read_sql -> feature engineering -> RandomForest.fit pipeline's memory
    footprint bounded regardless of how much history backfill_historical_trips
    has accumulated. Override via ML_TRAIN_LOOKBACK_DAYS, or pass None for
    all-time history on a machine with the RAM to spare.
    """
    model_version = model_version or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    env_lookback = os.environ.get("ML_TRAIN_LOOKBACK_DAYS")
    if env_lookback is not None:
        lookback_days = int(env_lookback) if env_lookback else None

    raw = load_hourly_demand(engine, lookback_days=lookback_days)
    if raw.empty:
        raise ValueError("hourly_station_demand is empty - run backfill_historical_trips first")

    df = build_demand_features(raw)
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN, BASELINE_COLUMN])
    if df.empty:
        raise ValueError(
            "no rows with complete lag/history features - need at least ~1 week of "
            "hourly_station_demand history per station"
        )

    train, test = train_test_split_chronological(df, test_fraction=0.2)
    if train.empty or test.empty:
        raise ValueError("chronological split produced an empty train or test set")

    X_train, y_train = train[FEATURE_COLUMNS], train[TARGET_COLUMN]
    X_test, y_test = test[FEATURE_COLUMNS], test[TARGET_COLUMN]

    # n_jobs defaults to 1 (not -1, and not a higher bounded value either):
    # this runs inside the same docker-compose stack as Postgres and
    # Airflow's own scheduler/dag-processor/triggerer. Beyond starving them
    # of CPU, sklearn's process-based parallelism duplicates the training
    # data per worker process, multiplying memory use on what's meant to be
    # a modest local-dev machine. Override via ML_TRAIN_N_JOBS if you have
    # cores and RAM to spare.
    model = RandomForestRegressor(
        n_estimators=200, max_depth=12, random_state=0, n_jobs=int(os.environ.get("ML_TRAIN_N_JOBS", "1"))
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Cast off numpy scalar types here - sklearn's metrics return np.float64,
    # which a single-dict (non-executemany) SQL param binding mishandles.
    model_mae = float(mean_absolute_error(y_test, y_pred))
    model_rmse = float(mean_squared_error(y_test, y_pred) ** 0.5)

    baseline_pred = test[BASELINE_COLUMN]
    baseline_mae = float(mean_absolute_error(y_test, baseline_pred))
    baseline_rmse = float(mean_squared_error(y_test, baseline_pred) ** 0.5)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"demand_forecast_{model_version}.joblib")
    joblib.dump(model, model_path)

    predictions = pd.DataFrame(
        {
            "station_id": test["station_id"].to_numpy(),
            # .dt.to_pydatetime() (not .values) - keeps native datetimes rather
            # than numpy.datetime64, which psycopg2 can't adapt directly.
            "target_ts": test["ts"].dt.to_pydatetime(),
            "predicted_value": y_pred.astype(float),
        }
    )

    return DemandForecastResult(
        model_version=model_version,
        model_path=model_path,
        model_mae=model_mae,
        model_rmse=model_rmse,
        baseline_mae=baseline_mae,
        baseline_rmse=baseline_rmse,
        predictions=predictions,
    )


def load_latest_model(engine: Engine) -> tuple[RandomForestRegressor, str]:
    """Load the joblib artifact for whichever model_version most recently
    wrote to model_predictions - i.e. the model train_and_score_models last
    produced, regardless of whether this process trained it."""
    with engine.connect() as conn:
        version = conn.execute(
            text("SELECT model_version FROM model_predictions ORDER BY generated_at DESC LIMIT 1")
        ).scalar_one_or_none()
    if version is None:
        raise ValueError("no trained demand_forecast model yet - run train_and_score_models first")
    model_path = os.path.join(MODEL_DIR, f"demand_forecast_{version}.joblib")
    return joblib.load(model_path), version


def predict_live(engine: Engine, as_of: dt.datetime | None = None) -> pd.DataFrame:
    """Score the latest trained model against CURRENT live conditions - a
    genuine forward-looking forecast for the upcoming hour, not a backtest.

    Distinct from train_and_score: this never trains anything, it just
    applies whatever model that function most recently produced. Returns
    columns: station_id, target_ts, predicted_value, model_version.
    """
    model, model_version = load_latest_model(engine)
    features = build_live_features(engine, as_of=as_of)
    if features.empty:
        return pd.DataFrame(columns=["station_id", "target_ts", "predicted_value", "model_version"])

    predicted = model.predict(features[FEATURE_COLUMNS])
    return pd.DataFrame(
        {
            "station_id": features["station_id"].to_numpy(),
            "target_ts": features["target_ts"],
            "predicted_value": predicted.astype(float),
            "model_version": model_version,
        }
    )
