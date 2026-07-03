"""Genuine live next-hour demand forecast.

Distinct from train_and_score_models, which only evaluates the demand
forecaster against a held-out slice of historical data to measure accuracy
(a backtest). This DAG never trains anything - it takes whatever model
train_and_score_models most recently produced and scores it against CURRENT
live conditions (common/gbfs_client.py-sourced station_status_snapshots),
writing a real forward-looking prediction (target_ts in the future) rather
than a retrospective one.
"""
from __future__ import annotations

import datetime as dt

from airflow.sdk import dag, task
from sqlalchemy import text

from common.db import get_engine
from ml import demand_forecast

UPSERT_PREDICTION_SQL = text(
    """
    INSERT INTO model_predictions (station_id, target_ts, predicted_value, model_version, generated_at)
    VALUES (:station_id, :target_ts, :predicted_value, :model_version, now())
    ON CONFLICT (station_id, target_ts, model_version) DO UPDATE SET
        predicted_value = EXCLUDED.predicted_value,
        generated_at = now()
    """
)


@dag(
    dag_id="predict_live_demand",
    schedule=dt.timedelta(minutes=12),
    start_date=dt.datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "live-inference"],
    # Writes to model_predictions, which FKs to `stations` - same shared-table
    # deadlock risk as ingest_live_status's writes to `stations` itself (see
    # that DAG's default_args). Retries are the fix there too.
    default_args={"retries": 2, "retry_delay": dt.timedelta(seconds=30)},
)
def predict_live_demand():
    @task
    def predict_and_write() -> int:
        engine = get_engine()
        predictions = demand_forecast.predict_live(engine)
        if predictions.empty:
            return 0
        with engine.begin() as conn:
            conn.execute(UPSERT_PREDICTION_SQL, predictions.to_dict(orient="records"))
        return len(predictions)

    predict_and_write()


predict_live_demand()
