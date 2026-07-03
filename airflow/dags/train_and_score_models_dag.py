"""Train and score the demand forecasting, anomaly detection, and clustering
models, and persist their results to Postgres.

Requires the two ingestion DAGs to have already produced usable data:
ingest_live_status recently (fresh station_status_snapshots) and
backfill_historical_trips at least once (any trips at all). Enforced here as
a direct data-freshness check rather than an Airflow ExternalTaskSensor,
since the two upstream DAGs run on unrelated schedules (every ~12 minutes vs.
monthly) that a run-alignment sensor can't sensibly wait on.
"""
from __future__ import annotations

import datetime as dt

from airflow.sdk import dag, task
from sqlalchemy import text

from common.db import get_engine
from ml import anomaly_detection, clustering, demand_forecast

FRESHNESS_WINDOW_MINUTES = 60

UPSERT_MODEL_RUN_SQL = text(
    """
    INSERT INTO model_runs (model_name, run_ts, metric_name, metric_value, baseline_metric_value)
    VALUES (:model_name, :run_ts, :metric_name, :metric_value, :baseline_metric_value)
    ON CONFLICT (model_name, run_ts, metric_name) DO UPDATE SET
        metric_value = EXCLUDED.metric_value,
        baseline_metric_value = EXCLUDED.baseline_metric_value
    """
)

UPSERT_PREDICTION_SQL = text(
    """
    INSERT INTO model_predictions (station_id, target_ts, predicted_value, model_version, generated_at)
    VALUES (:station_id, :target_ts, :predicted_value, :model_version, now())
    ON CONFLICT (station_id, target_ts, model_version) DO UPDATE SET
        predicted_value = EXCLUDED.predicted_value,
        generated_at = now()
    """
)

UPSERT_ANOMALY_FLAG_SQL = text(
    """
    INSERT INTO anomaly_flags (station_id, ts, risk_type, severity_score, model_version)
    VALUES (:station_id, :ts, :risk_type, :severity_score, :model_version)
    ON CONFLICT (station_id, ts, risk_type, model_version) DO UPDATE SET
        severity_score = EXCLUDED.severity_score
    """
)

UPSERT_STATION_CLUSTER_SQL = text(
    """
    INSERT INTO station_clusters (station_id, cluster_id, cluster_label, model_version, computed_at)
    VALUES (:station_id, :cluster_id, :cluster_label, :model_version, now())
    ON CONFLICT (station_id, model_version) DO UPDATE SET
        cluster_id = EXCLUDED.cluster_id,
        cluster_label = EXCLUDED.cluster_label,
        computed_at = now()
    """
)


@dag(
    dag_id="train_and_score_models",
    schedule="@daily",
    start_date=dt.datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "training"],
)
def train_and_score_models():
    @task
    def check_upstream_data_ready() -> str:
        engine = get_engine()
        with engine.connect() as conn:
            recent_snapshots = conn.execute(
                text(
                    "SELECT count(*) FROM station_status_snapshots "
                    "WHERE ts >= now() - (:minutes * INTERVAL '1 minute')"
                ),
                {"minutes": FRESHNESS_WINDOW_MINUTES},
            ).scalar_one()
            any_trips = conn.execute(text("SELECT count(*) FROM trips")).scalar_one()
        if recent_snapshots == 0:
            raise ValueError(
                f"no station_status_snapshots in the last {FRESHNESS_WINDOW_MINUTES} "
                "minutes - run ingest_live_status first"
            )
        if any_trips == 0:
            raise ValueError(
                "trips table is empty - run backfill_historical_trips at least once first"
            )
        return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    @task
    def train_demand_forecast_task(model_version: str) -> None:
        engine = get_engine()
        result = demand_forecast.train_and_score(engine, model_version=model_version)
        run_ts = dt.datetime.now(dt.timezone.utc)
        with engine.begin() as conn:
            conn.execute(
                UPSERT_MODEL_RUN_SQL,
                [
                    {
                        "model_name": "demand_forecast",
                        "run_ts": run_ts,
                        "metric_name": "mae",
                        "metric_value": result.model_mae,
                        "baseline_metric_value": result.baseline_mae,
                    },
                    {
                        "model_name": "demand_forecast",
                        "run_ts": run_ts,
                        "metric_name": "rmse",
                        "metric_value": result.model_rmse,
                        "baseline_metric_value": result.baseline_rmse,
                    },
                ],
            )
            predictions = result.predictions.assign(model_version=model_version)
            conn.execute(UPSERT_PREDICTION_SQL, predictions.to_dict(orient="records"))

    @task
    def train_anomaly_detection_task(model_version: str) -> None:
        engine = get_engine()
        result = anomaly_detection.train_and_score(engine, model_version=model_version)
        if result.num_flags == 0:
            return
        with engine.begin() as conn:
            flags = result.flags.assign(model_version=model_version)
            conn.execute(UPSERT_ANOMALY_FLAG_SQL, flags.to_dict(orient="records"))

    @task
    def train_clustering_task(model_version: str) -> None:
        engine = get_engine()
        result = clustering.train_and_score(engine, model_version=model_version)
        run_ts = dt.datetime.now(dt.timezone.utc)
        with engine.begin() as conn:
            conn.execute(
                UPSERT_MODEL_RUN_SQL,
                {
                    "model_name": "clustering",
                    "run_ts": run_ts,
                    "metric_name": "silhouette_score",
                    "metric_value": result.silhouette,
                    "baseline_metric_value": None,
                },
            )
            assignments = result.assignments.assign(model_version=model_version)
            conn.execute(UPSERT_STATION_CLUSTER_SQL, assignments.to_dict(orient="records"))

    # Chained rather than run in parallel: this is a local, single-machine
    # tool, and training all three models concurrently multiplies peak memory
    # (RandomForest + IsolationForest + KMeans all resident at once) on top
    # of Airflow's own footprint - sequencing trades some wall-clock time for
    # a much smaller memory ceiling.
    version = check_upstream_data_ready()
    forecast_done = train_demand_forecast_task(version)
    anomaly_done = train_anomaly_detection_task(version)
    clustering_done = train_clustering_task(version)

    forecast_done >> anomaly_done >> clustering_done


train_and_score_models()
