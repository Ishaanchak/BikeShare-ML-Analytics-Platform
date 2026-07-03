"""Cached, read-only queries against the bikeshare Postgres DB.

The dashboard never recomputes models - every page here only reads
precomputed results written by the train_and_score_models DAG.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st
from sqlalchemy import text

from common.db import get_engine

CACHE_TTL_SECONDS = 60


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_stations() -> pd.DataFrame:
    query = text(
        """
        SELECT station_id, name, lat, lon, capacity
        FROM stations
        WHERE lat IS NOT NULL AND lon IS NOT NULL
        """
    )
    with get_engine().connect() as conn:
        return pd.read_sql(query, conn)


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_latest_status() -> pd.DataFrame:
    query = text(
        """
        SELECT DISTINCT ON (station_id)
            station_id, ts, num_bikes_available, num_docks_available, is_renting, is_returning
        FROM station_status_snapshots
        ORDER BY station_id, ts DESC
        """
    )
    with get_engine().connect() as conn:
        return pd.read_sql(query, conn)


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def data_freshness() -> pd.Timestamp | None:
    query = text("SELECT max(ts) FROM station_status_snapshots")
    with get_engine().connect() as conn:
        return conn.execute(query).scalar_one()


def _latest_model_version(table: str) -> str | None:
    ts_column = "generated_at" if table == "model_predictions" else "computed_at" if table == "station_clusters" else "ts"
    query = text(f"SELECT model_version FROM {table} ORDER BY {ts_column} DESC LIMIT 1")
    with get_engine().connect() as conn:
        row = conn.execute(query).fetchone()
    return row[0] if row else None


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_latest_predictions() -> pd.DataFrame:
    version = _latest_model_version("model_predictions")
    if version is None:
        return pd.DataFrame(columns=["station_id", "target_ts", "predicted_value", "model_version"])
    query = text(
        """
        SELECT station_id, target_ts, predicted_value, model_version
        FROM model_predictions
        WHERE model_version = :version
        """
    )
    with get_engine().connect() as conn:
        return pd.read_sql(query, conn, params={"version": version})


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_latest_anomaly_flags() -> pd.DataFrame:
    version = _latest_model_version("anomaly_flags")
    if version is None:
        return pd.DataFrame(columns=["station_id", "ts", "risk_type", "severity_score", "model_version"])
    query = text(
        """
        SELECT DISTINCT ON (station_id)
            station_id, ts, risk_type, severity_score, model_version
        FROM anomaly_flags
        WHERE model_version = :version
        ORDER BY station_id, ts DESC
        """
    )
    with get_engine().connect() as conn:
        return pd.read_sql(query, conn, params={"version": version})


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_anomaly_history(station_id: str) -> pd.DataFrame:
    query = text(
        """
        SELECT ts, risk_type, severity_score, model_version
        FROM anomaly_flags
        WHERE station_id = :station_id
        ORDER BY ts
        """
    )
    with get_engine().connect() as conn:
        return pd.read_sql(query, conn, params={"station_id": station_id})


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_latest_clusters() -> pd.DataFrame:
    version = _latest_model_version("station_clusters")
    if version is None:
        return pd.DataFrame(columns=["station_id", "cluster_id", "cluster_label", "model_version"])
    query = text(
        """
        SELECT station_id, cluster_id, cluster_label, model_version
        FROM station_clusters
        WHERE model_version = :version
        """
    )
    with get_engine().connect() as conn:
        return pd.read_sql(query, conn, params={"version": version})


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_predictions_for_station(station_id: str) -> pd.DataFrame:
    # Filtered to the latest model_version only - without this, predictions
    # from every past training run (each with its own overlapping test-set
    # date range) would all be mixed into one chart.
    version = _latest_model_version("model_predictions")
    if version is None:
        return pd.DataFrame(columns=["target_ts", "predicted_value", "model_version", "generated_at"])
    query = text(
        """
        SELECT target_ts, predicted_value, model_version, generated_at
        FROM model_predictions
        WHERE station_id = :station_id AND model_version = :version
        ORDER BY target_ts
        """
    )
    with get_engine().connect() as conn:
        return pd.read_sql(query, conn, params={"station_id": station_id, "version": version})


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_actual_net_flow_for_station(station_id: str) -> pd.DataFrame:
    query = text(
        """
        SELECT (date + (hour || ' hours')::interval) AT TIME ZONE 'UTC' AS ts, net_flow
        FROM hourly_station_demand
        WHERE station_id = :station_id
        ORDER BY ts
        """
    )
    with get_engine().connect() as conn:
        return pd.read_sql(query, conn, params={"station_id": station_id})


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_station_usage_summary() -> pd.DataFrame:
    """Lightweight per-station usage aggregates for the cluster scatter plot.

    Deliberately duplicates (a subset of) the read query in
    ml/clustering.py rather than importing that module - the dashboard image
    doesn't need scikit-learn's training path, just these two read-only
    aggregates for visualization.
    """
    # Pre-aggregate to daily grain first - COUNT(DISTINCT date) FILTER (...)
    # over the raw hourly table is dramatically slower (see the same note in
    # ml/clustering.py's _USAGE_QUERY).
    query = text(
        """
        WITH daily AS (
            SELECT station_id,
                   date,
                   sum(departures) AS departures,
                   sum(departures + arrivals) AS activity,
                   (extract(dow FROM date) IN (0, 6)) AS is_weekend
            FROM hourly_station_demand
            GROUP BY station_id, date
        )
        SELECT
            station_id,
            avg(activity) AS avg_trips_per_day,
            avg(departures) FILTER (WHERE NOT is_weekend) AS weekday_avg_departures,
            avg(departures) FILTER (WHERE is_weekend) AS weekend_avg_departures
        FROM daily
        GROUP BY station_id
        """
    )
    with get_engine().connect() as conn:
        return pd.read_sql(query, conn)


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_model_runs(model_name: str) -> pd.DataFrame:
    query = text(
        """
        SELECT run_ts, metric_name, metric_value, baseline_metric_value
        FROM model_runs
        WHERE model_name = :model_name
        ORDER BY run_ts
        """
    )
    with get_engine().connect() as conn:
        return pd.read_sql(query, conn, params={"model_name": model_name})
