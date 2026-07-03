"""Backfill historical Citi Bike trip data from the public S3 bucket.

Determines which months within the configured lookback window are missing
from `trips`, downloads + validates + upserts each one (idempotent on
ride_id, safe to retry), then rebuilds `hourly_station_demand` for the
affected date range.
"""
from __future__ import annotations

import datetime as dt
import os

import pandas as pd
from airflow.sdk import dag, task
from sqlalchemy import Engine, text

from common.db import get_engine
from common.tripdata_client import (
    download_month_zip,
    iter_month_csv_chunks,
    list_available_months,
    lookback_window,
    month_date_bounds,
)

TRIPDATA_BASE_URL = os.environ.get("TRIPDATA_BASE_URL", "https://s3.amazonaws.com/tripdata")
LOOKBACK_MONTHS = int(os.environ.get("TRIPDATA_LOOKBACK_MONTHS", "12"))

# Historical trips reference stations by short_name (see the note in
# db/schema.sql). For a short_name not already known (e.g. a decommissioned
# station absent from the live feed), insert a minimal placeholder row keyed
# off a synthetic station_id so it never collides with a real GBFS id.
UPSERT_LEGACY_STATION_SQL = text(
    """
    INSERT INTO stations (station_id, short_name, name, first_seen, last_seen)
    VALUES (:station_id, :short_name, :name, now(), now())
    ON CONFLICT (short_name) DO NOTHING
    """
)

UPSERT_TRIPS_SQL = text(
    """
    INSERT INTO trips
        (ride_id, rideable_type, started_at, ended_at, start_station_id, end_station_id, member_casual)
    VALUES
        (:ride_id, :rideable_type, :started_at, :ended_at, :start_station_id, :end_station_id, :member_casual)
    ON CONFLICT (ride_id) DO NOTHING
    """
)

MARK_MONTH_LOADED_SQL = text(
    """
    INSERT INTO ingested_trip_months (yyyymm, row_count, loaded_at)
    VALUES (:yyyymm, :row_count, now())
    ON CONFLICT (yyyymm) DO UPDATE SET row_count = EXCLUDED.row_count, loaded_at = now()
    """
)

# Rebuilt (not incrementally added) for the affected range: departures and
# arrivals are recomputed from `trips` from scratch and upserted, so calling
# this twice for the same range is a no-op the second time.
REBUILD_HOURLY_DEMAND_SQL = text(
    """
    WITH departures AS (
        SELECT start_station_id AS short_name,
               date_trunc('hour', started_at) AS hour_ts,
               count(*) AS departures
        FROM trips
        WHERE started_at >= :start_date AND started_at < :end_date
          AND start_station_id IS NOT NULL
        GROUP BY start_station_id, date_trunc('hour', started_at)
    ),
    arrivals AS (
        SELECT end_station_id AS short_name,
               date_trunc('hour', ended_at) AS hour_ts,
               count(*) AS arrivals
        FROM trips
        WHERE ended_at >= :start_date AND ended_at < :end_date
          AND end_station_id IS NOT NULL
        GROUP BY end_station_id, date_trunc('hour', ended_at)
    ),
    combined AS (
        SELECT COALESCE(d.short_name, a.short_name) AS short_name,
               COALESCE(d.hour_ts, a.hour_ts) AS hour_ts,
               COALESCE(d.departures, 0) AS departures,
               COALESCE(a.arrivals, 0) AS arrivals
        FROM departures d
        FULL OUTER JOIN arrivals a
            ON d.short_name = a.short_name AND d.hour_ts = a.hour_ts
    )
    INSERT INTO hourly_station_demand (station_id, date, hour, departures, arrivals, net_flow)
    SELECT s.station_id,
           c.hour_ts::date,
           EXTRACT(HOUR FROM c.hour_ts)::smallint,
           c.departures,
           c.arrivals,
           c.departures - c.arrivals
    FROM combined c
    JOIN stations s ON s.short_name = c.short_name
    ON CONFLICT (station_id, date, hour) DO UPDATE SET
        departures = EXCLUDED.departures,
        arrivals = EXCLUDED.arrivals,
        net_flow = EXCLUDED.net_flow
    """
)


def _upsert_legacy_stations(engine: Engine, chunk: pd.DataFrame) -> None:
    station_names: dict[str, str] = {}
    for id_col, name_col in (
        ("start_station_id", "start_station_name"),
        ("end_station_id", "end_station_name"),
    ):
        sub = chunk[[id_col, name_col]].dropna()
        for short_name, name in zip(sub[id_col], sub[name_col]):
            station_names.setdefault(short_name, name)
    if not station_names:
        return
    rows = [
        {"station_id": f"legacy:{short_name}", "short_name": short_name, "name": name}
        for short_name, name in station_names.items()
    ]
    with engine.begin() as conn:
        conn.execute(UPSERT_LEGACY_STATION_SQL, rows)


def _upsert_trips(engine: Engine, chunk: pd.DataFrame) -> int:
    chunk = chunk.astype(object).where(chunk.notna(), None)
    records = chunk.to_dict(orient="records")
    with engine.begin() as conn:
        conn.execute(UPSERT_TRIPS_SQL, records)
    return len(records)


@dag(
    dag_id="backfill_historical_trips",
    schedule="@monthly",
    start_date=dt.datetime(2024, 1, 1),
    catchup=False,
    tags=["ingestion", "trips"],
    # Both this DAG and ingest_live_status write to `stations` (legacy-station
    # upserts here, live station upserts there). Concurrent writes to
    # overlapping rows can hit a Postgres deadlock (one transaction wins, the
    # other is killed) - retries let the killed side just run again rather
    # than failing the whole backfill.
    default_args={"retries": 2, "retry_delay": dt.timedelta(seconds=30)},
)
def backfill_historical_trips():
    @task
    def determine_missing_months() -> list[str]:
        candidates = lookback_window(LOOKBACK_MONTHS)
        available = set(list_available_months(TRIPDATA_BASE_URL))
        engine = get_engine()
        with engine.connect() as conn:
            loaded = {
                row[0]
                for row in conn.execute(text("SELECT yyyymm FROM ingested_trip_months"))
            }
        return [m for m in candidates if m in available and m not in loaded]

    # Serialized (not run in parallel across mapped instances): each month's
    # zip is held fully in memory during download/parse, and this Docker
    # Compose stack runs on a memory-constrained host - loading several
    # multi-hundred-MB months concurrently was observed to OOM-kill a task.
    @task(max_active_tis_per_dagrun=1)
    def load_month(yyyymm: str) -> str:
        zip_bytes = download_month_zip(TRIPDATA_BASE_URL, yyyymm)
        engine = get_engine()
        row_count = 0
        for chunk in iter_month_csv_chunks(zip_bytes):
            _upsert_legacy_stations(engine, chunk)
            row_count += _upsert_trips(engine, chunk)
        with engine.begin() as conn:
            conn.execute(MARK_MONTH_LOADED_SQL, {"yyyymm": yyyymm, "row_count": row_count})
        return yyyymm

    @task
    def rebuild_hourly_demand(loaded_months: list[str]) -> int:
        if not loaded_months:
            return 0
        bounds = [month_date_bounds(m) for m in loaded_months]
        start_date = min(b[0] for b in bounds)
        end_date = max(b[1] for b in bounds)
        with get_engine().begin() as conn:
            result = conn.execute(
                REBUILD_HOURLY_DEMAND_SQL, {"start_date": start_date, "end_date": end_date}
            )
        return result.rowcount

    missing = determine_missing_months()
    loaded = load_month.expand(yyyymm=missing)
    rebuild_hourly_demand(loaded)


backfill_historical_trips()
