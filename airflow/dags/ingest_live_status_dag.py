"""Poll the live Citi Bike GBFS feed and land snapshots in Postgres.

Safe to retry: station upserts are idempotent on station_id, and status
snapshots are inserted with ON CONFLICT (station_id, ts) DO NOTHING, so a
retried task never duplicates rows.
"""
from __future__ import annotations

import datetime as dt
import os

from airflow.sdk import dag, task
from sqlalchemy import text

from common.db import get_engine
from common.gbfs_client import (
    StationStatus,
    discover_feeds,
    fetch_station_information,
    fetch_station_status,
    validate_station_statuses,
)

GBFS_DISCOVERY_URL = os.environ.get(
    "GBFS_DISCOVERY_URL", "https://gbfs.citibikenyc.com/gbfs/gbfs.json"
)

UPSERT_STATIONS_SQL = text(
    """
    INSERT INTO stations (station_id, short_name, name, lat, lon, capacity, first_seen, last_seen)
    VALUES (:station_id, :short_name, :name, :lat, :lon, :capacity, now(), now())
    ON CONFLICT (station_id) DO UPDATE SET
        short_name = EXCLUDED.short_name,
        name = EXCLUDED.name,
        lat = EXCLUDED.lat,
        lon = EXCLUDED.lon,
        capacity = EXCLUDED.capacity,
        last_seen = now()
    """
)

INSERT_SNAPSHOTS_SQL = text(
    """
    INSERT INTO station_status_snapshots
        (station_id, ts, num_bikes_available, num_docks_available, is_renting, is_returning)
    VALUES (:station_id, :ts, :num_bikes_available, :num_docks_available, :is_renting, :is_returning)
    ON CONFLICT (station_id, ts) DO NOTHING
    """
)


@dag(
    dag_id="ingest_live_status",
    schedule=dt.timedelta(minutes=12),
    start_date=dt.datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["ingestion", "gbfs"],
    # Both this DAG and backfill_historical_trips write to `stations`. Under
    # concurrent writes to overlapping rows, Postgres can pick this side as
    # the deadlock victim - a retry just re-runs the (idempotent) upsert.
    default_args={"retries": 2, "retry_delay": dt.timedelta(seconds=30)},
)
def ingest_live_status():
    @task
    def upsert_stations() -> int:
        feeds = discover_feeds(GBFS_DISCOVERY_URL)
        stations = fetch_station_information(feeds["station_information"])
        if not stations:
            raise ValueError("station_information feed returned zero rows")
        with get_engine().begin() as conn:
            conn.execute(
                UPSERT_STATIONS_SQL,
                [
                    {
                        "station_id": s.station_id,
                        "short_name": s.short_name,
                        "name": s.name,
                        "lat": s.lat,
                        "lon": s.lon,
                        "capacity": s.capacity,
                    }
                    for s in stations
                ],
            )
        return len(stations)

    @task
    def fetch_status() -> list[dict]:
        feeds = discover_feeds(GBFS_DISCOVERY_URL)
        statuses = fetch_station_status(feeds["station_status"])
        return [
            {
                "station_id": s.station_id,
                "ts": s.ts.isoformat(),
                "num_bikes_available": s.num_bikes_available,
                "num_docks_available": s.num_docks_available,
                "is_renting": s.is_renting,
                "is_returning": s.is_returning,
            }
            for s in statuses
        ]

    @task
    def quality_check(raw_statuses: list[dict]) -> list[dict]:
        statuses = [
            StationStatus(
                station_id=r["station_id"],
                ts=dt.datetime.fromisoformat(r["ts"]),
                num_bikes_available=r["num_bikes_available"],
                num_docks_available=r["num_docks_available"],
                is_renting=r["is_renting"],
                is_returning=r["is_returning"],
            )
            for r in raw_statuses
        ]
        validate_station_statuses(statuses)  # raises ValueError -> task fails loudly
        return raw_statuses

    @task
    def insert_status_snapshots(raw_statuses: list[dict]) -> int:
        with get_engine().begin() as conn:
            conn.execute(INSERT_SNAPSHOTS_SQL, raw_statuses)
        return len(raw_statuses)

    stations_upserted = upsert_stations()
    validated = quality_check(fetch_status())
    inserted = insert_status_snapshots(validated)

    stations_upserted >> inserted


ingest_live_status()
