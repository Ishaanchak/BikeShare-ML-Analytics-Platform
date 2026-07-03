"""Upsert idempotency: running the same ingest twice must not duplicate rows.

Exercises the actual UPSERT_STATIONS_SQL / INSERT_SNAPSHOTS_SQL statements
from the ingest_live_status DAG against a real Postgres. Skipped (via the
`engine` fixture) if bikeshare-postgres isn't reachable, e.g. running pytest
outside docker compose.
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import text

import ingest_live_status_dag as dag_module

TEST_STATION_ID = "test-idempotency-station"


def _cleanup(engine) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM station_status_snapshots WHERE station_id = :id"),
            {"id": TEST_STATION_ID},
        )
        conn.execute(text("DELETE FROM stations WHERE station_id = :id"), {"id": TEST_STATION_ID})


def test_station_upsert_is_idempotent(engine):
    _cleanup(engine)
    station = {
        "station_id": TEST_STATION_ID,
        "short_name": "test.01",
        "name": "Test Station",
        "lat": 40.0,
        "lon": -74.0,
        "capacity": 10,
    }
    try:
        with engine.begin() as conn:
            conn.execute(dag_module.UPSERT_STATIONS_SQL, [station])
            conn.execute(dag_module.UPSERT_STATIONS_SQL, [station])  # simulated retry

        with engine.connect() as conn:
            count = conn.execute(
                text("SELECT count(*) FROM stations WHERE station_id = :id"),
                {"id": TEST_STATION_ID},
            ).scalar_one()
        assert count == 1
    finally:
        _cleanup(engine)


def test_snapshot_insert_is_idempotent(engine):
    _cleanup(engine)
    with engine.begin() as conn:
        conn.execute(
            dag_module.UPSERT_STATIONS_SQL,
            [
                {
                    "station_id": TEST_STATION_ID,
                    "short_name": None,
                    "name": "Test Station",
                    "lat": None,
                    "lon": None,
                    "capacity": None,
                }
            ],
        )

    snapshot = {
        "station_id": TEST_STATION_ID,
        "ts": dt.datetime(2024, 1, 1, 12, 0, tzinfo=dt.timezone.utc).isoformat(),
        "num_bikes_available": 5,
        "num_docks_available": 5,
        "is_renting": True,
        "is_returning": True,
    }
    try:
        with engine.begin() as conn:
            conn.execute(dag_module.INSERT_SNAPSHOTS_SQL, [snapshot])
            conn.execute(dag_module.INSERT_SNAPSHOTS_SQL, [snapshot])  # simulated retry, same ts

        with engine.connect() as conn:
            count = conn.execute(
                text("SELECT count(*) FROM station_status_snapshots WHERE station_id = :id"),
                {"id": TEST_STATION_ID},
            ).scalar_one()
        assert count == 1
    finally:
        _cleanup(engine)
