"""Smoke tests: the three training functions run without error on a tiny
synthetic sample against a real Postgres.

Not a correctness/accuracy check (see the project README for the real
validation run against live data) - this only guards against the training
path raising on a small input.
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import text

from ml import anomaly_detection, clustering, demand_forecast

TEST_STATIONS = ["smoke-station-1", "smoke-station-2", "smoke-station-3"]
SHORT_NAMES = [f"{sid}-sn" for sid in TEST_STATIONS]


def _cleanup(engine) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM model_predictions WHERE station_id = ANY(:ids)"), {"ids": TEST_STATIONS}
        )
        conn.execute(
            text("DELETE FROM anomaly_flags WHERE station_id = ANY(:ids)"), {"ids": TEST_STATIONS}
        )
        conn.execute(
            text("DELETE FROM station_clusters WHERE station_id = ANY(:ids)"), {"ids": TEST_STATIONS}
        )
        conn.execute(
            text("DELETE FROM hourly_station_demand WHERE station_id = ANY(:ids)"), {"ids": TEST_STATIONS}
        )
        conn.execute(
            text("DELETE FROM station_status_snapshots WHERE station_id = ANY(:ids)"),
            {"ids": TEST_STATIONS},
        )
        conn.execute(text("DELETE FROM trips WHERE start_station_id = ANY(:sn)"), {"sn": SHORT_NAMES})
        conn.execute(text("DELETE FROM stations WHERE station_id = ANY(:ids)"), {"ids": TEST_STATIONS})


def _insert_synthetic_stations(engine) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO stations (station_id, short_name, name, lat, lon, capacity, first_seen, last_seen) "
                "VALUES (:id, :short_name, :name, 40.0, -74.0, 20, now(), now())"
            ),
            [
                {"id": sid, "short_name": f"{sid}-sn", "name": sid}
                for sid in TEST_STATIONS
            ],
        )


def _insert_synthetic_hourly_demand(engine, num_days: int = 10) -> None:
    rows = []
    start = dt.date(2024, 1, 1)
    for offset, sid in enumerate(TEST_STATIONS):
        for day_offset in range(num_days):
            date = start + dt.timedelta(days=day_offset)
            for hour in range(24):
                net_flow = (hour - 12) + offset
                rows.append(
                    {
                        "station_id": sid,
                        "date": date,
                        "hour": hour,
                        "departures": max(net_flow, 0),
                        "arrivals": max(-net_flow, 0),
                        "net_flow": net_flow,
                    }
                )
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO hourly_station_demand (station_id, date, hour, departures, arrivals, net_flow) "
                "VALUES (:station_id, :date, :hour, :departures, :arrivals, :net_flow)"
            ),
            rows,
        )


def _insert_synthetic_snapshots(engine, num_hours: int = 360) -> None:
    rows = []
    start = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    for sid in TEST_STATIONS:
        for i in range(num_hours):
            rows.append(
                {
                    "station_id": sid,
                    "ts": start + dt.timedelta(hours=i),
                    "num_bikes_available": 5 + (i % 7),
                    "num_docks_available": 15 - (i % 7),
                    "is_renting": True,
                    "is_returning": True,
                }
            )
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO station_status_snapshots "
                "(station_id, ts, num_bikes_available, num_docks_available, is_renting, is_returning) "
                "VALUES (:station_id, :ts, :num_bikes_available, :num_docks_available, :is_renting, :is_returning)"
            ),
            rows,
        )


def _insert_synthetic_trips(engine) -> None:
    rows = []
    start = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    for sid in TEST_STATIONS:
        for j in range(3):
            rows.append(
                {
                    "ride_id": f"smoke-ride-{sid}-{j}",
                    "rideable_type": "classic_bike",
                    "started_at": start + dt.timedelta(hours=j),
                    "ended_at": start + dt.timedelta(hours=j, minutes=10),
                    "start_station_id": f"{sid}-sn",
                    "end_station_id": f"{sid}-sn",
                    "member_casual": "member",
                }
            )
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO trips "
                "(ride_id, rideable_type, started_at, ended_at, start_station_id, end_station_id, member_casual) "
                "VALUES (:ride_id, :rideable_type, :started_at, :ended_at, :start_station_id, :end_station_id, :member_casual)"
            ),
            rows,
        )


def test_predict_live_runs_without_error_after_training(engine):
    """predict_live loads whichever model_version most recently wrote to
    model_predictions and scores it against current conditions - smoke-tests
    that path end to end, including the cold-start fallback (the synthetic
    snapshots here don't go back a full week, so lag_168h has to fall back to
    the historical average).

    train_and_score itself doesn't write to model_predictions (the DAG task
    does that) - so this writes a single marker row under this run's
    model_version, to make it the one predict_live's load_latest_model
    picks up (it only reads model_version, not the row's other columns).

    Deliberately NOT result.predictions here: train_and_score runs its
    chronological split against the *entire* hourly_station_demand table,
    which in a real (non-test) database has months of real-station history
    dated long after our synthetic stations' 2024 rows - so the test set,
    and therefore result.predictions, is real stations' data, not our
    synthetic ones. Writing all of it under a "smoke-test-live" tag would
    dump production-scale prediction rows into a shared table with no
    station-scoped cleanup to catch them.
    """
    _cleanup(engine)
    try:
        _insert_synthetic_stations(engine)
        _insert_synthetic_hourly_demand(engine)
        _insert_synthetic_snapshots(engine)

        demand_forecast.train_and_score(engine, model_version="smoke-test-live")
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO model_predictions "
                    "(station_id, target_ts, predicted_value, model_version, generated_at) "
                    "VALUES (:station_id, now(), 0.0, :model_version, now())"
                ),
                {"station_id": TEST_STATIONS[0], "model_version": "smoke-test-live"},
            )

        predictions = demand_forecast.predict_live(engine)

        assert not predictions.empty
        assert set(TEST_STATIONS).issubset(set(predictions["station_id"]))
        assert (predictions["model_version"] == "smoke-test-live").all()
        # target_ts is a real future hour, not a historical backtest point.
        assert (predictions["target_ts"] > dt.datetime.now(dt.timezone.utc)).all()
    finally:
        _cleanup(engine)


def test_training_functions_run_without_error_on_tiny_sample(engine):
    _cleanup(engine)
    try:
        _insert_synthetic_stations(engine)
        _insert_synthetic_hourly_demand(engine)
        _insert_synthetic_snapshots(engine)
        _insert_synthetic_trips(engine)

        forecast_result = demand_forecast.train_and_score(engine, model_version="smoke-test")
        assert forecast_result.model_mae >= 0
        assert len(forecast_result.predictions) > 0

        anomaly_result = anomaly_detection.train_and_score(
            engine, model_version="smoke-test", lookback_days=3650
        )
        assert anomaly_result.num_flags >= 0

        cluster_result = clustering.train_and_score(
            engine, model_version="smoke-test", k_candidates=range(2, 3)
        )
        assert cluster_result.chosen_k == 2
        # clustering.train_and_score has no station filter - it clusters
        # every station with usage history, not just our synthetic ones -
        # so just confirm ours got assigned, not that they're the only rows.
        assigned_station_ids = set(cluster_result.assignments["station_id"])
        assert set(TEST_STATIONS).issubset(assigned_station_ids)
    finally:
        _cleanup(engine)
