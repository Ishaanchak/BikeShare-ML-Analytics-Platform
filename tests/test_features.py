"""Feature engineering tests on a small synthetic hourly-demand dataset.

net_flow = hour + day_offset * 100, so lag_24h and lag_168h (7 days) never
collide with the same value by coincidence, and every assertion here checks
that a feature reads from the specific prior row it's supposed to - not just
"some" prior row.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
from sqlalchemy import text

from ml.features import (
    BASELINE_COLUMN,
    FEATURE_COLUMNS,
    build_demand_features,
    build_live_features,
    train_test_split_chronological,
)


def _synthetic_hourly_demand(num_days: int = 10) -> pd.DataFrame:
    rows = []
    start = dt.date(2024, 1, 1)
    for day_offset in range(num_days):
        date = start + dt.timedelta(days=day_offset)
        for hour in range(24):
            net_flow = hour + day_offset * 100
            rows.append(
                {
                    "station_id": "A",
                    "date": date,
                    "hour": hour,
                    "departures": net_flow,
                    "arrivals": 0,
                    "net_flow": net_flow,
                    "capacity": 20,
                }
            )
    return pd.DataFrame(rows)


def test_build_demand_features_first_row_has_no_lag_or_history():
    df = build_demand_features(_synthetic_hourly_demand())
    first_row = df.iloc[0]
    assert pd.isna(first_row["lag_1h"])
    assert pd.isna(first_row["lag_24h"])
    assert pd.isna(first_row["lag_168h"])
    assert pd.isna(first_row["hist_avg_hour_dow"])


def test_lag_1h_reads_the_immediately_prior_hour():
    df = build_demand_features(_synthetic_hourly_demand())
    assert df.iloc[1]["lag_1h"] == df.iloc[0]["net_flow"]


def test_lag_24h_reads_the_same_hour_one_day_earlier():
    df = build_demand_features(_synthetic_hourly_demand())
    # index 24 = day_offset=1, hour=0; its lag_24h should be day_offset=0, hour=0.
    assert df.iloc[24]["lag_24h"] == df.iloc[0]["net_flow"]


def test_lag_168h_is_the_naive_same_hour_last_week_baseline():
    df = build_demand_features(_synthetic_hourly_demand())
    # index 168 = day_offset=7, hour=0; lag_168h should be day_offset=0, hour=0 -
    # distinct from lag_24h (day_offset=6, hour=0) since net_flow varies by day.
    row = df.iloc[168]
    assert row["lag_168h"] == df.iloc[0]["net_flow"]
    assert row["lag_168h"] != row["lag_24h"]
    assert row[BASELINE_COLUMN] == row["lag_168h"]


def test_hist_avg_hour_dow_is_expanding_mean_of_strictly_prior_occurrences():
    # Same hour AND same day-of-week recurs every 7 days, not every day -
    # need enough days to get 3 occurrences of one (hour, dow) combination.
    df = build_demand_features(_synthetic_hourly_demand(num_days=22))
    target_dow = df.iloc[0]["day_of_week"]
    same_hour_dow = df[(df["hour"] == 5) & (df["day_of_week"] == target_dow)].reset_index(drop=True)
    assert len(same_hour_dow) >= 3

    # 1st occurrence: no prior data at all.
    assert pd.isna(same_hour_dow.iloc[0]["hist_avg_hour_dow"])

    # 2nd occurrence: only one prior value (the 1st) to average.
    assert same_hour_dow.iloc[1]["hist_avg_hour_dow"] == same_hour_dow.iloc[0]["net_flow"]

    # 3rd occurrence: mean of the first two occurrences' net_flow.
    expected = (same_hour_dow.iloc[0]["net_flow"] + same_hour_dow.iloc[1]["net_flow"]) / 2
    assert same_hour_dow.iloc[2]["hist_avg_hour_dow"] == expected


def test_train_test_split_is_chronological_not_random():
    df = build_demand_features(_synthetic_hourly_demand())
    train, test = train_test_split_chronological(df, test_fraction=0.2)
    assert len(train) + len(test) == len(df)
    assert train["ts"].max() < test["ts"].min()


# --- build_live_features (needs a real Postgres - uses the `engine` fixture) ---

LIVE_TEST_STATION = "livetest-station-1"


def _cleanup_live(engine) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM station_status_snapshots WHERE station_id = :id"),
            {"id": LIVE_TEST_STATION},
        )
        conn.execute(
            text("DELETE FROM hourly_station_demand WHERE station_id = :id"), {"id": LIVE_TEST_STATION}
        )
        conn.execute(text("DELETE FROM stations WHERE station_id = :id"), {"id": LIVE_TEST_STATION})


def test_build_live_features_computes_lag_1h_from_recent_snapshots_and_falls_back_for_the_rest(engine):
    _cleanup_live(engine)
    try:
        as_of = dt.datetime(2024, 3, 6, 14, 30, tzinfo=dt.timezone.utc)  # a Wednesday
        target_ts = dt.datetime(2024, 3, 6, 15, 0, tzinfo=dt.timezone.utc)

        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO stations (station_id, short_name, name, lat, lon, capacity, first_seen, last_seen) "
                    "VALUES (:id, :id, :id, 40.0, -74.0, 20, now(), now())"
                ),
                {"id": LIVE_TEST_STATION},
            )
            # Historical seasonal average for this exact (hour=15, Wednesday):
            # two prior Wednesdays at net_flow 10 and 20 -> average 15.
            conn.execute(
                text(
                    "INSERT INTO hourly_station_demand (station_id, date, hour, departures, arrivals, net_flow) "
                    "VALUES (:sid, :date, 15, :dep, :arr, :net)"
                ),
                [
                    {"sid": LIVE_TEST_STATION, "date": dt.date(2024, 2, 21), "dep": 10, "arr": 0, "net": 10},
                    {"sid": LIVE_TEST_STATION, "date": dt.date(2024, 2, 28), "dep": 20, "arr": 0, "net": 20},
                ],
            )
            # Live snapshots covering only the lag_1h window: bikes dropped
            # from 12 to 5 over the last hour -> 7 net departures (positive).
            conn.execute(
                text(
                    "INSERT INTO station_status_snapshots "
                    "(station_id, ts, num_bikes_available, num_docks_available, is_renting, is_returning) "
                    "VALUES (:sid, :ts, :bikes, 5, true, true)"
                ),
                [
                    {"sid": LIVE_TEST_STATION, "ts": target_ts - dt.timedelta(hours=1), "bikes": 12},
                    {"sid": LIVE_TEST_STATION, "ts": as_of, "bikes": 5},
                ],
            )

        features = build_live_features(engine, as_of=as_of)
        row = features[features["station_id"] == LIVE_TEST_STATION].iloc[0]

        assert row["target_ts"] == pd.Timestamp(target_ts)
        assert row["hour"] == 15
        assert row["day_of_week"] == 2  # Wednesday, pandas/Python convention (Monday=0)
        assert row["is_weekend"] == 0
        assert row["hist_avg_hour_dow"] == 15.0  # (10 + 20) / 2

        # lag_1h has live coverage: bikes dropped 12 -> 5 => net departures of 7.
        assert row["lag_1h"] == 7.0
        # lag_24h/lag_168h have no snapshot coverage at all -> fall back to
        # the historical seasonal average rather than staying NaN.
        assert row["lag_24h"] == 15.0
        assert row["lag_168h"] == 15.0
        assert not features[FEATURE_COLUMNS].isna().any().any()
    finally:
        _cleanup_live(engine)
