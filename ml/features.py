"""Feature engineering shared across demand forecasting, anomaly detection, and clustering.

Net bike flow (departures - arrivals) is the forecasting target rather than
bikes-available: hourly_station_demand (built from `trips`) has continuous
multi-month history from day one via the backfill DAG, while
station_status_snapshots only accumulates from whenever ingest_live_status
started polling - net_flow is the choice that actually has enough history to
train and evaluate a chronological split against.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
from sqlalchemy import Engine, text

FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "is_weekend",
    "capacity",
    "hist_avg_hour_dow",
    "lag_1h",
    "lag_24h",
    "lag_168h",
]
TARGET_COLUMN = "net_flow"
# Naive baseline: "same value, same hour, last week" is exactly the 168h lag.
BASELINE_COLUMN = "lag_168h"


def load_hourly_demand(engine: Engine, lookback_days: int | None = None) -> pd.DataFrame:
    """One row per (station_id, date, hour), joined with the station's capacity."""
    where_clause = ""
    params: dict = {}
    if lookback_days is not None:
        where_clause = "WHERE hsd.date >= (CURRENT_DATE - :lookback_days * INTERVAL '1 day')"
        params["lookback_days"] = lookback_days
    query = text(
        f"""
        SELECT hsd.station_id, hsd.date, hsd.hour, hsd.departures, hsd.arrivals, hsd.net_flow,
               s.capacity
        FROM hourly_station_demand hsd
        JOIN stations s ON s.station_id = hsd.station_id
        {where_clause}
        ORDER BY hsd.station_id, hsd.date, hsd.hour
        """
    )
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params=params)


def expanding_hour_dow_mean(df: pd.DataFrame, group_keys: list[str], value_col: str) -> pd.Series:
    """Expanding mean of `value_col` within `group_keys`, shifted by one row so a
    row's own value never leaks into its own historical-average feature.

    Shared by the demand forecaster (historical average net_flow) and the
    anomaly detector (historical average bikes-available) - both need "this
    station's typical value for this hour/day-of-week, using only the past".

    Computed via cumsum/cumcount rather than groupby(...).apply(lambda s:
    s.expanding()...): group_keys here is (station_id, hour, day_of_week),
    which produces hundreds of thousands of groups on the full trip history -
    apply()'s per-group Python callback at that cardinality is slow enough to
    make full-history training runs OOM. cumsum/cumcount are vectorized
    groupby reductions (no per-group callback) that give the identical
    result: for each row, (sum of prior rows in its group) / (count of prior
    rows in its group), which is 0/0 = NaN for a group's first row - same as
    expanding().mean().shift(1).
    """
    grouped = df.groupby(group_keys)[value_col]
    prior_sum = grouped.cumsum() - df[value_col]
    prior_count = grouped.cumcount()
    return prior_sum / prior_count


def build_demand_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach time, lag, and historical-average features to an hourly demand frame.

    Expects one row per (station_id, date, hour) as produced by
    load_hourly_demand. All lag/rolling features are computed using only
    values strictly before the current row (shift(1) or later), so nothing
    here leaks future information into a row's own features.
    """
    df = df.copy()
    df["ts"] = pd.to_datetime(df["date"]) + pd.to_timedelta(df["hour"], unit="h")
    df = df.sort_values(["station_id", "ts"]).reset_index(drop=True)

    df["day_of_week"] = df["ts"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    grouped = df.groupby("station_id")["net_flow"]
    df["lag_1h"] = grouped.shift(1)
    df["lag_24h"] = grouped.shift(24)
    df["lag_168h"] = grouped.shift(168)

    df["hist_avg_hour_dow"] = expanding_hour_dow_mean(
        df, ["station_id", "hour", "day_of_week"], "net_flow"
    )

    return df


def train_test_split_chronological(
    df: pd.DataFrame, test_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split on `ts` (never randomly) so the test set is strictly later in time."""
    df = df.sort_values("ts")
    cutoff_idx = int(len(df) * (1 - test_fraction))
    cutoff_ts = df["ts"].iloc[cutoff_idx]
    train = df[df["ts"] < cutoff_ts]
    test = df[df["ts"] >= cutoff_ts]
    return train, test


# --- Live (forward-looking) feature construction -----------------------
#
# build_demand_features above only ever runs on hourly_station_demand, which
# is built from historical trips - there's no keyless live trip-event feed,
# only Citi Bike's periodic monthly batch dumps (see backfill_historical_trips).
# So a genuine live forecast can't use "net flow N hours ago" directly the way
# training does. What IS live is station_status_snapshots (bikes-available,
# polled every ~12 min by ingest_live_status) - build_live_features approximates
# each lag feature as the drop in bikes-available across the corresponding
# hour window (bikes decreasing => net departures => positive, matching
# net_flow's sign convention), falling back to the historical seasonal
# average for any station/window that doesn't have live snapshot coverage
# yet (e.g. lag_168h needs a week of live history that a freshly-deployed
# system won't have).


def _closest_bikes_available(
    engine: Engine, as_of: dt.datetime, tolerance_minutes: int = 20
) -> pd.Series:
    """station_id -> num_bikes_available from the snapshot closest to `as_of`,
    within `tolerance_minutes`. Missing for stations with no snapshot that
    close (e.g. before ingest_live_status had run, or a gap in polling)."""
    query = text(
        """
        SELECT DISTINCT ON (station_id) station_id, num_bikes_available
        FROM station_status_snapshots
        WHERE ts BETWEEN :lo AND :hi
        ORDER BY station_id, ABS(EXTRACT(EPOCH FROM (ts - :as_of)))
        """
    )
    params = {
        "lo": as_of - dt.timedelta(minutes=tolerance_minutes),
        "hi": as_of + dt.timedelta(minutes=tolerance_minutes),
        "as_of": as_of,
    }
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params=params)
    return df.set_index("station_id")["num_bikes_available"]


def _flow_proxy(engine: Engine, window_start: dt.datetime, window_end: dt.datetime) -> pd.Series:
    """Approximate net flow during [window_start, window_end) as the drop in
    bikes-available across the window. NaN (via the outer-join subtraction)
    for a station missing either endpoint snapshot."""
    start_bikes = _closest_bikes_available(engine, window_start)
    end_bikes = _closest_bikes_available(engine, window_end)
    return start_bikes.subtract(end_bikes)  # NaN where either side is missing


def _historical_hour_dow_avg(engine: Engine, hour: int, day_of_week: int) -> pd.Series:
    """station_id -> average net_flow historically observed at this hour and
    day-of-week. EXTRACT(ISODOW ...) is 1=Monday..7=Sunday in Postgres; -1
    converts it to pandas' `.dt.dayofweek` convention (0=Monday..6=Sunday)
    used everywhere else in this module, so the comparison lines up."""
    query = text(
        """
        SELECT station_id, avg(net_flow) AS hist_avg_hour_dow
        FROM hourly_station_demand
        WHERE hour = :hour AND (EXTRACT(ISODOW FROM date)::int - 1) = :day_of_week
        GROUP BY station_id
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"hour": hour, "day_of_week": day_of_week})
    return df.set_index("station_id")["hist_avg_hour_dow"]


def build_live_features(engine: Engine, as_of: dt.datetime | None = None) -> pd.DataFrame:
    """One row per station, feature-engineered for a genuine forward-looking
    prediction rather than a historical backtest. `target_ts` is the next
    top-of-hour after `as_of` (defaults to now) - the hour this row predicts
    net flow for.
    """
    as_of = as_of or dt.datetime.now(dt.timezone.utc)
    target_ts = (as_of + dt.timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

    with engine.connect() as conn:
        stations = pd.read_sql(
            text("SELECT station_id, capacity FROM stations WHERE capacity IS NOT NULL"), conn
        )
    if stations.empty:
        return stations.assign(
            **{c: pd.Series(dtype="float64") for c in ["target_ts", *FEATURE_COLUMNS]}
        )

    hist_avg = _historical_hour_dow_avg(engine, hour=target_ts.hour, day_of_week=target_ts.weekday())

    # lag_1h: the hour immediately before target_ts, i.e. [target_ts - 1h, target_ts)
    # - already (almost) over, so `as_of` stands in for its end (target_ts itself
    # is still in the future). lag_24h/lag_168h windows are fully in the past.
    lag_windows = {
        "lag_1h": (target_ts - dt.timedelta(hours=1), as_of),
        "lag_24h": (target_ts - dt.timedelta(hours=24), target_ts - dt.timedelta(hours=23)),
        "lag_168h": (target_ts - dt.timedelta(hours=168), target_ts - dt.timedelta(hours=167)),
    }

    df = stations.set_index("station_id")
    df["hour"] = target_ts.hour
    df["day_of_week"] = target_ts.weekday()
    df["is_weekend"] = int(target_ts.weekday() in (5, 6))
    df["hist_avg_hour_dow"] = df.index.map(hist_avg)

    for name, (window_start, window_end) in lag_windows.items():
        proxy = _flow_proxy(engine, window_start, window_end)
        df[name] = df.index.map(proxy)

    # Cold-start fallback: no live snapshot coverage yet for this station/lag
    # (e.g. lag_168h needs a week of polling history a fresh deployment won't
    # have) - assume the station's typical value for this hour/day-of-week
    # rather than leaving a gap the model can't predict on. Fill
    # hist_avg_hour_dow's own gaps first, so a station with no history at
    # all for this hour/dow still gets a usable (0.0) fallback for its lags.
    df["hist_avg_hour_dow"] = df["hist_avg_hour_dow"].fillna(0.0)
    for name in lag_windows:
        df[name] = df[name].fillna(df["hist_avg_hour_dow"])

    df["target_ts"] = target_ts
    return df.reset_index().dropna(subset=FEATURE_COLUMNS)
