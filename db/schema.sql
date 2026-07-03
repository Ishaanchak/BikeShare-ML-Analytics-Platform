-- Bikeshare Analytics application schema.
-- Applied once via bikeshare-postgres's /docker-entrypoint-initdb.d on first
-- container init against an empty data volume. All statements are idempotent
-- (IF NOT EXISTS) so the same file can also be run by hand against an
-- existing database without error.

-- NOTE on station_id vs. short_name: the live GBFS feed's `station_id` (a long
-- Lyft-internal id, sometimes a UUID) is NOT the identifier used in historical
-- trip CSVs. Trip data instead uses Citi Bike's legacy short station code
-- (e.g. "5329.08"), which GBFS also exposes as `short_name` on the same
-- station. `short_name` is what lets `trips` and `hourly_station_demand`
-- resolve to the same physical station as `station_status_snapshots` /
-- `model_predictions` / `anomaly_flags` / `station_clusters` (all keyed on
-- the canonical `station_id`).
CREATE TABLE IF NOT EXISTS stations (
    station_id  TEXT PRIMARY KEY,
    short_name  TEXT UNIQUE,
    name        TEXT NOT NULL,
    lat         DOUBLE PRECISION,
    lon         DOUBLE PRECISION,
    capacity    INTEGER,
    first_seen  TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- station_status_snapshots: one row per station per poll of the live GBFS
-- feed. The unique constraint is what makes re-running ingest_live_status
-- (e.g. on Airflow task retry) safe: a re-insert of the same (station_id, ts)
-- is a no-op via ON CONFLICT DO NOTHING in the loader.
CREATE TABLE IF NOT EXISTS station_status_snapshots (
    id                  BIGSERIAL PRIMARY KEY,
    station_id          TEXT NOT NULL REFERENCES stations(station_id),
    ts                  TIMESTAMPTZ NOT NULL,
    num_bikes_available INTEGER NOT NULL,
    num_docks_available INTEGER NOT NULL,
    is_renting          BOOLEAN NOT NULL,
    is_returning        BOOLEAN NOT NULL,
    CONSTRAINT uq_station_status_station_ts UNIQUE (station_id, ts)
);

CREATE INDEX IF NOT EXISTS ix_station_status_station_ts
    ON station_status_snapshots (station_id, ts);

-- trips: historical rides in the post-2021 Lyft-standard GBFS trip schema.
-- ride_id is already unique per Citi Bike's export, which gives us
-- idempotency on re-ingest for free via ON CONFLICT (ride_id) DO NOTHING.
-- start_station_id/end_station_id hold the legacy short station code used
-- in trip exports, so they reference stations(short_name), not the PK - see
-- the note on the `stations` table above. Nullable because some historical
-- trips reference stations that no longer appear in the live GBFS feed; the
-- tripdata loader upserts a minimal `stations` row (short_name + name only)
-- for any short code it encounters that isn't already known.
CREATE TABLE IF NOT EXISTS trips (
    ride_id           TEXT PRIMARY KEY,
    rideable_type     TEXT,
    started_at        TIMESTAMPTZ NOT NULL,
    ended_at          TIMESTAMPTZ NOT NULL,
    start_station_id  TEXT REFERENCES stations(short_name),
    end_station_id    TEXT REFERENCES stations(short_name),
    member_casual     TEXT
);

CREATE INDEX IF NOT EXISTS ix_trips_start_station_started_at
    ON trips (start_station_id, started_at);
CREATE INDEX IF NOT EXISTS ix_trips_end_station_ended_at
    ON trips (end_station_id, ended_at);

-- ingested_trip_months: explicit record of which monthly source files have
-- been fully loaded. determine_missing_months checks this rather than
-- "does trips have any rows in this month's date range" - Citi Bike's
-- monthly exports have a handful of boundary-overlap rides (e.g. the May
-- file contains a few trips that started April 30), so a row-count-based
-- check can wrongly treat an unloaded month as already done.
CREATE TABLE IF NOT EXISTS ingested_trip_months (
    yyyymm      TEXT PRIMARY KEY,
    row_count   INTEGER NOT NULL,
    loaded_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- hourly_station_demand: aggregated training features, rebuilt from `trips`
-- for the affected date range by the backfill DAG. station_id here is the
-- canonical id (resolved from trips' short_name-based ids via a join
-- through `stations`), so this joins cleanly against station_status_snapshots
-- / model_predictions for feature engineering. The unique constraint lets
-- the rebuild use upsert-on-conflict rather than delete+insert.
CREATE TABLE IF NOT EXISTS hourly_station_demand (
    station_id  TEXT NOT NULL REFERENCES stations(station_id),
    date        DATE NOT NULL,
    hour        SMALLINT NOT NULL CHECK (hour BETWEEN 0 AND 23),
    departures  INTEGER NOT NULL DEFAULT 0,
    arrivals    INTEGER NOT NULL DEFAULT 0,
    net_flow    INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (station_id, date, hour)
);

-- model_predictions: next-hour forecasts written by train_and_score_models.
-- Unique on (station_id, target_ts, model_version) so re-running a training
-- run with the same version identifier doesn't duplicate rows.
CREATE TABLE IF NOT EXISTS model_predictions (
    id              BIGSERIAL PRIMARY KEY,
    station_id      TEXT NOT NULL REFERENCES stations(station_id),
    target_ts       TIMESTAMPTZ NOT NULL,
    predicted_value DOUBLE PRECISION NOT NULL,
    model_version   TEXT NOT NULL,
    generated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_model_predictions UNIQUE (station_id, target_ts, model_version)
);

CREATE INDEX IF NOT EXISTS ix_model_predictions_station_target
    ON model_predictions (station_id, target_ts);

-- anomaly_flags: stockout/overflow risk flags with a severity score rather
-- than a bare boolean, per the spec's "more than a fixed threshold" ask.
CREATE TABLE IF NOT EXISTS anomaly_flags (
    id              BIGSERIAL PRIMARY KEY,
    station_id      TEXT NOT NULL REFERENCES stations(station_id),
    ts              TIMESTAMPTZ NOT NULL,
    risk_type       TEXT NOT NULL CHECK (risk_type IN ('stockout', 'overflow')),
    severity_score  DOUBLE PRECISION NOT NULL,
    model_version   TEXT NOT NULL,
    CONSTRAINT uq_anomaly_flags UNIQUE (station_id, ts, risk_type, model_version)
);

CREATE INDEX IF NOT EXISTS ix_anomaly_flags_station_ts
    ON anomaly_flags (station_id, ts);

-- station_clusters: one row per station per training run (model_version),
-- so cluster history is preserved; the dashboard reads the latest version.
CREATE TABLE IF NOT EXISTS station_clusters (
    station_id      TEXT NOT NULL REFERENCES stations(station_id),
    cluster_id      INTEGER NOT NULL,
    cluster_label   TEXT NOT NULL,
    model_version   TEXT NOT NULL,
    computed_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (station_id, model_version)
);

-- model_runs: forecast accuracy vs. naive baseline over time, plus anomaly
-- and clustering run metadata, all keyed by model_name + run_ts + metric_name.
CREATE TABLE IF NOT EXISTS model_runs (
    id                     BIGSERIAL PRIMARY KEY,
    model_name             TEXT NOT NULL,
    run_ts                 TIMESTAMPTZ NOT NULL DEFAULT now(),
    metric_name            TEXT NOT NULL,
    metric_value           DOUBLE PRECISION NOT NULL,
    baseline_metric_value  DOUBLE PRECISION,
    CONSTRAINT uq_model_runs UNIQUE (model_name, run_ts, metric_name)
);

CREATE INDEX IF NOT EXISTS ix_model_runs_model_run_ts
    ON model_runs (model_name, run_ts);
