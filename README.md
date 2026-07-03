# Bike-Share Operations Analytics Platform

A local, Docker-based operations analytics platform for Citi Bike (NYC's GBFS
bike-share system). It ingests live and historical bike-share data on a
schedule via Airflow, stores it in Postgres, runs a scikit-learn ML pipeline
(demand forecasting, stockout/anomaly detection, station clustering) as a
scheduled job, and serves the results through a Streamlit dashboard.

## Architecture

```
GBFS live feed ─┐                                    ┌─> model_predictions
                ├─> Airflow DAGs ─> Postgres ─> ML ───┼─> anomaly_flags      ─> Streamlit dashboard
Citi Bike S3 ───┘   (ingest/backfill/train)  (scikit) └─> station_clusters
    (trips)
```

- **Airflow** (LocalExecutor) runs three DAGs: `ingest_live_status` polls the
  live GBFS feed every ~12 minutes; `backfill_historical_trips` loads
  historical monthly trip CSVs from Citi Bike's public S3 bucket and rebuilds
  hourly demand aggregates; `train_and_score_models` trains/scores all three
  ML components daily and writes results back to Postgres.
- **Postgres** holds two databases: one for Airflow's own metadata
  (`airflow-postgres`), and one for the application data
  (`bikeshare-postgres`) - stations, trips, snapshots, predictions, flags,
  clusters, and model run history. Schema in [`db/schema.sql`](db/schema.sql).
- **ML pipeline** (`ml/`) trains a demand forecaster (RandomForest, evaluated
  against a naive "same hour, last week" baseline), an anomaly detector
  (IsolationForest over rate-of-change + deviation-from-own-history), and a
  station clusterer (KMeans, k chosen by silhouette score). All three read
  from and write to Postgres only - no notebooks in the shipped pipeline.
- **Streamlit dashboard** (`dashboard/`) is a 4-page app that reads only
  precomputed results from Postgres; it never recomputes models live.

A note on a real-world data quirk this project has to handle: the live GBFS
feed's `station_id` (long, sometimes a UUID) is **not** the identifier used
in historical trip CSVs, which instead use Citi Bike's legacy short station
code (e.g. `"5329.08"`). GBFS also exposes that code as `short_name` on the
same station, which is what lets `trips`/`hourly_station_demand` join
correctly against `station_status_snapshots`/`model_predictions` for the same
physical station - see the comment on the `stations` table in
[`db/schema.sql`](db/schema.sql).

## Day-to-day usage (resource-constrained machines)

This stack is heavier than it looks - Postgres x2, all of Airflow's
components, and the ML training together can exceed what an 8GB machine has
to spare, especially alongside other tools already using memory. Rather than
leaving everything running continuously like a real server would, treat it
as start-for-a-session / stop-when-done:

```bash
./scripts/demo-up.sh   # bikeshare-postgres + dashboard only - for just viewing results
./scripts/full-up.sh   # + all of Airflow - for showing the Airflow UI or running a DAG
./scripts/stop.sh      # stops whichever is running; data is preserved either way
```

`demo-up.sh` is enough for almost everything: the dashboard reads
precomputed results already sitting in Postgres from prior pipeline runs, so
Airflow doesn't need to be running just to look at it - and Airflow's four
components are the heaviest, most crash-prone part of this stack under
memory pressure. Reach for `full-up.sh` only when you actually want to
trigger a DAG or show the scheduler working. All three DAGs are left paused
by default so nothing fires in the background unattended - unpause/trigger
manually when you want a live run.

## Setup

1. Copy `.env.example` to `.env` (defaults work out of the box for local dev).
2. `docker compose up -d airflow-postgres bikeshare-postgres airflow-init` -
   waits for `airflow-init` to finish (creates the Airflow admin user and
   applies `db/schema.sql` to a fresh `bikeshare-postgres` volume).
3. `docker compose up -d` - brings up the rest: Airflow's api-server,
   scheduler, dag-processor, triggerer, and the Streamlit dashboard.
4. Airflow UI: http://localhost:8080 (user/pass: `airflow` / `airflow`,
   overridable in `.env`). Streamlit dashboard: http://localhost:8501.

All three DAGs are paused on creation. To run the initial backfill:

```bash
docker compose exec airflow-scheduler airflow dags unpause ingest_live_status
docker compose exec airflow-scheduler airflow dags unpause backfill_historical_trips
docker compose exec airflow-scheduler airflow dags unpause train_and_score_models

docker compose exec airflow-scheduler airflow dags trigger ingest_live_status
docker compose exec airflow-scheduler airflow dags trigger backfill_historical_trips
# once both have landed data at least once:
docker compose exec airflow-scheduler airflow dags trigger train_and_score_models
```

`backfill_historical_trips` pulls `TRIPDATA_LOOKBACK_MONTHS` (default 12)
trailing months from Citi Bike's public S3 bucket - each month is several
hundred MB zipped, so the first run can take a while depending on bandwidth.
It only loads months not already present in `trips`, so it's safe to re-run.

`train_and_score_models` requires `ingest_live_status` to have produced a
snapshot within the last hour and `trips` to be non-empty; it fails loudly
(rather than training on stale/missing data) if either isn't true yet.

### Running tests

```bash
docker compose run --rm airflow-scheduler pytest /opt/airflow/tests
```

Tests that need a live database (upsert idempotency, model training smoke
test) skip automatically if `bikeshare-postgres` isn't reachable; the rest
(GBFS/trip-data parsing, feature engineering) have no such dependency.

## What this demonstrates

- Scheduled, idempotent data ingestion from two real-world public APIs (a
  live GBFS feed and a historical-data S3 bucket), with retry-safe upserts
  verified by dedicated tests.
- A reproducible, scheduled ML pipeline (not a notebook): feature engineering
  shared across models, chronological train/test splitting to avoid leakage,
  and forecast accuracy benchmarked against a naive baseline with the
  comparison persisted for later inspection.
- More-than-a-threshold anomaly detection (IsolationForest over
  deviation-from-own-history) and data-driven segmentation (KMeans with a
  documented k-selection method and post-hoc cluster labeling).
- A dashboard built for an ops/rebalancing workflow - a map, a per-station
  drill-down, a ranked actionable table, and a model-performance view -
  reading only precomputed state, the way an internal tool actually would.
