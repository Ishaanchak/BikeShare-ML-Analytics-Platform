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
  
## Day-to-day usage (resource-constrained machines)

For usage in its current state: treat as start-for-a-session / stop-when-done:

```bash
./scripts/demo-up.sh   # bikeshare-postgres + dashboard only - for just viewing results
./scripts/full-up.sh   # + all of Airflow - for showing the Airflow UI or running a DAG
./scripts/stop.sh      # stops whichever is running; data is preserved either way
```
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
