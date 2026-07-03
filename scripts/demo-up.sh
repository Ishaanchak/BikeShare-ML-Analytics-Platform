#!/usr/bin/env bash
# Lightweight startup for "just look at the dashboard": brings up only
# bikeshare-postgres + the Streamlit dashboard, skipping Airflow entirely.
# All the data/models the dashboard reads already exist in bikeshare-postgres
# from prior pipeline runs, so Airflow isn't needed just to view results -
# and it's the heaviest, most crash-prone part of this stack on an 8GB
# machine. Use full-up.sh instead if you specifically want to show the
# Airflow UI or trigger a DAG run.
set -euo pipefail
cd "$(dirname "$0")/.."

docker compose up -d bikeshare-postgres dashboard

echo
echo "Waiting for the dashboard to come up..."
for _ in $(seq 1 30); do
    if curl -sf -o /dev/null http://localhost:8501; then
        echo "Dashboard ready: http://localhost:8501"
        exit 0
    fi
    sleep 2
done

echo "Dashboard didn't come up within 60s - check 'docker compose logs dashboard'."
exit 1
