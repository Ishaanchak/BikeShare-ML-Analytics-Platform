#!/usr/bin/env bash
# Full startup: brings up Postgres + all of Airflow + the dashboard - use
# this when you specifically want to show the Airflow UI or trigger a DAG
# run, not for routine dashboard viewing (see demo-up.sh for that, which is
# much lighter on an 8GB machine).
#
# DAGs come up paused (see stop.sh / the pause left in place from setup) -
# unpause + trigger manually from the Airflow UI or CLI when you want a run,
# rather than leaving schedules live in the background.
set -euo pipefail
cd "$(dirname "$0")/.."

docker compose up -d

echo
echo "Waiting for Airflow's API server..."
for _ in $(seq 1 60); do
    if curl -sf -o /dev/null http://localhost:8080/api/v2/monitor/health; then
        echo "Airflow ready:   http://localhost:8080  (airflow / airflow)"
        echo "Dashboard ready: http://localhost:8501"
        exit 0
    fi
    sleep 2
done

echo "Airflow didn't come up within 120s - check 'docker compose logs airflow-scheduler'."
exit 1
