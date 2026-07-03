#!/usr/bin/env bash
# Stops everything (whichever subset is running). Data is preserved in
# Docker volumes - safe to run between demo sessions on a memory-constrained
# machine instead of leaving the stack up continuously.
set -euo pipefail
cd "$(dirname "$0")/.."

docker compose down
echo "Stopped. Data preserved (bikeshare-postgres and airflow-postgres volumes untouched)."
