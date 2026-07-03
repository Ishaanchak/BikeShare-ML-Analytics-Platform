"""Shared pytest fixtures/path setup for the bikeshare-analytics test suite.

Adds the repo root (for `common`/`ml`) and the Airflow dags directory (for
importing DAG modules directly, e.g. to reuse their SQL statements in
idempotency tests) to sys.path. Handles both layouts: running from the repo
root on the host (bikeshare-analytics/airflow/dags) and running inside the
airflow container, where docker-compose mounts dags at /opt/airflow/dags
directly (no "airflow/" prefix).
"""
from __future__ import annotations

import os
import sys

import pytest
from sqlalchemy.exc import OperationalError

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CANDIDATE_DIRS = [
    _BASE_DIR,
    os.path.join(_BASE_DIR, "airflow", "dags"),  # repo root on the host
    os.path.join(_BASE_DIR, "dags"),  # inside the airflow container
]
for _path in _CANDIDATE_DIRS:
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

from common.db import get_engine  # noqa: E402
from sqlalchemy import text  # noqa: E402


@pytest.fixture(scope="session")
def engine():
    """A live bikeshare-postgres connection, or a skip for DB-dependent tests."""
    eng = get_engine()
    try:
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
    except OperationalError:
        pytest.skip("bikeshare-postgres is not reachable - skipping DB-dependent tests")
    return eng
