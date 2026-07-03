"""Shared SQLAlchemy engine/session helpers for the bikeshare application database.

Deliberately separate from Airflow's own metadata database connection - this
module only ever talks to the `bikeshare-postgres` service.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker


def _database_url() -> str:
    user = os.environ.get("BIKESHARE_DB_USER", "bikeshare")
    password = os.environ.get("BIKESHARE_DB_PASSWORD", "bikeshare")
    host = os.environ.get("BIKESHARE_DB_HOST", "bikeshare-postgres")
    port = os.environ.get("BIKESHARE_DB_PORT", "5432")
    name = os.environ.get("BIKESHARE_DB_NAME", "bikeshare")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return create_engine(_database_url(), pool_pre_ping=True)


@lru_cache(maxsize=1)
def _session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine())


@contextmanager
def get_session() -> Iterator[Session]:
    """Yield a SQLAlchemy session, committing on success and rolling back on error."""
    session = _session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
