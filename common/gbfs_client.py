"""GBFS client: feed auto-discovery plus station_information/station_status parsing.

Sub-feed URLs are resolved from the auto-discovery document at call time
(never hardcoded) since GBFS feed paths are versioned and can move.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import requests

DEFAULT_TIMEOUT_SECONDS = 15


@dataclass(frozen=True)
class StationInfo:
    station_id: str
    # short_name is the legacy short station code (e.g. "5329.08"). Historical
    # trip CSVs reference stations by this code, not by station_id, so it's
    # the join key between live GBFS data and common/tripdata_client.py.
    short_name: str | None
    name: str
    lat: float | None
    lon: float | None
    capacity: int | None


@dataclass(frozen=True)
class StationStatus:
    station_id: str
    ts: dt.datetime
    num_bikes_available: int
    num_docks_available: int
    is_renting: bool
    is_returning: bool


def discover_feeds(discovery_url: str, language: str = "en") -> dict[str, str]:
    """Return {feed_name: feed_url} from the GBFS auto-discovery document."""
    resp = requests.get(discovery_url, timeout=DEFAULT_TIMEOUT_SECONDS)
    resp.raise_for_status()
    data = resp.json()["data"]
    lang_block = data.get(language) or next(iter(data.values()))
    return {feed["name"]: feed["url"] for feed in lang_block["feeds"]}


def fetch_station_information(feed_url: str) -> list[StationInfo]:
    resp = requests.get(feed_url, timeout=DEFAULT_TIMEOUT_SECONDS)
    resp.raise_for_status()
    stations = resp.json()["data"]["stations"]
    return [
        StationInfo(
            station_id=str(s["station_id"]),
            short_name=s.get("short_name"),
            name=s["name"],
            lat=s.get("lat"),
            lon=s.get("lon"),
            capacity=s.get("capacity"),
        )
        for s in stations
    ]


def fetch_station_status(feed_url: str) -> list[StationStatus]:
    resp = requests.get(feed_url, timeout=DEFAULT_TIMEOUT_SECONDS)
    resp.raise_for_status()
    stations = resp.json()["data"]["stations"]
    out = []
    for s in stations:
        last_reported = s.get("last_reported")
        ts = (
            dt.datetime.fromtimestamp(last_reported, tz=dt.timezone.utc)
            if last_reported
            else dt.datetime.now(dt.timezone.utc)
        )
        out.append(
            StationStatus(
                station_id=str(s["station_id"]),
                ts=ts,
                num_bikes_available=int(s.get("num_bikes_available", 0)),
                num_docks_available=int(s.get("num_docks_available", 0)),
                is_renting=bool(s.get("is_renting", 0)),
                is_returning=bool(s.get("is_returning", 0)),
            )
        )
    return out


def validate_station_statuses(statuses: list[StationStatus]) -> None:
    """Basic data-quality gate for the ingest DAG: raise if the feed looks broken.

    Row count > 0, no null/empty station_ids, and values within sane bounds -
    intentionally simple thresholds, not a statistical anomaly model (that
    lives in ml/anomaly_detection.py instead).
    """
    if not statuses:
        raise ValueError("station_status feed returned zero rows")
    for s in statuses:
        if not s.station_id:
            raise ValueError("station_status row with a null/empty station_id")
        if s.num_bikes_available < 0 or s.num_docks_available < 0:
            raise ValueError(f"station {s.station_id} has a negative bike/dock count")
        if s.num_bikes_available + s.num_docks_available > 500:
            raise ValueError(
                f"station {s.station_id} reports an implausibly large bikes+docks total"
            )
