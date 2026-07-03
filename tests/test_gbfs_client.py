"""GBFS parsing tests: auto-discovery, station_information/station_status
parsing, and the ingest data-quality gate - all against mocked HTTP
responses, no network calls.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from common.gbfs_client import (
    StationStatus,
    discover_feeds,
    fetch_station_information,
    fetch_station_status,
    validate_station_statuses,
)


def _mock_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


def test_discover_feeds_picks_requested_language():
    payload = {
        "data": {
            "en": {"feeds": [{"name": "station_information", "url": "http://x/en/info.json"}]},
            "fr": {"feeds": [{"name": "station_information", "url": "http://x/fr/info.json"}]},
        }
    }
    with patch("common.gbfs_client.requests.get", return_value=_mock_response(payload)):
        feeds = discover_feeds("http://x/gbfs.json", language="fr")
    assert feeds == {"station_information": "http://x/fr/info.json"}


def test_discover_feeds_falls_back_when_requested_language_missing():
    payload = {"data": {"es": {"feeds": [{"name": "station_status", "url": "http://x/status.json"}]}}}
    with patch("common.gbfs_client.requests.get", return_value=_mock_response(payload)):
        feeds = discover_feeds("http://x/gbfs.json", language="en")
    assert feeds == {"station_status": "http://x/status.json"}


def test_fetch_station_information_parses_fields():
    payload = {
        "data": {
            "stations": [
                {
                    "station_id": "123",
                    "short_name": "1.01",
                    "name": "Test St",
                    "lat": 40.7,
                    "lon": -73.9,
                    "capacity": 20,
                }
            ]
        }
    }
    with patch("common.gbfs_client.requests.get", return_value=_mock_response(payload)):
        stations = fetch_station_information("http://x/info.json")
    assert len(stations) == 1
    station = stations[0]
    assert station.station_id == "123"
    assert station.short_name == "1.01"
    assert station.capacity == 20


def test_fetch_station_status_converts_ints_to_bool_and_epoch_to_datetime():
    payload = {
        "data": {
            "stations": [
                {
                    "station_id": "123",
                    "num_bikes_available": 5,
                    "num_docks_available": 10,
                    "is_renting": 1,
                    "is_returning": 0,
                    "last_reported": 1_700_000_000,
                }
            ]
        }
    }
    with patch("common.gbfs_client.requests.get", return_value=_mock_response(payload)):
        statuses = fetch_station_status("http://x/status.json")
    assert len(statuses) == 1
    status = statuses[0]
    assert status.is_renting is True
    assert status.is_returning is False
    assert status.ts.year == 2023


def test_validate_station_statuses_rejects_empty_feed():
    with pytest.raises(ValueError, match="zero rows"):
        validate_station_statuses([])


def test_validate_station_statuses_rejects_null_station_id():
    bad = StationStatus(
        station_id="",
        ts=None,
        num_bikes_available=1,
        num_docks_available=1,
        is_renting=True,
        is_returning=True,
    )
    with pytest.raises(ValueError, match="null/empty station_id"):
        validate_station_statuses([bad])


def test_validate_station_statuses_rejects_negative_counts():
    bad = StationStatus(
        station_id="1",
        ts=None,
        num_bikes_available=-1,
        num_docks_available=1,
        is_renting=True,
        is_returning=True,
    )
    with pytest.raises(ValueError, match="negative"):
        validate_station_statuses([bad])


def test_validate_station_statuses_rejects_implausible_totals():
    bad = StationStatus(
        station_id="1",
        ts=None,
        num_bikes_available=300,
        num_docks_available=300,
        is_renting=True,
        is_returning=True,
    )
    with pytest.raises(ValueError, match="implausibly large"):
        validate_station_statuses([bad])


def test_validate_station_statuses_accepts_sane_feed():
    good = StationStatus(
        station_id="1",
        ts=None,
        num_bikes_available=5,
        num_docks_available=10,
        is_renting=True,
        is_returning=True,
    )
    validate_station_statuses([good])  # should not raise
