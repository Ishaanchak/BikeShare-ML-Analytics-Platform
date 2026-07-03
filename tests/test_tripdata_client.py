"""Trip-data CSV/zip parsing tests, including expected-schema validation and
month-window helpers.
"""
from __future__ import annotations

import datetime as dt
import io
import zipfile
from unittest.mock import MagicMock, patch

import pytest

from common.tripdata_client import (
    EXPECTED_COLUMNS,
    iter_month_csv_chunks,
    list_available_months,
    lookback_window,
    month_date_bounds,
)


def _make_zip(csv_bodies: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, body in csv_bodies.items():
            zf.writestr(name, body)
    return buf.getvalue()


GOOD_HEADER = ",".join(EXPECTED_COLUMNS)
GOOD_ROW = (
    "r1,classic_bike,2024-01-01 00:00:00,2024-01-01 00:10:00,"
    "A St,1.01,B St,2.01,40.7,-73.9,40.8,-74.0,member"
)


def test_iter_month_csv_chunks_parses_valid_csv():
    zip_bytes = _make_zip({"202401-citibike-tripdata_1.csv": f"{GOOD_HEADER}\n{GOOD_ROW}\n"})
    chunks = list(iter_month_csv_chunks(zip_bytes, chunksize=10))
    assert len(chunks) == 1
    assert list(chunks[0].columns) == EXPECTED_COLUMNS
    assert len(chunks[0]) == 1


def test_iter_month_csv_chunks_skips_macosx_and_reads_multiple_parts():
    zip_bytes = _make_zip(
        {
            "__MACOSX/._202401-citibike-tripdata_1.csv": "junk,not,real,csv",
            "202401-citibike-tripdata_1.csv": f"{GOOD_HEADER}\n{GOOD_ROW}\n",
            "202401-citibike-tripdata_2.csv": f"{GOOD_HEADER}\n{GOOD_ROW}\n",
        }
    )
    chunks = list(iter_month_csv_chunks(zip_bytes, chunksize=10))
    assert sum(len(c) for c in chunks) == 2


def test_iter_month_csv_chunks_raises_on_unexpected_schema():
    zip_bytes = _make_zip({"bad.csv": "wrong,columns\n1,2\n"})
    with pytest.raises(ValueError, match="unexpected columns"):
        list(iter_month_csv_chunks(zip_bytes, chunksize=10))


def test_iter_month_csv_chunks_raises_when_zip_has_no_csv_entries():
    zip_bytes = _make_zip({"__MACOSX/._junk": "junk"})
    with pytest.raises(ValueError, match="no CSV entries"):
        list(iter_month_csv_chunks(zip_bytes, chunksize=10))


def test_lookback_window_returns_trailing_full_months_before_as_of():
    months = lookback_window(3, as_of=dt.date(2026, 3, 15))
    assert months == ["202512", "202601", "202602"]


def test_month_date_bounds_handles_december_rollover():
    start, end = month_date_bounds("202512")
    assert start == dt.date(2025, 12, 1)
    assert end == dt.date(2026, 1, 1)


def test_list_available_months_parses_bucket_listing_and_ignores_non_month_keys():
    xml = (
        "<ListBucketResult>"
        "<Contents><Key>202401-citibike-tripdata.zip</Key></Contents>"
        "<Contents><Key>202402-citibike-tripdata.zip</Key></Contents>"
        "<Contents><Key>JC-202401-citibike-tripdata.zip</Key></Contents>"
        "</ListBucketResult>"
    )
    resp = MagicMock()
    resp.text = xml
    resp.raise_for_status.return_value = None
    with patch("common.tripdata_client.requests.get", return_value=resp):
        months = list_available_months("http://x/tripdata")
    assert months == ["202401", "202402"]
