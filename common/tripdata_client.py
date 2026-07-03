"""Citi Bike historical trip data client.

Lists available months on the public S3 bucket, downloads/unzips a given
month, and parses its CSV member(s) against the expected post-2021
Lyft-standard schema. Each month's zip may contain more than one CSV part
(and, on some months, stray macOS metadata entries) - all real CSV members
are parsed; anything not matching the expected header raises loudly rather
than silently loading partial data.
"""
from __future__ import annotations

import datetime as dt
import io
import re
import zipfile
from typing import Iterator

import pandas as pd
import requests

DEFAULT_TIMEOUT_SECONDS = 60
DOWNLOAD_TIMEOUT_SECONDS = 300

EXPECTED_COLUMNS = [
    "ride_id",
    "rideable_type",
    "started_at",
    "ended_at",
    "start_station_name",
    "start_station_id",
    "end_station_name",
    "end_station_id",
    "start_lat",
    "start_lng",
    "end_lat",
    "end_lng",
    "member_casual",
]

# Station id/name columns must stay strings - some legacy station codes
# (e.g. "5329.10") would otherwise be inferred as float64 and lose trailing
# zeros on the way back to text, silently corrupting the join key.
_STRING_COLUMNS = {
    "ride_id",
    "rideable_type",
    "start_station_name",
    "start_station_id",
    "end_station_name",
    "end_station_id",
    "member_casual",
}

_MONTH_KEY_RE = re.compile(r"^(\d{6})-citibike-tripdata\.zip$")


def list_available_months(base_url: str) -> list[str]:
    """Return sorted YYYYMM strings for every top-level month zip in the bucket."""
    resp = requests.get(
        base_url, params={"list-type": "2"}, timeout=DEFAULT_TIMEOUT_SECONDS
    )
    resp.raise_for_status()
    months = {
        m.group(1)
        for m in (_MONTH_KEY_RE.match(key) for key in re.findall(r"<Key>(.*?)</Key>", resp.text))
        if m
    }
    return sorted(months)


def lookback_window(lookback_months: int, as_of: dt.date | None = None) -> list[str]:
    """Trailing `lookback_months` YYYYMM strings for full months before `as_of`."""
    as_of = as_of or dt.date.today()
    year, month = as_of.year, as_of.month
    out = []
    for _ in range(lookback_months):
        month -= 1
        if month == 0:
            month, year = 12, year - 1
        out.append(f"{year:04d}{month:02d}")
    return sorted(out)


def month_date_bounds(yyyymm: str) -> tuple[dt.date, dt.date]:
    """[start, end) date bounds for a YYYYMM string."""
    year, month = int(yyyymm[:4]), int(yyyymm[4:6])
    start = dt.date(year, month, 1)
    end = dt.date(year + (month == 12), (month % 12) + 1, 1)
    return start, end


def download_month_zip(base_url: str, yyyymm: str) -> bytes:
    url = f"{base_url.rstrip('/')}/{yyyymm}-citibike-tripdata.zip"
    resp = requests.get(url, timeout=DOWNLOAD_TIMEOUT_SECONDS)
    resp.raise_for_status()
    return resp.content


def iter_month_csv_chunks(zip_bytes: bytes, chunksize: int = 50_000) -> Iterator[pd.DataFrame]:
    """Yield DataFrame chunks from every real CSV member of a month's zip.

    Raises ValueError on the first chunk whose header doesn't match
    EXPECTED_COLUMNS, so a malformed/mismatched file fails the task loudly.
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        csv_names = sorted(
            n
            for n in zf.namelist()
            if n.lower().endswith(".csv") and "__MACOSX" not in n and not n.endswith("/")
        )
        if not csv_names:
            raise ValueError("month zip contains no CSV entries")
        for name in csv_names:
            # Check the header before handing the file to read_csv: with
            # parse_dates=["started_at", "ended_at"], pandas raises its own
            # (confusing) "Missing column provided to 'parse_dates'" error
            # for a mismatched schema instead of ever reaching the header
            # check below, if those exact columns aren't present.
            with zf.open(name) as header_check:
                header_cols = header_check.readline().decode("utf-8", errors="replace").strip().split(",")
            if header_cols != EXPECTED_COLUMNS:
                raise ValueError(f"{name} has unexpected columns: {header_cols}")

            with zf.open(name) as f:
                reader = pd.read_csv(
                    f,
                    chunksize=chunksize,
                    parse_dates=["started_at", "ended_at"],
                    dtype={col: str for col in _STRING_COLUMNS},
                )
                yield from reader
