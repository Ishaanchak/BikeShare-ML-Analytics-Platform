from __future__ import annotations

import streamlit as st

from data import data_freshness, load_stations

st.set_page_config(page_title="Citi Bike Ops Analytics", page_icon="🚲", layout="wide")

st.title("Citi Bike Ops Analytics")
st.caption("Local ops analytics platform: GBFS + historical trips -> Postgres -> scikit-learn -> this dashboard.")

col1, col2, col3 = st.columns(3)

try:
    stations = load_stations()
    col1.metric("Stations tracked", len(stations))
except Exception as exc:  # noqa: BLE001 - surface DB connectivity issues plainly on the landing page
    st.error(f"Could not reach the database: {exc}")
    st.stop()

freshness = data_freshness()
if freshness is not None:
    col2.metric("Latest live snapshot", freshness.strftime("%Y-%m-%d %H:%M UTC"))
else:
    col2.metric("Latest live snapshot", "no data yet")

col3.metric("Data source", "Live GBFS + S3 trip history")

st.markdown(
    """
Use the pages in the sidebar:

- **Overview / Map** - all stations, color-coded by current stockout/overflow risk, filterable by usage cluster.
- **Station Detail** - live status, predicted vs. actual demand, cluster label, and anomaly history for one station.
- **Rebalancing Priority** - stations ranked by predicted risk in the next hour - the actionable view.
- **Model Performance** - forecast accuracy vs. a naive baseline, cluster visualization, and data freshness.

Every page here reads precomputed results from Postgres; nothing is recomputed live in the browser.
"""
)
