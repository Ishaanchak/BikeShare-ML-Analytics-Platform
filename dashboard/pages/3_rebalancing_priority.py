from __future__ import annotations

import streamlit as st

from data import load_latest_anomaly_flags, load_latest_clusters, load_latest_status, load_stations

st.set_page_config(page_title="Rebalancing Priority", page_icon="📋", layout="wide")
st.title("Rebalancing Priority")
st.caption(
    "Stations ranked by current predicted stockout/overflow risk severity - "
    "the actionable view for ops."
)

stations = load_stations()
status = load_latest_status()
flags = load_latest_anomaly_flags()
clusters = load_latest_clusters()

if flags.empty:
    st.info("No anomaly flags yet - run train_and_score_models.")
    st.stop()

df = flags.merge(stations[["station_id", "name", "capacity"]], on="station_id", how="left")
df = df.merge(
    status[["station_id", "num_bikes_available", "num_docks_available"]], on="station_id", how="left"
)
if not clusters.empty:
    df = df.merge(clusters[["station_id", "cluster_label"]], on="station_id", how="left")
else:
    df["cluster_label"] = None
df = df.sort_values("severity_score", ascending=False)

risk_options = sorted(df["risk_type"].dropna().unique().tolist())
risk_filter = st.multiselect("Risk type", risk_options, default=risk_options)
df = df[df["risk_type"].isin(risk_filter)]

st.dataframe(
    df[
        [
            "name",
            "risk_type",
            "severity_score",
            "num_bikes_available",
            "num_docks_available",
            "capacity",
            "cluster_label",
            "ts",
        ]
    ].rename(columns={"name": "station", "ts": "flagged_at"}),
    use_container_width=True,
    hide_index=True,
)
