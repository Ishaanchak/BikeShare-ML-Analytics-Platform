from __future__ import annotations

import pandas as pd
import pydeck as pdk
import streamlit as st

from data import load_latest_anomaly_flags, load_latest_clusters, load_latest_status, load_stations

st.set_page_config(page_title="Overview / Map", page_icon="🗺️", layout="wide")
st.title("Overview / Map")

stations = load_stations()
status = load_latest_status()
flags = load_latest_anomaly_flags()
clusters = load_latest_clusters()

df = stations.merge(status, on="station_id", how="left")
if not flags.empty:
    df = df.merge(flags[["station_id", "risk_type", "severity_score"]], on="station_id", how="left")
else:
    df["risk_type"], df["severity_score"] = None, None
if not clusters.empty:
    df = df.merge(clusters[["station_id", "cluster_label"]], on="station_id", how="left")
else:
    df["cluster_label"] = None

cluster_options = sorted(df["cluster_label"].dropna().unique().tolist())
if cluster_options:
    selected_clusters = st.multiselect("Filter by usage cluster", cluster_options, default=cluster_options)
    df = df[df["cluster_label"].isin(selected_clusters) | df["cluster_label"].isna()]


def _color(row: pd.Series) -> list[int]:
    if pd.isna(row.get("risk_type")):
        return [100, 149, 237, 160]  # normal: cornflower blue
    severity = row.get("severity_score") or 0.0
    intensity = int(80 + 175 * min(max(severity, 0.0), 1.0))
    if row["risk_type"] == "stockout":
        return [intensity, 30, 30, 200]  # red
    return [230, 140, 20, 200]  # overflow: orange


df["color"] = df.apply(_color, axis=1)

st.caption(f"{len(df)} stations shown. Color = current stockout/overflow risk flag.")

layer = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position=["lon", "lat"],
    get_fill_color="color",
    get_radius=60,
    pickable=True,
)
view_state = pdk.ViewState(
    latitude=float(df["lat"].mean()) if not df.empty else 40.73,
    longitude=float(df["lon"].mean()) if not df.empty else -73.99,
    zoom=11,
)
st.pydeck_chart(
    pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": "{name}\nBikes: {num_bikes_available} / Docks: {num_docks_available}\nRisk: {risk_type}"},
    )
)

legend_col1, legend_col2, legend_col3 = st.columns(3)
legend_col1.markdown("🔵 Normal")
legend_col2.markdown("🔴 Stockout risk")
legend_col3.markdown("🟠 Overflow risk")
