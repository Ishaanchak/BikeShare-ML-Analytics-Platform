from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data import (
    load_actual_net_flow_for_station,
    load_anomaly_history,
    load_latest_clusters,
    load_latest_status,
    load_predictions_for_station,
    load_stations,
)

st.set_page_config(page_title="Station Detail", page_icon="🔍", layout="wide")
st.title("Station Detail")

stations = load_stations().sort_values("name")
if stations.empty:
    st.info("No stations yet - run ingest_live_status.")
    st.stop()

station_name = st.selectbox("Station", stations["name"])
station_id = stations.loc[stations["name"] == station_name, "station_id"].iloc[0]

status = load_latest_status()
live = status[status["station_id"] == station_id]
clusters = load_latest_clusters()
cluster_row = clusters[clusters["station_id"] == station_id] if not clusters.empty else clusters

col1, col2, col3, col4 = st.columns(4)
if not live.empty:
    live_row = live.iloc[0]
    col1.metric("Bikes available", int(live_row["num_bikes_available"]))
    col2.metric("Docks available", int(live_row["num_docks_available"]))
    col3.metric("Last reported", pd.Timestamp(live_row["ts"]).strftime("%Y-%m-%d %H:%M UTC"))
else:
    col1.metric("Bikes available", "n/a")
    col2.metric("Docks available", "n/a")
    col3.metric("Last reported", "n/a")
col4.metric("Usage cluster", cluster_row.iloc[0]["cluster_label"] if not cluster_row.empty else "n/a")

st.subheader("Predicted vs. actual net flow (departures - arrivals)")
st.caption("Net flow, not bikes-available, is the forecasting target - see ml/features.py for why.")
predictions = load_predictions_for_station(station_id)
actual = load_actual_net_flow_for_station(station_id)

if predictions.empty:
    st.info("No predictions yet for this station - run train_and_score_models.")
else:
    now = pd.Timestamp.now(tz="UTC")
    # target_ts > now => a genuine forward-looking forecast from
    # predict_live_demand, not yet comparable to an "actual" (hasn't
    # happened). Everything else is the train_and_score_models backtest.
    live = predictions[predictions["target_ts"] > now]
    historical = predictions[predictions["target_ts"] <= now]

    if not live.empty:
        live_row = live.iloc[-1]
        st.metric(
            "Live forecast - next hour",
            f"{live_row['predicted_value']:+.1f} bikes/hr",
            help=(
                f"Predicted at {pd.Timestamp(live_row['generated_at']).strftime('%Y-%m-%d %H:%M UTC')} "
                f"for the hour starting {pd.Timestamp(live_row['target_ts']).strftime('%Y-%m-%d %H:%M UTC')}. "
                "Positive = net departures (draining); negative = net arrivals (filling). "
                "From predict_live_demand, scored against current conditions - not a backtest."
            ),
        )

    merged = historical.merge(actual, left_on="target_ts", right_on="ts", how="left")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged["target_ts"], y=merged["net_flow"], name="actual"))
    fig.add_trace(go.Scatter(x=merged["target_ts"], y=merged["predicted_value"], name="predicted (backtest)"))
    if not live.empty:
        fig.add_trace(
            go.Scatter(
                x=live["target_ts"],
                y=live["predicted_value"],
                name="predicted (live, next hour)",
                mode="markers",
                marker=dict(size=14, symbol="star", color="gold", line=dict(width=1, color="black")),
            )
        )
    fig.update_layout(xaxis_title="time", yaxis_title="net flow (bikes/hour)", height=400)
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Anomaly flag timeline")
history = load_anomaly_history(station_id)
if history.empty:
    st.info("No anomaly flags recorded yet for this station.")
else:
    st.dataframe(history, use_container_width=True, hide_index=True)
