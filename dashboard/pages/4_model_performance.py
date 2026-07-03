from __future__ import annotations

import plotly.express as px
import streamlit as st

from data import data_freshness, load_latest_clusters, load_model_runs, load_station_usage_summary

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")
st.title("Model Performance")

freshness = data_freshness()
st.metric(
    "Live data freshness (most recent snapshot)",
    freshness.strftime("%Y-%m-%d %H:%M UTC") if freshness is not None else "no data yet",
)

st.subheader("Demand forecast: model vs. naive baseline")
st.caption('Baseline = "same value, same hour, last week."')
runs = load_model_runs("demand_forecast")
if runs.empty:
    st.info("No training runs yet - run train_and_score_models.")
else:
    for metric_name in sorted(runs["metric_name"].unique()):
        metric_runs = runs[runs["metric_name"] == metric_name]
        fig = px.line(
            metric_runs,
            x="run_ts",
            y=["metric_value", "baseline_metric_value"],
            labels={"value": metric_name.upper(), "run_ts": "run time"},
            title=f"{metric_name.upper()} over time",
        )
        fig.data[0].name = "model"
        fig.data[1].name = "naive baseline"
        st.plotly_chart(fig, use_container_width=True)

st.subheader("Station usage clusters")
clusters = load_latest_clusters()
usage = load_station_usage_summary()
if clusters.empty or usage.empty:
    st.info("No clustering results yet - run train_and_score_models.")
else:
    scatter_df = usage.merge(clusters[["station_id", "cluster_label"]], on="station_id", how="inner")
    scatter_df["weekday_weekend_ratio"] = scatter_df["weekday_avg_departures"] / (
        scatter_df["weekend_avg_departures"].fillna(0) + 1e-3
    )
    fig = px.scatter(
        scatter_df,
        x="avg_trips_per_day",
        y="weekday_weekend_ratio",
        color="cluster_label",
        labels={"avg_trips_per_day": "avg trips / day", "weekday_weekend_ratio": "weekday:weekend ratio"},
        title="Stations by usage archetype",
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Clustering run quality")
cluster_runs = load_model_runs("clustering")
if not cluster_runs.empty:
    st.dataframe(
        cluster_runs[cluster_runs["metric_name"] == "silhouette_score"],
        use_container_width=True,
        hide_index=True,
    )
