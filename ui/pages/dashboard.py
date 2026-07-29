"""Model monitoring dashboard: API health, uptime, active version, prediction & latency metrics."""
from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ui.api_client import get_health, get_metrics, get_model_info  # noqa: E402

st.set_page_config(page_title="Dashboard | RurangaSort AI", page_icon="📊", layout="wide")
st.title("📊 Model & API Dashboard")

if st.button("Refresh"):
    st.rerun()

try:
    health = get_health()
    model_info = get_model_info()
    metrics = get_metrics()
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not reach the API: {exc}")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("API status", health["status"].upper())
col2.metric("Model loaded", "Yes" if health["model_loaded"] else "No")
col3.metric("Active version", model_info.get("model_version") or "—")
col4.metric("Process uptime", f"{health['uptime_seconds'] / 60:.1f} min")

col5, col6, col7, col8 = st.columns(4)
col5.metric("Total predictions", metrics["predictions_total"])
col6.metric("Failed predictions", metrics["predictions_failed"])
col7.metric("Avg latency", f"{metrics['avg_latency_ms']:.1f} ms")
col8.metric("P95 latency", f"{metrics['p95_latency_ms']:.1f} ms")

st.divider()
st.subheader("Model metadata")
st.json(model_info)

st.divider()
last_retraining = metrics.get("last_retraining")
st.subheader("Last retraining event")
if last_retraining:
    st.json(last_retraining)
else:
    st.info("No retraining has run yet.")

st.divider()
st.subheader("Prediction class distribution (live traffic)")
class_dist = metrics.get("class_prediction_distribution") or {}
if class_dist:
    fig = px.bar(x=list(class_dist.keys()), y=list(class_dist.values()), labels={"x": "Class", "y": "Predictions"})
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No predictions recorded yet — try the Prediction page.")

st.caption(f"Metrics backend: {metrics.get('metrics_backend')} (Redis-shared across API replicas when available).")
