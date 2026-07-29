"""RurangaSort AI - Streamlit landing page. Use the sidebar to navigate to
Dashboard, Prediction, Visualizations, and Retraining."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Streamlit only puts this script's own directory (ui/) on sys.path, not its
# parent -- so `from ui.api_client import ...` can't resolve unless the repo
# root is added explicitly first. Every page under ui/pages/ already does this;
# app.py (the entrypoint) needs it too, since it's not exempt from the same rule.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ui.api_client import BASE_URL, get_health  # noqa: E402

st.set_page_config(page_title="RurangaSort AI", page_icon="♻️", layout="wide")

st.title("♻️ RurangaSort AI")
st.caption("End-to-end waste-image classification: predict, visualize, upload, retrain, monitor.")

st.markdown(
    """
Use the pages in the left sidebar:

- **Dashboard** — API health, model uptime, active version, prediction & latency metrics.
- **Prediction** — upload one image and get the predicted waste category.
- **Visualizations** — dataset class distribution, brightness, colour, and dimension statistics.
- **Retraining** — bulk-upload labelled images and trigger a background retraining job.
"""
)

st.divider()

st.subheader("Backend connection")
st.write(f"API base URL: `{BASE_URL}`")

try:
    health = get_health()
    status = health.get("status")
    if status == "ok":
        st.success(f"Connected — model version `{health.get('model_version')}` loaded.")
    else:
        st.warning("Connected to the API, but no model is loaded yet. Train and promote a model first.")
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not reach the API at {BASE_URL}: {exc}")
    st.info("Start it with: `uvicorn api.main:app --host 0.0.0.0 --port 8000`")
