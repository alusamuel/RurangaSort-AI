"""Bulk data upload + retraining trigger page."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ui.api_client import get_training_status, trigger_retrain, upload_bulk_zip  # noqa: E402

st.set_page_config(page_title="Retraining | RurangaSort AI", page_icon="🔁", layout="wide")
st.title("🔁 Bulk Upload & Retraining")

api_key = st.text_input("API key (required for upload & retrain)", type="password")

st.subheader("1. Upload labelled images (bulk)")
st.markdown(
    "Upload a `.zip` with one top-level folder per class "
    "(`cardboard/`, `glass/`, `metal/`, `paper/`, `plastic/`, `trash/`), each containing images of that class."
)
zip_file = st.file_uploader("Choose a ZIP file", type=["zip"])

if zip_file is not None and st.button("Upload"):
    if not api_key:
        st.error("Enter the API key first.")
    else:
        with st.spinner("Validating and extracting ZIP..."):
            try:
                report = upload_bulk_zip(zip_file.getvalue(), zip_file.name, api_key)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Upload failed: {exc}")
                report = None
        if report:
            st.success(report["message"])
            with st.expander("Validation details"):
                st.json(report)

st.divider()
st.subheader("2. Trigger retraining")

col1, col2, col3 = st.columns(3)
model_name = col1.selectbox("Model architecture", ["baseline_cnn", "mobilenet_v2"])
epochs = col2.number_input("Epochs", min_value=1, max_value=200, value=10)
fine_tune_epochs = col3.number_input("Fine-tune epochs (MobileNetV2 only)", min_value=0, max_value=100, value=5)

if st.button("Start Retraining", type="primary"):
    if not api_key:
        st.error("Enter the API key first.")
    else:
        try:
            response = trigger_retrain(model_name, int(epochs), int(fine_tune_epochs), api_key)
            st.session_state["retrain_job_id"] = response["job_id"]
            st.success(f"Retraining queued — job id `{response['job_id']}`.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not start retraining: {exc}")

st.divider()
st.subheader("3. Training status")

job_id = st.session_state.get("retrain_job_id")
job_id_input = st.text_input("Job ID", value=job_id or "")

if st.button("Check Status"):
    if not job_id_input:
        st.warning("No job id yet — start a retraining run first, or paste one above.")
    else:
        try:
            status = get_training_status(job_id_input)
            state = status["state"]
            if state == "PROGRESS":
                st.info(f"In progress — stage: **{status.get('stage')}**")
            elif state == "SUCCESS":
                result = status.get("result") or {}
                if result.get("status") == "promoted":
                    st.success(f"Retraining finished — candidate PROMOTED to {result.get('model_version')}.")
                else:
                    st.warning("Retraining finished — candidate was REJECTED (active model unchanged).")
                st.json(result)
            elif state == "FAILURE":
                st.error("Retraining job failed.")
                st.json(status.get("result"))
            else:
                st.info(f"State: {state}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not fetch status: {exc}")
