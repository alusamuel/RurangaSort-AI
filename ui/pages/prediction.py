"""Single-image prediction page."""
from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ui.api_client import predict_image  # noqa: E402

st.set_page_config(page_title="Prediction | RurangaSort AI", page_icon="🔍", layout="wide")
st.title("🔍 Predict a Waste Category")

st.write("Upload a single image of an item and RurangaSort AI will classify it.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    left, right = st.columns(2)
    left.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)

    if right.button("Predict", type="primary"):
        with st.spinner("Running inference..."):
            try:
                result = predict_image(
                    uploaded_file.getvalue(), uploaded_file.name, uploaded_file.type or "image/jpeg"
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Prediction failed: {exc}")
                st.stop()

        right.success(f"Predicted class: **{result['prediction']}**")
        right.metric("Confidence", f"{result['confidence'] * 100:.1f}%")
        right.metric("Latency", f"{result['latency_ms']:.1f} ms")
        right.caption(f"Model version: {result['model_version']}")

        st.subheader("Top predictions")
        top = result["top_predictions"]
        labels = [t["class"] for t in top]
        values = [t["confidence"] for t in top]
        fig = px.bar(x=values, y=labels, orientation="h", labels={"x": "Confidence", "y": "Class"})
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
