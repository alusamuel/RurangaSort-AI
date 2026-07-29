"""Dataset visualizations: class distribution, brightness, RGB, and dimension statistics."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ui.api_client import get_dataset_summary  # noqa: E402

st.set_page_config(page_title="Visualizations | RurangaSort AI", page_icon="📈", layout="wide")
st.title("📈 Dataset Visualizations")

try:
    summary = get_dataset_summary()
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not reach the API: {exc}")
    st.stop()

stats = summary["class_statistics"]
rows = []
for class_name, s in stats.items():
    rgb = s.get("rgb_mean") or {}
    rows.append(
        {
            "class": class_name,
            "count": s.get("count") or 0,
            "brightness_mean": s.get("brightness_mean"),
            "r": rgb.get("r"),
            "g": rgb.get("g"),
            "b": rgb.get("b"),
            "avg_width": s.get("avg_width"),
            "avg_height": s.get("avg_height"),
            "avg_aspect_ratio": s.get("avg_aspect_ratio"),
        }
    )
df = pd.DataFrame(rows)

st.subheader("1. Class distribution")
st.plotly_chart(px.bar(df, x="class", y="count", labels={"count": "Number of images"}), use_container_width=True)
st.markdown(
    "**Interpretation:** an uneven bar chart means the model sees far more examples of some categories "
    "than others. Without correction (class weights, augmentation, or more data collection) the model "
    "tends to over-predict the majority class and under-recall minority classes — watch per-class recall "
    "in the evaluation report, not just overall accuracy."
)

st.divider()
st.subheader("2. Average image brightness per class")
if df["brightness_mean"].notna().any():
    st.plotly_chart(
        px.bar(df, x="class", y="brightness_mean", labels={"brightness_mean": "Mean grayscale intensity (0-255)"}),
        use_container_width=True,
    )
    st.markdown(
        "**Interpretation:** systematic brightness differences between classes usually mean the photos "
        "were taken under different lighting/backgrounds rather than reflecting a real property of the "
        "material. A CNN can latch onto that shortcut instead of learning object shape — if brightness "
        "varies a lot by class, augmenting with brightness jitter (already applied to the training set) "
        "and collecting more varied backgrounds both help."
    )

st.divider()
st.subheader("3. Average RGB channel means per class")
if df[["r", "g", "b"]].notna().any().any():
    rgb_long = df.melt(id_vars="class", value_vars=["r", "g", "b"], var_name="channel", value_name="mean_value")
    st.plotly_chart(
        px.bar(rgb_long, x="class", y="mean_value", color="channel", barmode="group"), use_container_width=True
    )
    st.markdown(
        "**Interpretation:** classes with a clearly dominant channel (e.g. cardboard skewing red/brown, "
        "glass/metal skewing cooler/grey) tell us the model may partly be relying on colour rather than "
        "shape or texture — useful to know, since colour cues won't generalise to unusually coloured or "
        "tinted items of the same material."
    )

st.divider()
st.subheader("4. Image dimensions & aspect ratio")
if df["avg_aspect_ratio"].notna().any():
    st.plotly_chart(
        px.scatter(df, x="avg_width", y="avg_height", size="avg_aspect_ratio", color="class", hover_data=["class"]),
        use_container_width=True,
    )
    st.markdown(
        "**Interpretation:** wide spread in width/height/aspect ratio across classes suggests images were "
        "sourced from different cameras or cropping conventions — resizing to a fixed 224x224 square can "
        "distort thin/elongated items (e.g. bottles), which is a candidate explanation if that class "
        "underperforms in the confusion matrix."
    )

st.divider()
manifest = summary.get("manifest")
if manifest:
    st.subheader("Train / validation / test split")
    st.json(manifest.get("counts", {}))
else:
    st.info("No dataset manifest found yet — run `python scripts/prepare_data.py`.")
