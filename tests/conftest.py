from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pytest
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def make_image_bytes(color=(120, 80, 40), size=(64, 64), fmt="JPEG") -> bytes:
    image = Image.new("RGB", size, color)
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.fixture
def sample_image_bytes() -> bytes:
    return make_image_bytes()


@pytest.fixture
def make_dataset_zip(tmp_path):
    """Builds a small in-memory-style ZIP (written to tmp_path) with a configurable
    member list, for exercising validate_dataset_zip / safe_extract_zip edge cases."""

    def _build(members: dict[str, bytes], zip_name: str = "upload.zip") -> Path:
        zip_path = tmp_path / zip_name
        with zipfile.ZipFile(zip_path, "w") as zf:
            for name, data in members.items():
                zf.writestr(name, data)
        return zip_path

    return _build


@pytest.fixture
def raw_dataset(tmp_path):
    """Creates a tiny data/raw/<class>/*.jpg tree with a few images per class."""
    raw_dir = tmp_path / "raw"
    classes = ["cardboard", "glass"]
    for class_name in classes:
        class_dir = raw_dir / class_name
        class_dir.mkdir(parents=True)
        for i in range(4):
            color = (100 + i * 10, 50, 30) if class_name == "cardboard" else (30, 100, 150 + i * 5)
            (class_dir / f"{class_name}_{i}.jpg").write_bytes(make_image_bytes(color=color))
    return raw_dir, classes


@pytest.fixture
def tiny_model_dir(tmp_path):
    """Builds and saves a tiny Keras model (small input size for test speed) plus
    class_names.json + model_metadata.json under tmp_path, mimicking models/active/."""
    import json

    import tensorflow as tf

    class_names = ["cardboard", "glass"]
    image_size = 16

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(image_size, image_size, 3)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(len(class_names), activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    model_dir = tmp_path / "active"
    model_dir.mkdir()
    model_path = model_dir / "model.h5"
    model.save(model_path)

    with open(model_dir / "class_names.json", "w", encoding="utf-8") as fh:
        json.dump(class_names, fh)

    metadata = {"model_version": "v1", "model_name": "tiny_test_model", "status": "active"}
    with open(model_dir / "model_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh)

    return {
        "model_dir": model_dir,
        "model_path": model_path,
        "class_names_path": model_dir / "class_names.json",
        "metadata_path": model_dir / "model_metadata.json",
        "class_names": class_names,
        "image_size": image_size,
    }
