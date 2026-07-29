"""Single-image and batch prediction using the currently active model."""
from __future__ import annotations

import io
import json
import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.config import settings
from src.preprocessing import load_class_names

logger = logging.getLogger(__name__)


class ModelNotLoadedError(RuntimeError):
    pass


class Predictor:
    """Lazily loads the active model and serves predictions, tracking basic latency stats."""

    def __init__(self, model_path: Path | None = None, class_names_path: Path | None = None):
        self.model_path = model_path or settings.resolve(settings.model_path)
        # class_names.json and model_metadata.json are always written alongside model.h5
        # (see src.model_registry.register_candidate/promote_candidate), so they're derived
        # from the model's own directory rather than a separate global settings path --
        # otherwise a real promotion to v2/v3 would update models/active/model_metadata.json
        # while the predictor kept reading a stale, unrelated top-level path and silently
        # never noticed the version (or class list) had changed.
        self.class_names_path = class_names_path or self.model_path.parent / "class_names.json"
        self._metadata_path = self.model_path.parent / "model_metadata.json"
        self._model = None
        self._class_names: list[str] | None = None
        self._model_version = "unloaded"
        self._metadata_mtime: float | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def class_names(self) -> list[str]:
        if self._class_names is None:
            self._class_names = load_class_names(self.class_names_path)
        return self._class_names

    @property
    def model_version(self) -> str:
        return self._model_version

    def load(self) -> None:
        import tensorflow as tf

        if not self.model_path.exists():
            raise ModelNotLoadedError(f"No model file found at {self.model_path}. Train a model first.")

        self._model = tf.keras.models.load_model(self.model_path)
        self._class_names = load_class_names(self.class_names_path)

        if self._metadata_path.exists():
            with open(self._metadata_path, "r", encoding="utf-8") as fh:
                metadata = json.load(fh)
            self._model_version = metadata.get("model_version", "v1")
            self._metadata_mtime = self._metadata_path.stat().st_mtime
        else:
            self._model_version = "v1"

        logger.info("Loaded model %s (version %s)", self.model_path, self._model_version)

    def reload(self) -> None:
        self._model = None
        self._class_names = None
        self.load()

    def refresh_if_promoted(self) -> None:
        """Reload from disk if `models/active/` was updated by another process (e.g. a Celery
        worker promoting a new model onto the shared volume). Cheap: only stats the metadata file.
        """
        if not self._metadata_path.exists():
            return
        current_mtime = self._metadata_path.stat().st_mtime
        if self._metadata_mtime is None or current_mtime > self._metadata_mtime:
            logger.info("Detected updated active model metadata, reloading predictor.")
            self.reload()

    def _preprocess_bytes(self, image_bytes: bytes) -> np.ndarray:
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.verify()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("Uploaded file is not a valid image.") from exc

        image = image.resize((settings.image_size, settings.image_size))
        array = np.asarray(image, dtype=np.float32) / 255.0
        return np.expand_dims(array, axis=0)

    def predict(self, image_bytes: bytes, top_k: int = 3) -> dict:
        if not self.is_loaded:
            self.load()

        batch = self._preprocess_bytes(image_bytes)

        start = time.perf_counter()
        probabilities = self._model.predict(batch, verbose=0)[0]
        latency_ms = (time.perf_counter() - start) * 1000.0

        ranked_indices = np.argsort(probabilities)[::-1]
        top_predictions = [
            {"class": self.class_names[i], "confidence": float(probabilities[i])}
            for i in ranked_indices[:top_k]
        ]

        return {
            "prediction": self.class_names[ranked_indices[0]],
            "confidence": float(probabilities[ranked_indices[0]]),
            "top_predictions": top_predictions,
            "model_version": self._model_version,
            "latency_ms": round(latency_ms, 3),
        }

    def predict_batch_from_directory(self, directory: Path) -> tuple[np.ndarray, np.ndarray]:
        """Used by production evaluation: returns (y_true_onehot, y_pred_proba) over `directory/<class>/*`."""
        if not self.is_loaded:
            self.load()

        num_classes = len(self.class_names)
        class_to_index = {name: i for i, name in enumerate(self.class_names)}

        y_true, y_pred = [], []
        for class_name in self.class_names:
            class_dir = directory / class_name
            if not class_dir.exists():
                continue
            for path in sorted(class_dir.iterdir()):
                if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue
                with open(path, "rb") as fh:
                    image_bytes = fh.read()
                batch = self._preprocess_bytes(image_bytes)
                probs = self._model.predict(batch, verbose=0)[0]

                one_hot = np.zeros(num_classes)
                one_hot[class_to_index[class_name]] = 1.0
                y_true.append(one_hot)
                y_pred.append(probs)

        return np.array(y_true), np.array(y_pred)


_predictor: Predictor | None = None


def get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor
