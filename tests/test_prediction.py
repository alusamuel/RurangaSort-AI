from __future__ import annotations

import pytest

from src.config import settings
from src.prediction import ModelNotLoadedError, Predictor
from tests.conftest import make_image_bytes


@pytest.fixture
def predictor(tiny_model_dir, monkeypatch):
    monkeypatch.setattr(settings, "image_size", tiny_model_dir["image_size"])
    return Predictor(
        model_path=tiny_model_dir["model_path"],
        class_names_path=tiny_model_dir["class_names_path"],
    )


def test_predictor_raises_when_model_file_missing(tmp_path):
    predictor = Predictor(model_path=tmp_path / "missing.h5", class_names_path=tmp_path / "class_names.json")
    with pytest.raises(ModelNotLoadedError):
        predictor.load()


def test_predictor_loads_and_predicts(predictor):
    result = predictor.predict(make_image_bytes())

    assert result["prediction"] in {"cardboard", "glass"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["top_predictions"]) == 2
    assert result["model_version"] == "v1"
    assert result["latency_ms"] >= 0.0


def test_predictor_rejects_invalid_image_bytes(predictor):
    with pytest.raises(ValueError):
        predictor.predict(b"not a real image")


def test_predictor_lazy_loads_on_first_predict(predictor):
    assert predictor.is_loaded is False
    predictor.predict(make_image_bytes())
    assert predictor.is_loaded is True


def test_predictor_reload_resets_state(predictor):
    predictor.load()
    assert predictor.is_loaded is True
    predictor.reload()
    assert predictor.is_loaded is True
