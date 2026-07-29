from __future__ import annotations

import pytest

from src.config import settings
import src.model_registry as model_registry


@pytest.fixture(autouse=True)
def isolated_model_dir(tmp_path, monkeypatch):
    # A dedicated subdirectory, not tmp_path itself -- tiny_model_dir (used by some of
    # these tests) also writes an "active/" folder directly under tmp_path, and reusing
    # the same root would make the registry see that pre-seeded metadata as if it were
    # an already-active v1 model, throwing off every version number by one.
    registry_root = tmp_path / "registry"
    monkeypatch.setattr(settings, "model_dir", str(registry_root))
    return registry_root


def _metrics(macro_f1: float, avg_latency_ms: float = 50.0, per_class=None) -> dict:
    return {
        "macro_f1": macro_f1,
        "avg_latency_ms": avg_latency_ms,
        "per_class": per_class or {"cardboard": {"recall": 0.9}, "glass": {"recall": 0.85}},
    }


def test_first_model_is_always_promoted(tiny_model_dir):
    metrics = _metrics(macro_f1=0.5)
    model_registry.register_candidate(
        model_file=tiny_model_dir["model_path"],
        class_names=tiny_model_dir["class_names"],
        metrics=metrics,
        model_name="baseline_cnn",
        input_shape=(16, 16, 3),
    )

    decision = model_registry.evaluate_promotion(metrics, active_metrics=None)
    assert decision["should_promote"] is True

    metadata = model_registry.promote_candidate()
    assert metadata["model_version"] == "v1"
    assert metadata["status"] == "active"
    assert model_registry.get_active_metadata()["model_version"] == "v1"


def test_candidate_worse_than_active_is_rejected(tiny_model_dir):
    active_metrics = _metrics(macro_f1=0.9)
    model_registry.register_candidate(
        model_file=tiny_model_dir["model_path"],
        class_names=tiny_model_dir["class_names"],
        metrics=active_metrics,
        model_name="baseline_cnn",
        input_shape=(16, 16, 3),
    )
    model_registry.promote_candidate()

    worse_metrics = _metrics(macro_f1=0.5)
    model_registry.register_candidate(
        model_file=tiny_model_dir["model_path"],
        class_names=tiny_model_dir["class_names"],
        metrics=worse_metrics,
        model_name="baseline_cnn",
        input_shape=(16, 16, 3),
    )
    decision = model_registry.evaluate_promotion(worse_metrics, model_registry.get_active_metrics())
    assert decision["should_promote"] is False

    archive_path = model_registry.reject_candidate(decision["reasons"])
    assert archive_path.exists()
    assert model_registry.get_active_metadata()["model_version"] == "v1"


def test_promotion_archives_previous_active_version(tiny_model_dir):
    for macro_f1 in (0.5, 0.6, 0.7):
        metrics = _metrics(macro_f1=macro_f1)
        model_registry.register_candidate(
            model_file=tiny_model_dir["model_path"],
            class_names=tiny_model_dir["class_names"],
            metrics=metrics,
            model_name="baseline_cnn",
            input_shape=(16, 16, 3),
        )
        model_registry.promote_candidate()

    assert model_registry.get_active_metadata()["model_version"] == "v3"
    assert set(model_registry.list_archived_versions()) == {"v1", "v2"}


def test_rollback_restores_archived_version(tiny_model_dir):
    for macro_f1 in (0.5, 0.6):
        metrics = _metrics(macro_f1=macro_f1)
        model_registry.register_candidate(
            model_file=tiny_model_dir["model_path"],
            class_names=tiny_model_dir["class_names"],
            metrics=metrics,
            model_name="baseline_cnn",
            input_shape=(16, 16, 3),
        )
        model_registry.promote_candidate()

    assert model_registry.get_active_metadata()["model_version"] == "v2"
    model_registry.rollback_to_version("v1")
    assert model_registry.get_active_metadata()["model_version"] == "v1"


def test_high_latency_candidate_is_rejected_even_if_accurate(tiny_model_dir):
    metrics = _metrics(macro_f1=0.99, avg_latency_ms=settings.promotion_max_latency_ms + 1000)
    decision = model_registry.evaluate_promotion(metrics, active_metrics=None)
    assert decision["should_promote"] is False
