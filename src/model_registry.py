"""File-based model registry: versioning, promotion rules, and rollback.

models/
  active/      currently served model + class_names.json + metrics.json
  candidate/   newly trained model awaiting the promotion decision
  archived/    previous versions, one subfolder per version, kept for rollback
"""
from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from src.config import settings
from src.evaluation import compare_candidate_to_active

logger = logging.getLogger(__name__)


def _active_dir() -> Path:
    return settings.resolve(settings.model_dir) / "active"


def _candidate_dir() -> Path:
    return settings.resolve(settings.model_dir) / "candidate"


def _archived_dir() -> Path:
    return settings.resolve(settings.model_dir) / "archived"


def get_active_metadata() -> dict | None:
    metadata_path = _active_dir() / "model_metadata.json"
    if not metadata_path.exists():
        return None
    with open(metadata_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def get_active_metrics() -> dict | None:
    metrics_path = _active_dir() / "metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _next_version(existing_metadata: dict | None) -> str:
    if not existing_metadata or "model_version" not in existing_metadata:
        return "v1"
    try:
        current = int(str(existing_metadata["model_version"]).lstrip("v"))
    except ValueError:
        current = 1
    return f"v{current + 1}"


def register_candidate(
    model_file: Path,
    class_names: list[str],
    metrics: dict,
    model_name: str,
    input_shape: tuple[int, int, int],
) -> Path:
    """Stage a newly trained model as a candidate, ready for the promotion decision.

    `model_file` may itself already live inside `models/candidate/` (e.g. the default
    output of `python -m src.training`), so the source is copied out to a temp file
    *before* the candidate directory is wiped, rather than deleting out from under it.
    """
    import tempfile

    candidate_dir = _candidate_dir()

    with tempfile.TemporaryDirectory() as tmp_dir:
        staged_source = Path(tmp_dir) / model_file.name
        shutil.copy2(model_file, staged_source)

        if candidate_dir.exists():
            shutil.rmtree(candidate_dir)
        candidate_dir.mkdir(parents=True, exist_ok=True)

        dest_model_path = candidate_dir / "model.h5"
        shutil.copy2(staged_source, dest_model_path)

    with open(candidate_dir / "class_names.json", "w", encoding="utf-8") as fh:
        json.dump(class_names, fh, indent=2)
    with open(candidate_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    metadata = {
        "model_name": model_name,
        "framework": "TensorFlow",
        "input_shape": list(input_shape),
        "classes": class_names,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "macro_f1": metrics.get("macro_f1"),
        "status": "candidate",
        "model_file": dest_model_path.name,
    }
    with open(candidate_dir / "model_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    return dest_model_path


def evaluate_promotion(candidate_metrics: dict, active_metrics: dict | None) -> dict:
    """Apply the promotion rules and return a decision with the reasons behind it."""
    comparison = compare_candidate_to_active(candidate_metrics, active_metrics)

    reasons = []
    should_promote = True

    if not comparison["is_first_model"]:
        if comparison["macro_f1_delta"] < settings.promotion_min_macro_f1_delta:
            should_promote = False
            reasons.append(
                f"Macro F1 dropped by {abs(comparison['macro_f1_delta']):.4f}, "
                f"exceeding the allowed tolerance of {abs(settings.promotion_min_macro_f1_delta):.4f}."
            )
        if comparison["max_per_class_recall_drop"] > settings.promotion_max_per_class_recall_drop:
            should_promote = False
            reasons.append(
                f"A class's recall dropped by {comparison['max_per_class_recall_drop']:.4f}, "
                f"exceeding the allowed threshold of {settings.promotion_max_per_class_recall_drop:.4f}."
            )

    if candidate_metrics.get("avg_latency_ms", 0) > settings.promotion_max_latency_ms:
        should_promote = False
        reasons.append(
            f"Average latency {candidate_metrics.get('avg_latency_ms'):.1f}ms exceeds the limit of "
            f"{settings.promotion_max_latency_ms}ms."
        )

    if should_promote:
        reasons.append("All promotion checks passed.")

    return {"should_promote": should_promote, "reasons": reasons, "comparison": comparison}


def promote_candidate() -> dict:
    """Archive the current active model (if any) and promote the staged candidate to active."""
    active_dir = _active_dir()
    candidate_dir = _candidate_dir()
    archived_dir = _archived_dir()

    if not candidate_dir.exists() or not any(candidate_dir.iterdir()):
        raise FileNotFoundError("No candidate model is staged for promotion.")

    candidate_metadata_path = candidate_dir / "model_metadata.json"
    with open(candidate_metadata_path, "r", encoding="utf-8") as fh:
        candidate_metadata = json.load(fh)

    previous_metadata = get_active_metadata()
    new_version = _next_version(previous_metadata)

    if active_dir.exists() and any(active_dir.iterdir()):
        archive_target = archived_dir / (previous_metadata.get("model_version", "unknown") if previous_metadata else "unknown")
        if archive_target.exists():
            shutil.rmtree(archive_target)
        shutil.copytree(active_dir, archive_target)
        shutil.rmtree(active_dir)

    active_dir.mkdir(parents=True, exist_ok=True)
    for item in candidate_dir.iterdir():
        shutil.copy2(item, active_dir / item.name)

    candidate_metadata["model_version"] = new_version
    candidate_metadata["status"] = "active"
    candidate_metadata["promoted_at"] = datetime.now(timezone.utc).isoformat()
    with open(active_dir / "model_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(candidate_metadata, fh, indent=2)

    shutil.rmtree(candidate_dir)

    logger.info("Promoted candidate to active model %s", new_version)
    return candidate_metadata


def reject_candidate(reasons: list[str]) -> Path:
    """Archive a rejected candidate (for inspection) instead of promoting it."""
    candidate_dir = _candidate_dir()
    archived_dir = _archived_dir() / f"rejected-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    if candidate_dir.exists():
        shutil.copytree(candidate_dir, archived_dir)
        with open(archived_dir / "rejection_reasons.json", "w", encoding="utf-8") as fh:
            json.dump({"reasons": reasons}, fh, indent=2)
        shutil.rmtree(candidate_dir)
    return archived_dir


def list_archived_versions() -> list[str]:
    archived_dir = _archived_dir()
    if not archived_dir.exists():
        return []
    return sorted(p.name for p in archived_dir.iterdir() if p.is_dir())


def rollback_to_version(version: str) -> dict:
    """Restore an archived version as the active model (manual rollback)."""
    archive_source = _archived_dir() / version
    if not archive_source.exists():
        raise FileNotFoundError(f"No archived version '{version}' found.")

    active_dir = _active_dir()
    if active_dir.exists():
        shutil.rmtree(active_dir)
    shutil.copytree(archive_source, active_dir)

    metadata_path = active_dir / "model_metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as fh:
        metadata = json.load(fh)
    metadata["status"] = "active"
    metadata["rolled_back_at"] = datetime.now(timezone.utc).isoformat()
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    logger.info("Rolled back active model to version %s", version)
    return metadata
