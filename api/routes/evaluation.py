from __future__ import annotations

import csv
import json
import logging

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_current_predictor
from api.schemas import EvaluationResponse
from src.config import settings
from src.evaluation import compute_classification_metrics
from src.model_registry import get_active_metadata
from src.monitoring import get_metrics_snapshot
from src.prediction import ModelNotLoadedError, Predictor

logger = logging.getLogger(__name__)
router = APIRouter(tags=["evaluation"])


@router.post("/evaluate", response_model=EvaluationResponse)
def evaluate_active_model(predictor: Predictor = Depends(get_current_predictor)):
    """Runs the active model against the fixed production/test dataset and stores the result,
    so successive model versions can be compared over time."""
    try:
        if not predictor.is_loaded:
            predictor.load()
        y_true, y_pred = predictor.predict_batch_from_directory(settings.resolve(settings.data_test_dir))
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if len(y_true) == 0:
        raise HTTPException(
            status_code=422,
            detail=f"No test images found under {settings.data_test_dir}. Run scripts/prepare_data.py first.",
        )

    metrics = compute_classification_metrics(y_true, y_pred, predictor.class_names)
    live_metrics = get_metrics_snapshot()
    metrics["avg_latency_ms"] = live_metrics["avg_latency_ms"]
    metrics["p95_latency_ms"] = live_metrics["p95_latency_ms"]

    metadata = get_active_metadata() or {}
    version = metadata.get("model_version", "unknown")

    _persist_production_evaluation(version, metrics)

    return EvaluationResponse(model_version=version, **metrics)


def _persist_production_evaluation(version: str, metrics: dict) -> None:
    report_dir = settings.resolve("reports/production_evaluation")
    report_dir.mkdir(parents=True, exist_ok=True)

    with open(report_dir / f"{version}_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    comparison_path = report_dir / "model_comparison.csv"
    is_new_file = not comparison_path.exists()
    with open(comparison_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if is_new_file:
            writer.writerow(["model_version", "accuracy", "macro_f1", "weighted_f1", "roc_auc_ovr_macro", "avg_latency_ms"])
        writer.writerow(
            [
                version,
                metrics["accuracy"],
                metrics["macro_f1"],
                metrics["weighted_f1"],
                metrics.get("roc_auc_ovr_macro"),
                metrics["avg_latency_ms"],
            ]
        )
