from __future__ import annotations

import json

from fastapi import APIRouter

from api.schemas import DatasetSummaryResponse, MetricsResponse
from src.config import DEFAULT_CLASS_NAMES, settings
from src.monitoring import get_metrics_snapshot
from src.preprocessing import compute_dataset_statistics

router = APIRouter(tags=["metrics"])


@router.get("/metrics", response_model=MetricsResponse)
def metrics():
    return get_metrics_snapshot()


@router.get("/dataset-summary", response_model=DatasetSummaryResponse)
def dataset_summary():
    raw_dir = settings.resolve(settings.data_raw_dir)
    stats = compute_dataset_statistics(raw_dir, DEFAULT_CLASS_NAMES)

    manifest_path = settings.resolve(settings.data_processed_dir) / "manifest.json"
    manifest = None
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)

    return DatasetSummaryResponse(class_statistics=stats, manifest=manifest)
