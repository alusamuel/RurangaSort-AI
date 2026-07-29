from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_current_predictor
from api.schemas import HealthResponse, ModelInfoResponse
from src.config import settings
from src.model_registry import get_active_metadata
from src.monitoring import get_process_uptime_seconds
from src.prediction import Predictor

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(predictor: Predictor = Depends(get_current_predictor)):
    model_loaded = predictor.is_loaded
    if not model_loaded:
        try:
            predictor.load()
            model_loaded = True
        except Exception:
            model_loaded = False

    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_version=predictor.model_version if model_loaded else None,
        uptime_seconds=round(get_process_uptime_seconds(), 1),
        environment=settings.environment,
    )


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    metadata = get_active_metadata() or {}
    return ModelInfoResponse(
        model_version=metadata.get("model_version"),
        model_name=metadata.get("model_name"),
        framework=metadata.get("framework"),
        input_shape=metadata.get("input_shape"),
        classes=metadata.get("classes", []),
        created_at=metadata.get("created_at"),
        macro_f1=metadata.get("macro_f1"),
        status=metadata.get("status"),
    )
