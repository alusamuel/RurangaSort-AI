from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from api.dependencies import get_current_predictor
from api.schemas import PredictionResponse
from src.config import ALLOWED_IMAGE_MIME_TYPES, settings
from src.monitoring import record_prediction
from src.prediction import ModelNotLoadedError, Predictor

logger = logging.getLogger(__name__)
router = APIRouter(tags=["prediction"])

MAX_UPLOAD_BYTES = settings.max_upload_size_mb * 1024 * 1024


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), predictor: Predictor = Depends(get_current_predictor)):
    if file.content_type not in ALLOWED_IMAGE_MIME_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {file.content_type}")

    image_bytes = await file.read()
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded image exceeds the maximum allowed size.")

    try:
        result = predictor.predict(image_bytes)
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        record_prediction(success=False)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed")
        record_prediction(success=False)
        raise HTTPException(status_code=500, detail="Prediction failed unexpectedly.") from exc

    record_prediction(success=True, latency_ms=result["latency_ms"], predicted_class=result["prediction"])
    return result
