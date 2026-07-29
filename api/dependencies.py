"""Shared FastAPI dependencies: API-key auth and the process-wide predictor instance."""
from __future__ import annotations

from fastapi import Header, HTTPException, status

from src.config import settings
from src.prediction import Predictor, get_predictor


def get_current_predictor() -> Predictor:
    predictor = get_predictor()
    if predictor.is_loaded:
        predictor.refresh_if_promoted()
    return predictor


def require_api_key(x_api_key: str = Header(default="")) -> None:
    """Guards write endpoints (upload / retrain). Not applied to /predict, which stays public."""
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing X-API-Key header.")
