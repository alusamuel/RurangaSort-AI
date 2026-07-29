"""RurangaSort AI FastAPI application entrypoint."""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from api.routes import evaluation, health, metrics, prediction, retraining, upload
from src.config import settings
from src.prediction import get_predictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor = get_predictor()
    try:
        predictor.load()
        logger.info("Startup: active model loaded (version %s).", predictor.model_version)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Startup: no active model available yet (%s). /predict will 503 until one is trained.", exc)
    yield


app = FastAPI(
    title="RurangaSort AI",
    description="Waste-image classification API: prediction, bulk upload, retraining, monitoring.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info("%s %s -> %s (%.1fms)", request.method, request.url.path, response.status_code, duration_ms)
    return response


app.include_router(health.router)
app.include_router(prediction.router)
app.include_router(upload.router)
app.include_router(retraining.router)
app.include_router(evaluation.router)
app.include_router(metrics.router)


@app.get("/", tags=["root"])
def root():
    return {
        "service": settings.app_name,
        "docs": "/docs",
        "health": "/health",
    }
