"""Pydantic request/response models for the FastAPI service."""
from __future__ import annotations

from pydantic import BaseModel, Field


class TopPrediction(BaseModel):
    class_: str = Field(alias="class")
    confidence: float

    model_config = {"populate_by_name": True}


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    top_predictions: list[dict]
    model_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None
    uptime_seconds: float
    environment: str


class ModelInfoResponse(BaseModel):
    model_version: str | None
    model_name: str | None
    framework: str | None
    input_shape: list[int] | None
    classes: list[str]
    created_at: str | None
    macro_f1: float | None
    status: str | None


class UploadValidationResponse(BaseModel):
    total_files: int
    accepted: int
    corrupted: list[str]
    duplicates: list[str]
    unsupported_format: list[str]
    unknown_class: list[str]
    unsafe_path: list[str]
    oversized: list[str]
    message: str


class RetrainRequest(BaseModel):
    model_name: str = Field(default="baseline_cnn", description="'baseline_cnn' or 'mobilenet_v2'")
    epochs: int = Field(default=10, ge=1, le=200)
    fine_tune_epochs: int = Field(default=5, ge=0, le=100)


class RetrainResponse(BaseModel):
    job_id: str
    status: str


class TrainingStatusResponse(BaseModel):
    job_id: str
    state: str
    stage: str | None = None
    result: dict | None = None


class EvaluationResponse(BaseModel):
    model_version: str | None
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    confusion_matrix: list[list[int]]
    class_names: list[str]
    roc_auc_ovr_macro: float | None
    pr_auc_macro: float | None
    log_loss: float | None
    avg_latency_ms: float
    p95_latency_ms: float


class MetricsResponse(BaseModel):
    predictions_total: int
    predictions_failed: int
    predictions_succeeded: int
    avg_latency_ms: float
    p95_latency_ms: float
    class_prediction_distribution: dict[str, int]
    last_retraining: dict | None
    process_uptime_seconds: float
    metrics_backend: str


class DatasetSummaryResponse(BaseModel):
    class_statistics: dict
    manifest: dict | None
