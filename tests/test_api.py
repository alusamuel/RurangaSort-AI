from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app
from src.config import settings


def test_root_endpoint():
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == settings.app_name


def test_health_endpoint_reports_degraded_without_a_model(tmp_path):
    from src.prediction import get_predictor

    predictor = get_predictor()
    predictor.model_path = tmp_path / "no_model_here.h5"
    predictor._model = None

    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False
    assert body["uptime_seconds"] >= 0


def test_model_info_returns_empty_metadata_when_no_active_model(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "model_dir", str(tmp_path / "no_models"))
    with TestClient(app) as client:
        response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] is None
    assert body["classes"] == []


def test_metrics_endpoint_shape():
    with TestClient(app) as client:
        response = client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert "predictions_total" in body
    assert "avg_latency_ms" in body
    assert "metrics_backend" in body


def test_dataset_summary_handles_empty_raw_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_raw_dir", str(tmp_path / "empty_raw"))
    with TestClient(app) as client:
        response = client.get("/dataset-summary")
    assert response.status_code == 200
    body = response.json()
    assert set(body["class_statistics"].keys()) == {
        "cardboard", "glass", "metal", "paper", "plastic", "trash",
    }
    assert all(v["count"] == 0 for v in body["class_statistics"].values())


def test_retrain_requires_api_key():
    with TestClient(app) as client:
        response = client.post("/retrain", json={"model_name": "baseline_cnn", "epochs": 1})
    assert response.status_code == 401


def test_retrain_rejects_wrong_api_key():
    with TestClient(app) as client:
        response = client.post(
            "/retrain",
            json={"model_name": "baseline_cnn", "epochs": 1},
            headers={"X-API-Key": "wrong-key"},
        )
    assert response.status_code == 401


def test_training_status_unknown_job_returns_404_or_503_without_redis():
    # 404 when a real Celery/Redis backend is reachable and confirms the job id is unknown;
    # 503 when Redis itself isn't running (e.g. local `pytest` without `docker compose up`).
    with TestClient(app) as client:
        response = client.get("/training-status/does-not-exist")
    assert response.status_code in (404, 503)


def test_predict_returns_503_without_a_model(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "model_path", str(tmp_path / "no_model_here.h5"))
    from src.prediction import get_predictor

    get_predictor().model_path = tmp_path / "no_model_here.h5"
    get_predictor()._model = None

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={"file": ("x.jpg", b"fake", "image/jpeg")},
        )
    assert response.status_code == 503
