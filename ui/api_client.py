"""Thin HTTP client the Streamlit UI uses to talk to the FastAPI backend."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import requests
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import settings  # noqa: E402


def _resolve_base_url() -> str:
    """Order of precedence: Streamlit Cloud secrets -> plain env var -> local default.

    On Streamlit Community Cloud the backend URL is set via the app's Secrets manager
    (st.secrets), not a regular environment variable, so that has to be checked
    explicitly -- os.environ alone isn't populated from it.
    """
    try:
        if "API_BASE_URL" in st.secrets:
            return str(st.secrets["API_BASE_URL"])
    except Exception:
        pass  # no secrets.toml present (e.g. running locally) -- fall through
    return os.environ.get("API_BASE_URL", settings.api_base_url)


BASE_URL = _resolve_base_url().rstrip("/")
DEFAULT_TIMEOUT = 30


def get_health() -> dict:
    response = requests.get(f"{BASE_URL}/health", timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    return response.json()


def get_model_info() -> dict:
    response = requests.get(f"{BASE_URL}/model-info", timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    return response.json()


def get_metrics() -> dict:
    response = requests.get(f"{BASE_URL}/metrics", timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    return response.json()


def get_dataset_summary() -> dict:
    response = requests.get(f"{BASE_URL}/dataset-summary", timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    return response.json()


def predict_image(file_bytes: bytes, filename: str, content_type: str) -> dict:
    files = {"file": (filename, file_bytes, content_type)}
    response = requests.post(f"{BASE_URL}/predict", files=files, timeout=DEFAULT_TIMEOUT)
    if not response.ok:
        raise RuntimeError(response.json().get("detail", response.text))
    return response.json()


def upload_bulk_zip(file_bytes: bytes, filename: str, api_key: str) -> dict:
    files = {"file": (filename, file_bytes, "application/zip")}
    headers = {"X-API-Key": api_key}
    response = requests.post(f"{BASE_URL}/upload-data", files=files, headers=headers, timeout=120)
    if not response.ok:
        raise RuntimeError(response.json().get("detail", response.text))
    return response.json()


def trigger_retrain(model_name: str, epochs: int, fine_tune_epochs: int, api_key: str) -> dict:
    headers = {"X-API-Key": api_key}
    payload = {"model_name": model_name, "epochs": epochs, "fine_tune_epochs": fine_tune_epochs}
    response = requests.post(f"{BASE_URL}/retrain", json=payload, headers=headers, timeout=DEFAULT_TIMEOUT)
    if not response.ok:
        raise RuntimeError(response.json().get("detail", response.text))
    return response.json()


def get_training_status(job_id: str) -> dict:
    response = requests.get(f"{BASE_URL}/training-status/{job_id}", timeout=DEFAULT_TIMEOUT)
    if not response.ok:
        raise RuntimeError(response.json().get("detail", response.text))
    return response.json()


def evaluate_active_model() -> dict:
    response = requests.post(f"{BASE_URL}/evaluate", timeout=120)
    if not response.ok:
        raise RuntimeError(response.json().get("detail", response.text))
    return response.json()
