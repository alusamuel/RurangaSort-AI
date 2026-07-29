from __future__ import annotations

import io
import zipfile

from fastapi.testclient import TestClient

from api.main import app
from src.config import settings
from tests.conftest import make_image_bytes


def _make_zip_bytes(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buffer.getvalue()


def test_upload_data_requires_api_key(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "upload_directory", str(tmp_path / "uploaded"))
    zip_bytes = _make_zip_bytes({"cardboard/a.jpg": make_image_bytes()})

    with TestClient(app) as client:
        response = client.post(
            "/upload-data", files={"file": ("data.zip", zip_bytes, "application/zip")}
        )
    assert response.status_code == 401


def test_upload_data_accepts_valid_zip_with_correct_key(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "upload_directory", str(tmp_path / "uploaded"))
    zip_bytes = _make_zip_bytes(
        {
            "cardboard/a.jpg": make_image_bytes(color=(1, 1, 1)),
            "cardboard/b.jpg": make_image_bytes(color=(2, 2, 2)),
        }
    )

    with TestClient(app) as client:
        response = client.post(
            "/upload-data",
            files={"file": ("data.zip", zip_bytes, "application/zip")},
            headers={"X-API-Key": settings.api_key},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["accepted"] == 2
    assert (tmp_path / "uploaded" / "cardboard").exists()


def test_upload_data_rejects_non_zip_file():
    with TestClient(app) as client:
        response = client.post(
            "/upload-data",
            files={"file": ("data.txt", b"not a zip", "text/plain")},
            headers={"X-API-Key": settings.api_key},
        )
    assert response.status_code == 400


def test_upload_data_reports_path_traversal_without_extracting(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "upload_directory", str(tmp_path / "uploaded"))
    zip_bytes = _make_zip_bytes({"../../evil.jpg": make_image_bytes()})

    with TestClient(app) as client:
        response = client.post(
            "/upload-data",
            files={"file": ("data.zip", zip_bytes, "application/zip")},
            headers={"X-API-Key": settings.api_key},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["accepted"] == 0
    assert len(body["unsafe_path"]) == 1
    assert not (tmp_path / "uploaded").exists() or not any((tmp_path / "uploaded").rglob("evil.jpg"))
