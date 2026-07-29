from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from api.dependencies import require_api_key
from api.schemas import UploadValidationResponse
from src.config import DEFAULT_CLASS_NAMES, settings
from src.preprocessing import safe_extract_zip

logger = logging.getLogger(__name__)
router = APIRouter(tags=["upload"])

MAX_UPLOAD_BYTES = settings.max_upload_size_mb * 1024 * 1024


@router.post("/upload-data", response_model=UploadValidationResponse, dependencies=[Depends(require_api_key)])
async def upload_data(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Bulk upload must be a .zip archive.")

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded ZIP exceeds the maximum allowed size.")

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = Path(tmp.name)

    try:
        upload_dir = settings.resolve(settings.upload_directory)
        report = safe_extract_zip(tmp_path, upload_dir, known_classes=DEFAULT_CLASS_NAMES)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    message = (
        f"Accepted {report.accepted}/{report.total_files} images. "
        f"Press 'Retrain' to fold this data into the next training run."
        if report.accepted
        else "No valid images were accepted from this upload."
    )

    return UploadValidationResponse(**report.to_dict(), message=message)
