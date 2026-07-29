from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import require_api_key
from api.schemas import RetrainRequest, RetrainResponse, TrainingStatusResponse

router = APIRouter(tags=["retraining"])


@router.post("/retrain", response_model=RetrainResponse, dependencies=[Depends(require_api_key)])
def trigger_retrain(request: RetrainRequest):
    from worker.tasks import retrain_model

    async_result = retrain_model.delay(
        model_name=request.model_name, epochs=request.epochs, fine_tune_epochs=request.fine_tune_epochs
    )
    return RetrainResponse(job_id=async_result.id, status="queued")


@router.get("/training-status/{job_id}", response_model=TrainingStatusResponse)
def training_status(job_id: str):
    import redis as redis_lib

    from worker.celery_app import celery_app

    async_result = celery_app.AsyncResult(job_id)

    try:
        state = async_result.state
    except redis_lib.exceptions.ConnectionError as exc:
        raise HTTPException(
            status_code=503, detail="Task queue backend (Redis) is unavailable; cannot check job status."
        ) from exc

    if state == "PENDING":
        raise HTTPException(status_code=404, detail="Unknown job id, or the job has not started yet.")

    stage = None
    result = None

    if state == "PROGRESS":
        stage = (async_result.info or {}).get("stage")
    elif state == "SUCCESS":
        result = async_result.result
    elif state == "FAILURE":
        result = {"error": str(async_result.info)}

    return TrainingStatusResponse(job_id=job_id, state=state, stage=stage, result=result)
