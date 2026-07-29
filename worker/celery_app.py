"""Celery application definition. Retraining runs here, never inside the API process,
so prediction requests stay available while a candidate model trains."""
from __future__ import annotations

from celery import Celery

from src.config import settings

celery_app = Celery(
    "rurangasort",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=60 * 60 * 24,
)

celery_app.autodiscover_tasks(["worker"])
