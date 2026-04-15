"""
Celery Application
===================
Configures the Celery worker for async agent task execution.
Broker and result backend are both Redis.

Start worker:
    celery -A celery_app worker --loglevel=info --concurrency=4

Tasks registered:
    tasks.run_churn_pipeline  — full agent pipeline for one customer
    tasks.run_batch_pipeline  — batch analysis for a list of customers
"""

from __future__ import annotations

import structlog
from celery import Celery

from config.settings import settings

logger = structlog.get_logger(__name__)

celery_app = Celery(
    "churn_platform",
    broker=settings.db.redis_url,
    backend=settings.db.redis_url,
    include=["tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=86400,          # 24 hours
    task_track_started=True,
    worker_prefetch_multiplier=1,  # one task at a time per worker slot
    task_acks_late=True,           # re-queue on worker crash
    timezone="UTC",
    enable_utc=True,
)
