"""
Prometheus custom metrics for the Churn Intelligence Platform.

Exposes:
  - churn_predictions_total        (counter)  — predictions served
  - churn_probability_histogram    (histogram) — distribution of churn probs
  - agent_pipeline_duration_seconds (histogram) — end-to-end pipeline latency
  - hitl_pending_count             (gauge)    — open HITL approvals
  - model_auc_roc_current          (gauge)    — live AUC-ROC
  - feature_drift_psi              (gauge)    — PSI per feature
  - celery_tasks_pending           (gauge)    — task queue depth
  - http_requests_total            (counter)  — API request counts by route
  - http_request_duration_seconds  (histogram) — API latency by route

Usage (add to app/main.py):
    from monitoring.prometheus_metrics import setup_metrics
    setup_metrics(app)

Then mount:
    from prometheus_client import make_asgi_app
    app.mount("/metrics", make_asgi_app())
"""
from __future__ import annotations

import time
from typing import Callable

try:
    from prometheus_client import (  # type: ignore
        Counter,
        Gauge,
        Histogram,
        start_http_server,
        REGISTRY,
    )
    _PROMETHEUS_OK = True
except ImportError:
    _PROMETHEUS_OK = False

import logging

logger = logging.getLogger(__name__)

if _PROMETHEUS_OK:
    # ── Prediction metrics ────────────────────────────────────────
    CHURN_PREDICTIONS_TOTAL = Counter(
        "churn_predictions_total",
        "Total number of churn predictions served",
        ["risk_tier", "model_version"],
    )

    CHURN_PROBABILITY_HISTOGRAM = Histogram(
        "churn_probability",
        "Distribution of predicted churn probabilities",
        buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0],
    )

    # ── Agent pipeline metrics ────────────────────────────────────
    AGENT_PIPELINE_DURATION = Histogram(
        "agent_pipeline_duration_seconds",
        "End-to-end agent pipeline duration",
        ["pipeline_type"],
        buckets=[0.5, 1, 2, 5, 10, 20, 30, 60],
    )

    AGENT_PIPELINE_ERRORS = Counter(
        "agent_pipeline_errors_total",
        "Total agent pipeline errors",
        ["step", "error_type"],
    )

    # ── HITL metrics ──────────────────────────────────────────────
    HITL_PENDING_COUNT = Gauge(
        "hitl_pending_count",
        "Number of HITL decisions currently awaiting human review",
    )

    HITL_DECISIONS_TOTAL = Counter(
        "hitl_decisions_total",
        "Total HITL decisions",
        ["decision"],  # approved / rejected / auto_approved
    )

    # ── ML model quality metrics ──────────────────────────────────
    MODEL_AUC_ROC = Gauge(
        "model_auc_roc_current",
        "Current champion model AUC-ROC on holdout set",
        ["model_version"],
    )

    FEATURE_DRIFT_PSI = Gauge(
        "feature_drift_psi",
        "Population Stability Index per feature",
        ["feature_name"],
    )

    # ── HTTP request metrics ──────────────────────────────────────
    HTTP_REQUESTS_TOTAL = Counter(
        "http_requests_total",
        "Total HTTP requests",
        ["method", "endpoint", "status_code"],
    )

    HTTP_REQUEST_DURATION = Histogram(
        "http_request_duration_seconds",
        "HTTP request duration in seconds",
        ["method", "endpoint"],
        buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    )

    # ── Celery queue depth ────────────────────────────────────────
    CELERY_TASKS_PENDING = Gauge(
        "celery_tasks_pending",
        "Number of tasks waiting in the Celery queue",
        ["queue"],
    )

    # ── Retention optimizer metrics ───────────────────────────────
    OPTIMIZER_RUNS_TOTAL = Counter(
        "optimizer_runs_total",
        "Total retention optimizer runs",
    )

    OPTIMIZER_ROI = Histogram(
        "optimizer_roi",
        "Estimated ROI from retention optimizer runs",
        buckets=[0.5, 1, 1.5, 2, 3, 5, 10, 20],
    )


# ── FastAPI middleware integration ────────────────────────────────────────────

def setup_metrics(app):  # type: ignore[no-untyped-def]
    """
    Attach Prometheus metrics middleware to a FastAPI app.
    Also mounts /metrics endpoint using prometheus_client.
    """
    if not _PROMETHEUS_OK:
        logger.warning("prometheus_client not installed — metrics disabled")
        return

    from fastapi import Request, Response
    from prometheus_client import make_asgi_app

    # Mount /metrics endpoint
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    @app.middleware("http")
    async def track_requests(request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        # Normalise path (replace dynamic segments like /customers/C-1234 → /customers/{id})
        path = request.url.path
        for segment in path.split("/"):
            if segment and not segment.replace("-", "").replace("_", "").isalpha():
                path = path.replace(segment, "{id}", 1)
                break

        HTTP_REQUESTS_TOTAL.labels(
            method=request.method,
            endpoint=path,
            status_code=response.status_code,
        ).inc()
        HTTP_REQUEST_DURATION.labels(
            method=request.method,
            endpoint=path,
        ).observe(duration)

        return response


# ── Helper functions for business metrics ─────────────────────────────────────

def record_prediction(churn_probability: float, risk_tier: str, model_version: str = "champion") -> None:
    if not _PROMETHEUS_OK:
        return
    CHURN_PREDICTIONS_TOTAL.labels(risk_tier=risk_tier, model_version=model_version).inc()
    CHURN_PROBABILITY_HISTOGRAM.observe(churn_probability)


def record_pipeline_duration(duration_seconds: float, pipeline_type: str = "single") -> None:
    if not _PROMETHEUS_OK:
        return
    AGENT_PIPELINE_DURATION.labels(pipeline_type=pipeline_type).observe(duration_seconds)


def set_hitl_pending(count: int) -> None:
    if not _PROMETHEUS_OK:
        return
    HITL_PENDING_COUNT.set(count)


def record_hitl_decision(decision: str) -> None:
    if not _PROMETHEUS_OK:
        return
    HITL_DECISIONS_TOTAL.labels(decision=decision).inc()


def set_model_auc(auc: float, model_version: str = "champion") -> None:
    if not _PROMETHEUS_OK:
        return
    MODEL_AUC_ROC.labels(model_version=model_version).set(auc)


def set_feature_drift(feature_name: str, psi: float) -> None:
    if not _PROMETHEUS_OK:
        return
    FEATURE_DRIFT_PSI.labels(feature_name=feature_name).set(psi)


def set_celery_queue_depth(queue: str, count: int) -> None:
    if not _PROMETHEUS_OK:
        return
    CELERY_TASKS_PENDING.labels(queue=queue).set(count)
