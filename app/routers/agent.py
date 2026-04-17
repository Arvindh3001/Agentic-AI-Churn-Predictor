"""
Agent Router
=============
POST /api/v1/agent/analyse         — trigger single-customer pipeline
GET  /api/v1/agent/status/{run_id} — poll pipeline progress + partial results
POST /api/v1/agent/batch           — submit batch analysis via Celery
GET  /api/v1/agent/batch/{task_id} — poll batch task progress
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.middleware.auth import CurrentUser

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["agent"])


# ------------------------------------------------------------------ #
# Request models
# ------------------------------------------------------------------ #

class AnalyseRequest(BaseModel):
    customer_id: str
    async_mode: bool = True  # True → Celery task; False → inline sync (dev)


class BatchRequest(BaseModel):
    customer_ids: list[str]


# ------------------------------------------------------------------ #
# Single-customer analysis
# ------------------------------------------------------------------ #

@router.post("/agent/analyse")
async def analyse_customer(
    payload: AnalyseRequest,
    current_user: CurrentUser,
) -> dict[str, Any]:
    """
    Trigger the full 6-agent churn pipeline for a single customer.

    async_mode=true  → queues a Celery task; returns run_id immediately.
                       Poll /agent/status/{run_id} or connect to
                       ws://host/ws/agent/{run_id} for live progress.

    async_mode=false → runs synchronously (blocks until complete).
                       Useful in dev without a Celery worker running.
    """
    triggered_by = f"api:{current_user['username']}"

    if payload.async_mode:
        try:
            from agents.orchestrator import ChurnOrchestrator
            orch = ChurnOrchestrator()
            run_id = orch.run_async(payload.customer_id, triggered_by=triggered_by)
            logger.info(
                "Analysis queued",
                customer_id=payload.customer_id,
                run_id=run_id,
                user=current_user["username"],
            )
            return {
                "run_id": run_id,
                "status": "queued",
                "customer_id": payload.customer_id,
                "mode": "async",
            }
        except Exception as exc:
            logger.warning("Async dispatch failed — falling back to sync", error=str(exc))

    # Synchronous fallback (no Celery required)
    run_id = str(uuid.uuid4())[:8]
    logger.info(
        "Analysis running synchronously",
        customer_id=payload.customer_id,
        run_id=run_id,
    )
    from agents.orchestrator import ChurnOrchestrator

    def _run() -> dict:
        orch = ChurnOrchestrator()
        return orch.run_sync(payload.customer_id, triggered_by=triggered_by, run_id=run_id)

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _run)
    except Exception as exc:
        logger.error("Sync pipeline failed", error=str(exc), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

    prediction = result.get("prediction") or {}
    explanation = result.get("explanation") or {}
    retention = result.get("retention_plan") or {}
    hitl = result.get("hitl_decision") or {}

    return {
        "run_id": run_id,
        "status": "complete" if not result.get("should_abort") else "error",
        "customer_id": payload.customer_id,
        "mode": "sync",
        "churn_probability": prediction.get("churn_probability"),
        "risk_tier": prediction.get("risk_tier"),
        "confidence_interval": prediction.get("confidence_interval"),
        "narrative": explanation.get("narrative_text"),
        "top_risk_factors": explanation.get("top_risk_factors", []),
        "retention_plan": retention,
        "hitl_decision": hitl,
        "completed_steps": result.get("completed_steps", []),
        "errors": result.get("errors", []),
    }


# ------------------------------------------------------------------ #
# Status polling
# ------------------------------------------------------------------ #

@router.get("/agent/status/{run_id}")
async def get_pipeline_status(
    run_id: str,
    current_user: CurrentUser,
) -> dict[str, Any]:
    """
    Poll the current status and partial results of a pipeline run.

    For real-time progress, prefer the WebSocket endpoint:
        ws://host/ws/agent/{run_id}
    """
    from agents.orchestrator import ChurnOrchestrator
    orch = ChurnOrchestrator()
    return orch.get_status(run_id)


# ------------------------------------------------------------------ #
# Batch analysis
# ------------------------------------------------------------------ #

@router.post("/agent/batch")
async def batch_analyse(
    payload: BatchRequest,
    current_user: CurrentUser,
) -> dict[str, Any]:
    """
    Submit a batch analysis for up to 100 customers via Celery.

    Returns a task_id — poll /agent/batch/{task_id} for progress.
    """
    if len(payload.customer_ids) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 customers per batch request",
        )
    if not payload.customer_ids:
        raise HTTPException(status_code=400, detail="customer_ids cannot be empty")

    try:
        from celery_app import celery_app
        task = celery_app.send_task(
            "tasks.run_batch_pipeline",
            args=[payload.customer_ids],
            kwargs={"triggered_by": f"api:{current_user['username']}"},
        )
        logger.info(
            "Batch task queued",
            task_id=task.id,
            total=len(payload.customer_ids),
            user=current_user["username"],
        )
        return {
            "task_id": task.id,
            "status": "queued",
            "total": len(payload.customer_ids),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Celery queue unavailable: {exc}. Start a Celery worker to use batch mode.",
        )


@router.get("/agent/batch/{task_id}")
async def get_batch_status(
    task_id: str,
    current_user: CurrentUser,
) -> dict[str, Any]:
    """Poll the status of a running batch Celery task."""
    try:
        from celery_app import celery_app
        result = celery_app.AsyncResult(task_id)
        response: dict[str, Any] = {"task_id": task_id, "status": result.status}
        if result.status == "PROGRESS":
            response["progress"] = result.info
        elif result.status == "SUCCESS":
            response["result"] = result.result
        elif result.status == "FAILURE":
            response["error"] = str(result.result)
        return response
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))
