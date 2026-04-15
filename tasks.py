"""
Celery Task Definitions
========================
Async tasks for the Churn Intelligence Platform.

All tasks are discoverable by the Celery worker via celery_app.include=["tasks"].
"""

from __future__ import annotations

from typing import Any

import structlog

from celery_app import celery_app

logger = structlog.get_logger(__name__)


@celery_app.task(
    name="tasks.run_churn_pipeline",
    bind=True,
    max_retries=2,
    default_retry_delay=10,
)
def run_churn_pipeline(
    self: Any,
    customer_id: str,
    triggered_by: str = "celery",
) -> dict[str, Any]:
    """
    Run the full churn analysis pipeline for a single customer.

    Args:
        customer_id: Customer to analyse.
        triggered_by: Audit label for the triggering source.

    Returns:
        Serialisable summary of the pipeline result.
    """
    from agents.orchestrator import ChurnOrchestrator

    try:
        orchestrator = ChurnOrchestrator()
        result = orchestrator.run_sync(
            customer_id,
            triggered_by=triggered_by,
            run_id=self.request.id,  # store Redis state under the Celery task ID
        )

        prediction = result.get("prediction", {})
        explanation = result.get("explanation", {})

        return {
            "customer_id": customer_id,
            "status": "complete" if not result.get("should_abort") else "error",
            "churn_probability": prediction.get("churn_probability"),
            "risk_tier": prediction.get("risk_tier"),
            "narrative": explanation.get("narrative_text"),
            "completed_steps": result.get("completed_steps", []),
            "errors": result.get("errors", []),
        }

    except Exception as exc:
        logger.error("Celery task failed", customer_id=customer_id, error=str(exc))
        raise self.retry(exc=exc)


@celery_app.task(
    name="tasks.run_batch_pipeline",
    bind=True,
)
def run_batch_pipeline(
    self: Any,
    customer_ids: list[str],
    triggered_by: str = "batch",
) -> dict[str, Any]:
    """
    Run pipeline for a list of customers (batch mode).
    Executes sequentially within the task; each customer is processed
    and results accumulated.
    """
    from agents.orchestrator import ChurnOrchestrator

    orchestrator = ChurnOrchestrator()
    results: list[dict[str, Any]] = []
    errors: list[str] = []

    for i, customer_id in enumerate(customer_ids):
        try:
            # Update task progress meta
            self.update_state(
                state="PROGRESS",
                meta={"current": i + 1, "total": len(customer_ids), "customer_id": customer_id},
            )
            result = orchestrator.run_sync(customer_id, triggered_by=triggered_by)
            prediction = result.get("prediction", {})
            results.append({
                "customer_id": customer_id,
                "churn_probability": prediction.get("churn_probability"),
                "risk_tier": prediction.get("risk_tier"),
                "errors": result.get("errors", []),
            })
        except Exception as exc:
            errors.append(f"{customer_id}: {exc}")
            logger.warning("Batch task: customer failed", customer_id=customer_id, error=str(exc))

    return {
        "status": "complete",
        "total": len(customer_ids),
        "processed": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors,
    }
