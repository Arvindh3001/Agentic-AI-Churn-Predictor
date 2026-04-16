"""
HITL Webhook Router
=====================
FastAPI router handling Slack interactive payloads and direct CSM
feedback submissions.

Routes:
    POST /hitl/decision            — CSM approve / reject (direct API or Slack button)
    POST /hitl/slack/interactive   — Slack interactive component callback
    POST /hitl/feedback            — Record CSM outcome after intervention
    GET  /hitl/status/{run_id}     — Poll HITL decision status for a pipeline run
    GET  /hitl/feedback/stats      — Aggregate feedback and retraining statistics
    GET  /hitl/audit               — Recent audit log entries (last N)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
import urllib.parse
from typing import Any

import structlog
from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request
from pydantic import BaseModel

from agents.feedback_agent import FeedbackAgent
from config.settings import settings
from memory.redis_state import RedisStateManager
from src.feedback.audit_log import AuditLog

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/hitl", tags=["HITL"])

# Module-level singletons (lazily connected — won't crash import if Redis is down)
_redis = RedisStateManager()
_feedback_agent = FeedbackAgent(redis_store=_redis)
_audit = AuditLog()


# ------------------------------------------------------------------ #
# Request / response models
# ------------------------------------------------------------------ #

class HITLDecisionRequest(BaseModel):
    run_id: str
    decision: str           # "approved" | "rejected"
    decided_by: str = "csm"
    notes: str = ""


class FeedbackRequest(BaseModel):
    run_id: str
    customer_id: str
    outcome: str            # "retained" | "churned" | "unknown"
    notes: str = ""
    submitted_by: str = "api"
    ab_group: str | None = None
    crm_action_id: str | None = None
    churn_prob: float | None = None


# ------------------------------------------------------------------ #
# HITL Decision — direct API call
# ------------------------------------------------------------------ #

@router.post("/decision")
async def record_hitl_decision(payload: HITLDecisionRequest) -> dict[str, Any]:
    """
    Record a CSM approve / reject decision for a pending HITL review.

    Called by:
      - CSM dashboard direct API call
      - Slack interactive button callback (via /hitl/slack/interactive)

    The HITLAgent polling loop reads this decision from Redis at:
        churn:hitl:{run_id}:decision
    """
    if payload.decision not in ("approved", "rejected"):
        raise HTTPException(
            status_code=400,
            detail="decision must be 'approved' or 'rejected'",
        )

    _write_decision(
        run_id=payload.run_id,
        status=payload.decision,
        decided_by=payload.decided_by,
        notes=payload.notes,
    )

    _audit.log_hitl_decision(
        run_id=payload.run_id,
        customer_id="api_caller",
        risk_tier="CRITICAL",
        churn_prob=0.0,
        status=payload.decision,
        decided_by=payload.decided_by,
        notes=payload.notes,
    )

    logger.info(
        "HITL decision recorded",
        run_id=payload.run_id,
        decision=payload.decision,
        decided_by=payload.decided_by,
    )

    return {
        "status": "recorded",
        "run_id": payload.run_id,
        "decision": payload.decision,
        "decided_by": payload.decided_by,
    }


# ------------------------------------------------------------------ #
# Slack Interactive Component Callback
# ------------------------------------------------------------------ #

@router.post("/slack/interactive")
async def slack_interactive_callback(request: Request) -> dict[str, Any]:
    """
    Handle Slack interactive component payloads (Approve / Reject button clicks).

    Verifies the Slack request signature (X-Slack-Signature header) before
    processing.  Writes the decision to Redis to unblock the HITLAgent.
    """
    body_bytes = await request.body()

    if settings.slack.slack_signing_secret:
        if not _verify_slack_signature(request.headers, body_bytes):
            raise HTTPException(status_code=401, detail="Invalid Slack signature")

    try:
        body_str = body_bytes.decode("utf-8")
        # Slack sends: payload=<url-encoded-json>
        if body_str.startswith("payload="):
            payload_str = urllib.parse.unquote(body_str[len("payload="):])
        else:
            payload_str = body_str

        slack_payload = json.loads(payload_str)
        actions = slack_payload.get("actions", [])
        slack_user = slack_payload.get("user", {}).get("name", "slack_user")

        for action in actions:
            value_str = action.get("value", "{}")
            try:
                value = json.loads(value_str)
            except (json.JSONDecodeError, TypeError):
                continue

            run_id = value.get("run_id", "")
            decision = value.get("decision", "")

            if run_id and decision in ("approved", "rejected"):
                decided_by = f"slack:{slack_user}"
                _write_decision(
                    run_id=run_id,
                    status=decision,
                    decided_by=decided_by,
                    notes=f"Decision via Slack button by {slack_user}",
                )
                logger.info(
                    "HITL decision from Slack",
                    run_id=run_id,
                    decision=decision,
                    user=slack_user,
                )

        # Slack expects a 200 response quickly
        return {"status": "ok"}

    except Exception as exc:
        logger.error("Slack interactive callback failed", error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc))


# ------------------------------------------------------------------ #
# Feedback endpoint
# ------------------------------------------------------------------ #

@router.post("/feedback")
async def record_feedback(
    payload: FeedbackRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    """
    Record a CSM-observed customer outcome after a retention intervention.

    Triggers model retraining signal if the cumulative feedback count
    reaches the configured threshold (settings.model.feedback_retrain_threshold).
    """
    if payload.outcome not in ("retained", "churned", "unknown"):
        raise HTTPException(
            status_code=400,
            detail="outcome must be 'retained', 'churned', or 'unknown'",
        )

    result = _feedback_agent.record_feedback(
        run_id=payload.run_id,
        customer_id=payload.customer_id,
        outcome=payload.outcome,
        notes=payload.notes,
        submitted_by=payload.submitted_by,
        ab_group=payload.ab_group,
        crm_action_id=payload.crm_action_id,
        churn_prob=payload.churn_prob,
    )

    return result


# ------------------------------------------------------------------ #
# Status & audit read endpoints
# ------------------------------------------------------------------ #

@router.get("/status/{run_id}")
async def get_hitl_status(run_id: str) -> dict[str, Any]:
    """Check the current HITL decision status for a pipeline run."""
    key = f"churn:hitl:{run_id}:decision"
    if _redis._client:
        raw = _redis._client.get(key)
    else:
        raw = _redis._fallback.get(key)

    if raw:
        decision = json.loads(raw)
        return {"run_id": run_id, "found": True, **decision}
    return {"run_id": run_id, "found": False, "status": "pending"}


@router.get("/feedback/stats")
async def get_feedback_stats() -> dict[str, Any]:
    """Return aggregate feedback count and retraining threshold status."""
    return _feedback_agent.get_feedback_stats()


@router.get("/audit")
async def get_audit_log(n: int = 50) -> dict[str, Any]:
    """Return the last *n* entries from the audit log."""
    entries = _audit.read_recent(n=n)
    return {"count": len(entries), "entries": entries}


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _write_decision(
    run_id: str,
    status: str,
    decided_by: str,
    notes: str = "",
) -> None:
    key = f"churn:hitl:{run_id}:decision"
    payload = json.dumps({
        "status": status,
        "decided_by": decided_by,
        "decided_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "notes": notes,
    })
    if _redis._client:
        _redis._client.setex(key, 86400, payload)
    else:
        _redis._fallback[key] = payload


def _verify_slack_signature(headers: Any, body: bytes) -> bool:
    """Verify Slack request signature using HMAC-SHA256."""
    try:
        timestamp = headers.get("x-slack-request-timestamp", "")
        slack_sig = headers.get("x-slack-signature", "")

        # Reject stale requests (> 5 minutes)
        if abs(time.time() - float(timestamp)) > 300:
            return False

        base_string = f"v0:{timestamp}:{body.decode('utf-8')}"
        secret = settings.slack.slack_signing_secret.encode("utf-8")
        computed = "v0=" + hmac.new(
            secret,
            base_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(computed, slack_sig)
    except Exception:
        return False
