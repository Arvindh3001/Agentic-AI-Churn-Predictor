"""
Audit Log
==========
Append-only JSONL audit trail for all HITL decisions, feedback records,
CRM actions dispatched post-approval, and model retraining triggers.

Each entry is a single JSON line with an ISO timestamp, written to:
    logs/audit.jsonl   (created automatically)

Usage:
    from src.feedback.audit_log import AuditLog

    audit = AuditLog()
    audit.log_hitl_decision(run_id="abc", customer_id="C-1", ...)
    audit.log_feedback(feedback_id="fb-abc", ...)
    recent = audit.read_recent(n=20)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

_AUDIT_LOG_PATH = Path("logs/audit.jsonl")


class AuditLog:
    """Append-only structured audit logger backed by a JSONL file."""

    def __init__(self, log_path: Path = _AUDIT_LOG_PATH) -> None:
        self._path = log_path
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass  # best-effort; write failures are handled per-call

    # ------------------------------------------------------------------ #
    # Public log methods
    # ------------------------------------------------------------------ #

    def log_hitl_decision(
        self,
        run_id: str,
        customer_id: str,
        risk_tier: str,
        churn_prob: float,
        status: str,
        decided_by: str,
        slack_sent: bool = False,
        notes: str = "",
    ) -> None:
        """Log a HITL approve/reject/auto-approve decision."""
        self._append({
            "event": "hitl_decision",
            "run_id": run_id,
            "customer_id": customer_id,
            "risk_tier": risk_tier,
            "churn_prob": round(churn_prob, 4),
            "status": status,
            "decided_by": decided_by,
            "slack_sent": slack_sent,
            "notes": notes,
        })

    def log_feedback(
        self,
        feedback_id: str,
        run_id: str,
        customer_id: str,
        outcome: str,
        submitted_by: str,
    ) -> None:
        """Log a CSM outcome feedback record."""
        self._append({
            "event": "feedback_recorded",
            "feedback_id": feedback_id,
            "run_id": run_id,
            "customer_id": customer_id,
            "outcome": outcome,
            "submitted_by": submitted_by,
        })

    def log_crm_action(
        self,
        action_id: str,
        customer_id: str,
        action_type: str,
        status: str,
        ab_group: str,
        cost_usd: float,
    ) -> None:
        """Log a CRM action dispatched after HITL approval."""
        self._append({
            "event": "crm_action",
            "action_id": action_id,
            "customer_id": customer_id,
            "action_type": action_type,
            "status": status,
            "ab_group": ab_group,
            "cost_usd": round(cost_usd, 2),
        })

    def log_retrain_trigger(self, feedback_count: int) -> None:
        """Log that a model retraining signal was fired."""
        self._append({
            "event": "retrain_trigger",
            "feedback_count": feedback_count,
            "reason": "feedback_threshold_reached",
        })

    def read_recent(self, n: int = 50) -> list[dict[str, Any]]:
        """Return the last *n* audit entries from the log file."""
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
            return [json.loads(line) for line in lines[-n:] if line.strip()]
        except FileNotFoundError:
            return []
        except Exception as exc:
            logger.warning("Audit log read failed", error=str(exc))
            return []

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _append(self, record: dict[str, Any]) -> None:
        record["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        line = json.dumps(record, default=str)
        try:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as exc:
            logger.warning(
                "Audit log write failed",
                error=str(exc),
                audit_event=record.get("event"),
            )
        logger.debug("Audit entry written", audit_event=record.get("event"), run_id=record.get("run_id"))
