"""
Feedback Agent
===============
Records CSM-observed outcomes (retained / churned / unknown) against a
completed pipeline run and feeds them back into the platform:

  1. Persists the feedback record to Redis (90-day TTL)
  2. Updates the ChromaDB interaction record with the actual outcome
  3. Updates the A/B experiment with the observed retention metric
  4. Writes an audit log entry
  5. Increments the global feedback counter and fires a retraining
     signal when the configured threshold is reached

Usage (programmatic or via FastAPI /hitl/feedback endpoint):
    agent = FeedbackAgent()
    result = agent.record_feedback(
        run_id="abc123",
        customer_id="cust-uuid",
        outcome="retained",
        notes="CSM called — renewed 12-month contract",
        submitted_by="csm@example.com",
    )
"""

from __future__ import annotations

import json
import time
from typing import Any, Literal

import structlog

from config.settings import settings
from memory.redis_state import RedisStateManager
from memory.vector_store import VectorStore
from src.feedback.audit_log import AuditLog

logger = structlog.get_logger(__name__)

Outcome = Literal["retained", "churned", "unknown"]

_FEEDBACK_COUNTER_KEY = "churn:feedback:total_count"
_FEEDBACK_KEY_PREFIX = "churn:feedback"


class FeedbackAgent:
    """Records CSM outcomes and triggers platform feedback loops."""

    def __init__(
        self,
        redis_store: RedisStateManager | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self._redis = redis_store or RedisStateManager()
        self._vector = vector_store or VectorStore()
        self._retrain_threshold = settings.model.feedback_retrain_threshold
        self._audit = AuditLog()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def record_feedback(
        self,
        run_id: str,
        customer_id: str,
        outcome: Outcome,
        notes: str = "",
        submitted_by: str = "api",
        ab_group: str | None = None,
        crm_action_id: str | None = None,
        churn_prob: float | None = None,
    ) -> dict[str, Any]:
        """
        Record the actual customer outcome after a retention intervention.

        Args:
            run_id:        Pipeline run ID this feedback is linked to.
            customer_id:   Customer the feedback is for.
            outcome:       "retained" | "churned" | "unknown"
            notes:         Optional free-text CSM note.
            submitted_by:  Who submitted the feedback (email / username).
            ab_group:      A/B group from the run ("treatment" | "control").
            crm_action_id: CRM action that was dispatched.
            churn_prob:    Original model churn probability for analysis.

        Returns:
            Dict with feedback_id, status, outcome, timestamp,
            and retrain_triggered flag.
        """
        feedback_id = f"fb-{run_id[:8]}-{int(time.time())}"
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        record: dict[str, Any] = {
            "feedback_id": feedback_id,
            "run_id": run_id,
            "customer_id": customer_id,
            "outcome": outcome,
            "notes": notes,
            "submitted_by": submitted_by,
            "ab_group": ab_group,
            "crm_action_id": crm_action_id,
            "churn_prob": churn_prob,
            "timestamp": timestamp,
        }

        self._persist_feedback(record)
        self._update_chromadb_outcome(customer_id, run_id, outcome)

        if ab_group:
            self._update_ab_outcome(run_id, customer_id, ab_group, outcome)

        self._audit.log_feedback(
            feedback_id=feedback_id,
            run_id=run_id,
            customer_id=customer_id,
            outcome=outcome,
            submitted_by=submitted_by,
        )

        retrain_needed = self._check_retrain_trigger()

        logger.info(
            "Feedback recorded",
            feedback_id=feedback_id,
            customer_id=customer_id,
            outcome=outcome,
            retrain_triggered=retrain_needed,
        )

        return {
            "feedback_id": feedback_id,
            "status": "recorded",
            "outcome": outcome,
            "timestamp": timestamp,
            "retrain_triggered": retrain_needed,
        }

    def get_feedback_stats(self) -> dict[str, Any]:
        """Return aggregate feedback and retraining threshold statistics."""
        count = self._get_counter()
        return {
            "total_feedback_count": count,
            "retrain_threshold": self._retrain_threshold,
            "feedback_until_retrain": max(0, self._retrain_threshold - count),
        }

    # ------------------------------------------------------------------ #
    # Storage helpers
    # ------------------------------------------------------------------ #

    def _persist_feedback(self, record: dict[str, Any]) -> None:
        key = f"{_FEEDBACK_KEY_PREFIX}:{record['run_id']}:{record['customer_id']}"
        payload = json.dumps(record, default=str)
        if self._redis._client:
            self._redis._client.setex(key, 86400 * 90, payload)
            self._redis._client.incr(_FEEDBACK_COUNTER_KEY)
        else:
            self._redis._fallback[key] = payload
            count = int(self._redis._fallback.get(_FEEDBACK_COUNTER_KEY, "0"))
            self._redis._fallback[_FEEDBACK_COUNTER_KEY] = str(count + 1)

    def _update_chromadb_outcome(
        self,
        customer_id: str,
        run_id: str,
        outcome: str,
    ) -> None:
        try:
            self._vector.upsert_customer_interaction(
                customer_id=customer_id,
                interaction={
                    "customer_id": customer_id,
                    "run_id": run_id,
                    "outcome": outcome,
                    "feedback_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            )
        except Exception as exc:
            logger.warning("ChromaDB outcome update failed", error=str(exc))

    def _update_ab_outcome(
        self,
        run_id: str,
        customer_id: str,
        ab_group: str,
        outcome: str,
    ) -> None:
        try:
            from src.optimization.ab_testing import ABTestingManager
            ab = ABTestingManager()
            ab.log_outcome(
                customer_id=customer_id,
                experiment_id="retention_v1",
                group=ab_group,
                outcome_value=1.0 if outcome == "retained" else 0.0,
                metadata={"run_id": run_id, "outcome": outcome},
            )
        except Exception as exc:
            logger.debug("A/B outcome update skipped", error=str(exc))

    # ------------------------------------------------------------------ #
    # Retraining trigger
    # ------------------------------------------------------------------ #

    def _get_counter(self) -> int:
        if self._redis._client:
            return int(self._redis._client.get(_FEEDBACK_COUNTER_KEY) or 0)
        return int(self._redis._fallback.get(_FEEDBACK_COUNTER_KEY, "0"))

    def _check_retrain_trigger(self) -> bool:
        """
        Fire a retraining signal when accumulated feedback reaches the
        threshold.  Resets the counter after triggering.

        Returns True if retraining was triggered.
        """
        try:
            count = self._get_counter()
            if count >= self._retrain_threshold:
                self._trigger_retraining(count)
                if self._redis._client:
                    self._redis._client.set(_FEEDBACK_COUNTER_KEY, "0")
                else:
                    self._redis._fallback[_FEEDBACK_COUNTER_KEY] = "0"
                return True
        except Exception as exc:
            logger.debug("Retrain check failed", error=str(exc))
        return False

    def _trigger_retraining(self, feedback_count: int) -> None:
        """
        Publish retraining signal to Redis.
        Phase 8 will replace this with a Celery task enqueue.
        """
        key = "churn:retrain:trigger"
        payload = json.dumps({
            "triggered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "feedback_count": feedback_count,
            "reason": "feedback_threshold_reached",
        })
        if self._redis._client:
            self._redis._client.setex(key, 86400, payload)
        else:
            self._redis._fallback[key] = payload

        self._audit.log_retrain_trigger(feedback_count=feedback_count)

        logger.warning(
            "Retraining trigger fired",
            feedback_count=feedback_count,
            threshold=self._retrain_threshold,
        )
