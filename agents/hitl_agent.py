"""
HITL Agent (Human-in-the-Loop)
================================
LangGraph node that gates CRITICAL-tier retention actions on CSM approval
before CRM dispatch.  For HIGH-tier customers it sends an informational
Slack alert and auto-approves immediately.

Graph position:
    retention_strategist → hitl → END

Behaviour by risk tier:
  CRITICAL  — Posts Slack HITL request with Approve/Reject buttons.
               Polls Redis for a CSM decision up to hitl_timeout_seconds.
               Auto-approves on timeout (logs a warning).
               On approval: dispatches the CRM action.
               On rejection: records rejection, skips CRM dispatch.
  HIGH      — Posts Slack informational alert, auto-approves immediately.
               (CRM was already dispatched by RetentionStrategistAgent.)
  MEDIUM    — No notification, no gate; passes through as auto-approved.

The approve / reject decision is written to Redis by the FastAPI endpoint:
    POST /hitl/decision
and read here from the key:
    churn:hitl:{run_id}:decision
"""

from __future__ import annotations

import json
import time
from typing import Any

import structlog

from agents.state import AgentState, HITLDecision
from agents.tools.slack_tool import send_alert_notification, send_hitl_notification
from config.settings import settings
from memory.redis_state import RedisStateManager
from src.feedback.audit_log import AuditLog

logger = structlog.get_logger(__name__)

_POLL_INTERVAL_S: float = 2.0


class HITLAgent:
    """HITL review gate node for the LangGraph pipeline."""

    def __init__(
        self,
        redis_store: RedisStateManager | None = None,
        webhook_base_url: str = "http://localhost:8000",
    ) -> None:
        self._redis = redis_store or RedisStateManager()
        self._audit = AuditLog()
        self._timeout = settings.model.hitl_timeout_seconds
        self._webhook_base_url = webhook_base_url

    # ------------------------------------------------------------------ #
    # LangGraph node entrypoint
    # ------------------------------------------------------------------ #

    def run(self, state: AgentState) -> dict[str, Any]:
        t0 = time.time()
        customer_id = state["customer_id"]
        run_id = state["run_id"]
        prediction = state.get("prediction", {})
        risk_tier = prediction.get("risk_tier", "HIGH")
        churn_prob = prediction.get("churn_probability", 0.0)
        explanation = state.get("explanation", {})
        top_factors = explanation.get("top_risk_factors", [])
        retention_plan = state.get("retention_plan", {})

        logger.info(
            "HITLAgent started",
            customer_id=customer_id,
            risk_tier=risk_tier,
            churn_prob=round(churn_prob, 3),
        )

        if risk_tier == "CRITICAL":
            decision = self._critical_gate(
                run_id=run_id,
                customer_id=customer_id,
                churn_prob=churn_prob,
                risk_tier=risk_tier,
                top_factors=top_factors,
                retention_plan=retention_plan,
                state=state,
            )
        elif risk_tier == "HIGH":
            decision = self._high_notify(
                run_id=run_id,
                customer_id=customer_id,
                churn_prob=churn_prob,
                risk_tier=risk_tier,
                top_factors=top_factors,
                retention_plan=retention_plan,
            )
        else:
            # MEDIUM — pass through silently
            decision = HITLDecision(
                status="auto_approved",
                decided_by="system",
                decided_at=_now(),
                notes="Risk tier below HITL threshold",
            )

        duration = round(time.time() - t0, 3)
        logger.info(
            "HITLAgent completed",
            customer_id=customer_id,
            status=decision["status"],
            decided_by=decision["decided_by"],
            duration_s=duration,
        )

        self._redis.append_stream_event(run_id, {
            "step": "hitl",
            "status": decision["status"],
            "decided_by": decision["decided_by"],
            "duration_s": duration,
        })

        return {
            "hitl_decision": decision,
            "current_step": "hitl",
            "completed_steps": state["completed_steps"] + ["hitl"],
            "errors": state["errors"],
            "step_durations": {**state.get("step_durations", {}), "hitl": duration},
        }

    # ------------------------------------------------------------------ #
    # CRITICAL tier — block and wait for human decision
    # ------------------------------------------------------------------ #

    def _critical_gate(
        self,
        run_id: str,
        customer_id: str,
        churn_prob: float,
        risk_tier: str,
        top_factors: list[dict[str, Any]],
        retention_plan: dict[str, Any],
        state: AgentState,
    ) -> HITLDecision:
        """Post HITL notification, wait for CSM decision, dispatch CRM on approval."""
        # Mark as pending in Redis so the webhook can write the decision
        self._write_decision(run_id, "pending", "system", "Awaiting CSM review")

        # Send Slack notification
        slack_sent = send_hitl_notification(
            run_id=run_id,
            customer_id=customer_id,
            churn_prob=churn_prob,
            risk_tier=risk_tier,
            top_factors=top_factors,
            retention_plan=retention_plan,
            webhook_base_url=self._webhook_base_url,
        )

        self._audit.log_hitl_decision(
            run_id=run_id,
            customer_id=customer_id,
            risk_tier=risk_tier,
            churn_prob=churn_prob,
            status="pending",
            decided_by="system",
            slack_sent=slack_sent,
        )

        # Dev mode: no Slack configured → auto-approve immediately
        if not slack_sent:
            logger.info(
                "No Slack webhook — auto-approving CRITICAL (dev mode)",
                run_id=run_id,
            )
            decision = HITLDecision(
                status="auto_approved",
                decided_by="system_no_slack",
                decided_at=_now(),
                notes="Auto-approved: no Slack webhook configured",
            )
            self._write_decision(run_id, "approved", decision["decided_by"], decision["notes"])
            self._dispatch_crm_post_approval(state, retention_plan)
            self._audit.log_hitl_decision(
                run_id=run_id,
                customer_id=customer_id,
                risk_tier=risk_tier,
                churn_prob=churn_prob,
                status="auto_approved",
                decided_by=decision["decided_by"],
                slack_sent=False,
            )
            return decision

        # Poll Redis for human decision
        logger.info(
            "Awaiting HITL decision",
            run_id=run_id,
            timeout_s=self._timeout,
        )
        deadline = time.time() + self._timeout

        while time.time() < deadline:
            stored = self._read_decision(run_id)
            if stored and stored.get("status") in ("approved", "rejected"):
                decision = HITLDecision(
                    status=stored["status"],
                    decided_by=stored.get("decided_by", "csm"),
                    decided_at=stored.get("decided_at", _now()),
                    notes=stored.get("notes", ""),
                )
                if decision["status"] == "approved":
                    self._dispatch_crm_post_approval(state, retention_plan)
                else:
                    logger.info(
                        "HITL rejected — CRM dispatch skipped",
                        run_id=run_id,
                        decided_by=decision["decided_by"],
                    )
                self._audit.log_hitl_decision(
                    run_id=run_id,
                    customer_id=customer_id,
                    risk_tier=risk_tier,
                    churn_prob=churn_prob,
                    status=decision["status"],
                    decided_by=decision["decided_by"],
                    slack_sent=True,
                )
                return decision
            time.sleep(_POLL_INTERVAL_S)

        # Timeout — auto-approve
        logger.warning(
            "HITL timeout — auto-approving",
            run_id=run_id,
            timeout_s=self._timeout,
        )
        decision = HITLDecision(
            status="auto_approved",
            decided_by="system_timeout",
            decided_at=_now(),
            notes=f"Auto-approved after {self._timeout}s HITL timeout",
        )
        self._write_decision(run_id, "approved", decision["decided_by"], decision["notes"])
        self._dispatch_crm_post_approval(state, retention_plan)
        self._audit.log_hitl_decision(
            run_id=run_id,
            customer_id=customer_id,
            risk_tier=risk_tier,
            churn_prob=churn_prob,
            status="auto_approved",
            decided_by=decision["decided_by"],
            slack_sent=True,
        )
        return decision

    # ------------------------------------------------------------------ #
    # HIGH tier — notify only, auto-approve
    # ------------------------------------------------------------------ #

    def _high_notify(
        self,
        run_id: str,
        customer_id: str,
        churn_prob: float,
        risk_tier: str,
        top_factors: list[dict[str, Any]],
        retention_plan: dict[str, Any],
    ) -> HITLDecision:
        """Send informational Slack alert (CRM was already dispatched)."""
        crm_action_id = retention_plan.get("crm_action_id", "")
        send_alert_notification(
            run_id=run_id,
            customer_id=customer_id,
            churn_prob=churn_prob,
            risk_tier=risk_tier,
            crm_action_id=crm_action_id,
            top_factors=top_factors,
        )
        return HITLDecision(
            status="auto_approved",
            decided_by="system",
            decided_at=_now(),
            notes=f"Auto-approved: {risk_tier} tier notification sent",
        )

    # ------------------------------------------------------------------ #
    # CRM dispatch (called post-approval for CRITICAL tier)
    # ------------------------------------------------------------------ #

    def _dispatch_crm_post_approval(
        self,
        state: AgentState,
        retention_plan: dict[str, Any],
    ) -> None:
        """Dispatch the top CRM action after HITL approval for CRITICAL tier."""
        from agents.tools.crm_executor_tool import crm_executor_tool
        from agents.retention_strategist import _action_type_from_label

        selected = retention_plan.get("selected_actions", [])
        if not selected:
            logger.info("No actions to dispatch post-HITL", run_id=state["run_id"])
            return

        best = selected[0]
        try:
            crm_result = crm_executor_tool.invoke({
                "customer_id": state["customer_id"],
                "action_type": _action_type_from_label(best.get("label", "")),
                "action_description": best.get("label", best.get("action", "Retention action")),
                "estimated_cost_usd": best.get("cost_usd", 0.0),
                "run_id": state["run_id"],
                "ab_group": retention_plan.get("ab_group", "treatment"),
            })
            action_id = crm_result.get("action_id", "")
            self._audit.log_crm_action(
                action_id=action_id,
                customer_id=state["customer_id"],
                action_type=crm_result.get("action_type", ""),
                status=crm_result.get("status", "scheduled"),
                ab_group=retention_plan.get("ab_group", "treatment"),
                cost_usd=best.get("cost_usd", 0.0),
            )
            logger.info(
                "CRM action dispatched post-HITL-approval",
                action_id=action_id,
                customer_id=state["customer_id"],
            )
        except Exception as exc:
            logger.error("CRM dispatch post-HITL failed", error=str(exc))

    # ------------------------------------------------------------------ #
    # Redis helpers
    # ------------------------------------------------------------------ #

    def _write_decision(
        self,
        run_id: str,
        status: str,
        decided_by: str,
        notes: str = "",
    ) -> None:
        key = f"churn:hitl:{run_id}:decision"
        payload = json.dumps({
            "status": status,
            "decided_by": decided_by,
            "decided_at": _now(),
            "notes": notes,
        })
        if self._redis._client:
            self._redis._client.setex(key, 86400, payload)
        else:
            self._redis._fallback[key] = payload

    def _read_decision(self, run_id: str) -> dict[str, Any] | None:
        key = f"churn:hitl:{run_id}:decision"
        if self._redis._client:
            raw = self._redis._client.get(key)
        else:
            raw = self._redis._fallback.get(key)
        return json.loads(raw) if raw else None


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
