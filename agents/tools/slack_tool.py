"""
Slack Tool
===========
Sends Slack Block Kit notifications for HITL review requests and
informational churn alerts.  Falls back to structured console logging
when no webhook URL is configured (dev / CI mode).

Two notification types:
    1. HITL review request (CRITICAL tier) — includes Approve / Reject
       buttons whose callbacks hit the POST /hitl/decision endpoint.
    2. Alert notification (HIGH tier) — informational only, no buttons.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from config.settings import settings

logger = structlog.get_logger(__name__)


def send_hitl_notification(
    run_id: str,
    customer_id: str,
    churn_prob: float,
    risk_tier: str,
    top_factors: list[dict[str, Any]],
    retention_plan: dict[str, Any],
    webhook_base_url: str = "http://localhost:8000",
) -> bool:
    """
    Send a HITL review request to Slack with Approve / Reject buttons.

    The button values carry the run_id so the /hitl/decision webhook can
    write the CSM decision to Redis and unblock the waiting HITLAgent.

    Returns:
        True if the Slack webhook call succeeded, False otherwise.
        False is also returned when no webhook URL is configured (dev mode).
    """
    if not settings.slack.slack_webhook_url:
        _console_notify(
            "HITL_REVIEW_REQUIRED",
            run_id=run_id,
            customer_id=customer_id,
            churn_prob=f"{churn_prob:.0%}",
            risk_tier=risk_tier,
        )
        return False

    blocks = _build_hitl_blocks(
        run_id=run_id,
        customer_id=customer_id,
        churn_prob=churn_prob,
        risk_tier=risk_tier,
        top_factors=top_factors,
        retention_plan=retention_plan,
        webhook_base_url=webhook_base_url,
    )

    return _post_to_slack({
        "blocks": blocks,
        "text": f"HITL Review Required: {customer_id[:8]} — {risk_tier} Risk ({churn_prob:.0%})",
    })


def send_alert_notification(
    run_id: str,
    customer_id: str,
    churn_prob: float,
    risk_tier: str,
    crm_action_id: str,
    top_factors: list[dict[str, Any]],
) -> bool:
    """
    Send an informational Slack alert for HIGH-tier customers.
    No buttons — CRM action was dispatched automatically.

    Returns:
        True if the Slack webhook call succeeded, False otherwise.
    """
    if not settings.slack.slack_webhook_url:
        _console_notify(
            "CHURN_ALERT",
            run_id=run_id,
            customer_id=customer_id,
            churn_prob=f"{churn_prob:.0%}",
            risk_tier=risk_tier,
            crm_action_id=crm_action_id,
        )
        return False

    blocks = _build_alert_blocks(
        customer_id=customer_id,
        churn_prob=churn_prob,
        risk_tier=risk_tier,
        crm_action_id=crm_action_id,
        top_factors=top_factors,
    )

    return _post_to_slack({
        "blocks": blocks,
        "text": f"Churn Alert: {customer_id[:8]} — {risk_tier} Risk ({churn_prob:.0%})",
    })


# ------------------------------------------------------------------ #
# Block Kit builders
# ------------------------------------------------------------------ #

_TIER_EMOJI = {
    "CRITICAL": ":rotating_light:",
    "HIGH": ":warning:",
    "MEDIUM": ":information_source:",
}


def _build_hitl_blocks(
    run_id: str,
    customer_id: str,
    churn_prob: float,
    risk_tier: str,
    top_factors: list[dict[str, Any]],
    retention_plan: dict[str, Any],
    webhook_base_url: str,
) -> list[dict[str, Any]]:
    emoji = _TIER_EMOJI.get(risk_tier, ":white_circle:")
    prob_pct = f"{churn_prob:.0%}"

    factor_lines = "\n".join(
        f"  • {f.get('label', f.get('feature', 'Unknown'))}: {f.get('shap_value', 0):+.3f}"
        for f in top_factors[:3]
    ) or "  • No factors available"

    selected = retention_plan.get("selected_actions", [])
    action_lines = "\n".join(
        f"  • {a.get('label', a.get('action', 'Action'))[:60]} (${a.get('cost_usd', 0):.0f})"
        for a in selected[:3]
    ) or "  • No actions selected"

    total_cost = retention_plan.get("total_cost_usd", 0)
    roi = retention_plan.get("estimated_roi", 0)
    ab_group = retention_plan.get("ab_group", "treatment")
    decision_url = f"{webhook_base_url}/hitl/decision"

    return [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{emoji} HITL Review Required — {risk_tier} Churn Risk",
            },
        },
        {"type": "divider"},
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Customer ID*\n`{customer_id}`"},
                {"type": "mrkdwn", "text": f"*Churn Probability*\n*{prob_pct}*"},
                {"type": "mrkdwn", "text": f"*Risk Tier*\n{risk_tier}"},
                {"type": "mrkdwn", "text": f"*A/B Group*\n{ab_group}"},
                {"type": "mrkdwn", "text": f"*Est. Cost*\n${total_cost:.2f}"},
                {"type": "mrkdwn", "text": f"*Est. ROI*\n{roi:+.1%}"},
            ],
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Top Risk Factors*\n{factor_lines}"},
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Recommended Actions*\n{action_lines}"},
        },
        {"type": "divider"},
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Approve"},
                    "style": "primary",
                    "action_id": "hitl_approve",
                    "value": json.dumps({"run_id": run_id, "decision": "approved"}),
                    "url": decision_url,
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Reject"},
                    "style": "danger",
                    "action_id": "hitl_reject",
                    "value": json.dumps({"run_id": run_id, "decision": "rejected"}),
                    "url": decision_url,
                },
            ],
        },
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": (
                        f"Run ID: `{run_id}` | "
                        f"Auto-approves after {settings.model.hitl_timeout_seconds // 60} min"
                    ),
                }
            ],
        },
    ]


def _build_alert_blocks(
    customer_id: str,
    churn_prob: float,
    risk_tier: str,
    crm_action_id: str,
    top_factors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    emoji = _TIER_EMOJI.get(risk_tier, ":warning:")
    prob_pct = f"{churn_prob:.0%}"

    factor_lines = "\n".join(
        f"  • {f.get('label', f.get('feature', 'Unknown'))}: {f.get('shap_value', 0):+.3f}"
        for f in top_factors[:3]
    ) or "  • No factors available"

    return [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{emoji} Churn Alert — {risk_tier} Risk Customer",
            },
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Customer*\n`{customer_id}`"},
                {"type": "mrkdwn", "text": f"*Churn Prob*\n*{prob_pct}*"},
                {"type": "mrkdwn", "text": f"*Risk Tier*\n{risk_tier}"},
                {"type": "mrkdwn", "text": f"*CRM Action*\n`{crm_action_id}`"},
            ],
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Top Risk Factors*\n{factor_lines}"},
        },
        {
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": "Retention actions dispatched automatically."},
            ],
        },
    ]


# ------------------------------------------------------------------ #
# HTTP transport (stdlib only — no extra dependency)
# ------------------------------------------------------------------ #

def _post_to_slack(payload: dict[str, Any]) -> bool:
    """POST a JSON payload to the configured Slack incoming webhook URL."""
    import urllib.error
    import urllib.request

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            settings.slack.slack_webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                logger.info("Slack notification sent")
                return True
            logger.warning("Slack returned non-200", status=resp.status)
            return False
    except Exception as exc:
        logger.warning("Slack notification failed", error=str(exc))
        return False


def _console_notify(event_type: str, **kwargs: Any) -> None:
    """Structured console fallback when no Slack webhook is configured."""
    logger.info(f"[SLACK-FALLBACK] {event_type}", **kwargs)
