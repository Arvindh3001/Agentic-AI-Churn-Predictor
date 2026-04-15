"""
CRM Executor Tool
==================
LangChain-compatible tool that records and "executes" a retention action
in the CRM system. Phase 4 uses a mock implementation that logs to Redis
(with in-memory fallback). A real CRM integration (Salesforce / HubSpot)
is wired in Phase 8.

Action types supported:
    price_reduction      — apply a discount to the customer account
    csm_assignment       — assign a customer success manager
    loyalty_upgrade      — enroll in a loyalty / upgrade programme
    support_outreach     — create a proactive support case
    engagement_campaign  — trigger a re-engagement email sequence

Each execution returns a structured action record with a unique action_id,
timestamp, and status, stored for audit and A/B outcome tracking.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

import structlog
from langchain_core.tools import tool
from pydantic import BaseModel

logger = structlog.get_logger(__name__)

# In-memory action log: list of action records
_action_log: list[dict[str, Any]] = []

VALID_ACTION_TYPES = {
    "price_reduction",
    "csm_assignment",
    "loyalty_upgrade",
    "support_outreach",
    "engagement_campaign",
}


class CRMExecutorInput(BaseModel):
    customer_id: str
    action_type: str
    action_description: str
    estimated_cost_usd: float = 0.0
    run_id: str = ""
    ab_group: str = "treatment"


@tool("crm_executor_tool", args_schema=CRMExecutorInput)
def crm_executor_tool(
    customer_id: str,
    action_type: str,
    action_description: str,
    estimated_cost_usd: float = 0.0,
    run_id: str = "",
    ab_group: str = "treatment",
) -> dict[str, Any]:
    """
    Execute (or schedule) a retention action in the CRM system.

    Args:
        customer_id:          Customer to act on.
        action_type:          One of the supported action type strings.
        action_description:   Human-readable description of the action.
        estimated_cost_usd:   Expected spend for this action.
        run_id:               Pipeline run ID for audit linking.
        ab_group:             "treatment" or "control" — control group
                              actions are logged but not dispatched.

    Returns:
        Dict with action_id, status, timestamp, customer_id, ab_group.
    """
    action_id = f"crm-{str(uuid.uuid4())[:8]}"
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Normalise action type
    normalized_type = action_type.lower().replace(" ", "_").replace("-", "_")
    if normalized_type not in VALID_ACTION_TYPES:
        logger.warning(
            "Unknown action type — defaulting to engagement_campaign",
            action_type=action_type,
        )
        normalized_type = "engagement_campaign"

    # Control group: log but do not dispatch
    status = "scheduled" if ab_group == "treatment" else "control_holdout"

    record = {
        "action_id": action_id,
        "customer_id": customer_id,
        "action_type": normalized_type,
        "action_description": action_description,
        "estimated_cost_usd": round(estimated_cost_usd, 2),
        "status": status,
        "ab_group": ab_group,
        "run_id": run_id,
        "timestamp": timestamp,
        "dispatched": ab_group == "treatment",
    }

    # Persist to Redis if available, else in-memory
    _persist_action(record)

    logger.info(
        "CRM action recorded",
        action_id=action_id,
        customer_id=customer_id,
        action_type=normalized_type,
        ab_group=ab_group,
        status=status,
    )

    return record


def get_action_history(customer_id: str) -> list[dict[str, Any]]:
    """Return all logged CRM actions for a customer (in-memory)."""
    return [r for r in _action_log if r["customer_id"] == customer_id]


def get_all_actions() -> list[dict[str, Any]]:
    """Return the full in-memory CRM action log."""
    return list(_action_log)


def _persist_action(record: dict[str, Any]) -> None:
    """Append action record to Redis list and in-memory log."""
    _action_log.append(record)

    try:
        from memory.redis_state import RedisStateManager
        mgr = RedisStateManager()
        redis_client = getattr(mgr, "_redis", None) or getattr(mgr, "_client", None)
        if redis_client:
            key = f"churn:crm:{record['customer_id']}:actions"
            redis_client.rpush(key, json.dumps(record))
            redis_client.expire(key, 86400 * 90)  # 90-day retention
    except Exception as exc:
        logger.debug("Redis CRM persist skipped", error=str(exc))
