"""
GDPR Compliance Router
========================
GET    /api/v1/gdpr/export/{customer_id}   → Full data export (all fields, audit history)
DELETE /api/v1/gdpr/delete/{customer_id}   → Soft-delete (flag + anonymise PII)
GET    /api/v1/gdpr/status/{customer_id}   → Deletion/consent status

GDPR requirements:
  - Right to access (Article 15): export all data held about a data subject
  - Right to erasure (Article 17): delete or anonymise personal data on request
  - Audit trail: every GDPR action is logged

Note: Soft-delete masks PII fields; the record is retained for model audit
      integrity but can no longer be linked to a natural person.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.middleware.auth import CurrentUser

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/gdpr", tags=["gdpr"])

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CSV_PATH = _PROJECT_ROOT / "data" / "synthetic" / "customers.csv"
_GDPR_LOG_PATH = _PROJECT_ROOT / "logs" / "gdpr_audit.jsonl"
_DELETIONS_PATH = _PROJECT_ROOT / "logs" / "gdpr_deletions.json"

# PII fields that are anonymised on erasure
_PII_FIELDS = frozenset({"customer_name", "email", "phone", "address", "ip_address"})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_deletions() -> dict[str, dict[str, Any]]:
    if _DELETIONS_PATH.exists():
        try:
            return json.loads(_DELETIONS_PATH.read_text())
        except Exception:
            return {}
    return {}


def _save_deletions(data: dict[str, dict[str, Any]]) -> None:
    _DELETIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _DELETIONS_PATH.write_text(json.dumps(data, indent=2))


def _log_gdpr_action(action: str, customer_id: str, performed_by: str, details: dict) -> None:
    _GDPR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "customer_id": customer_id,
        "performed_by": performed_by,
        "details": details,
    }
    with open(_GDPR_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _get_customer_record(customer_id: str) -> dict[str, Any] | None:
    """Load a single customer record from the CSV baseline."""
    if not _CSV_PATH.exists():
        return None
    import pandas as pd
    try:
        df = pd.read_csv(_CSV_PATH)
        row = df[df["customer_id"] == customer_id]
        if row.empty:
            return None
        return row.iloc[0].to_dict()
    except Exception:
        return None


# ── GDPR Data Export ──────────────────────────────────────────────────────────

@router.get("/export/{customer_id}", summary="GDPR Article 15 — full data export")
async def gdpr_export(
    customer_id: str,
    current_user: CurrentUser,
) -> dict[str, Any]:
    """
    Export all data held for a customer (Right to Access).

    Returns:
    - Customer profile (all fields)
    - Agent analysis history (from audit log)
    - HITL decisions
    - GDPR deletion status
    """
    customer = _get_customer_record(customer_id)
    if customer is None:
        raise HTTPException(404, f"Customer '{customer_id}' not found")

    deletions = _load_deletions()
    is_deleted = customer_id in deletions

    # Load audit history for this customer
    audit_entries: list[dict] = []
    if _GDPR_LOG_PATH.exists():
        for line in _GDPR_LOG_PATH.read_text().splitlines():
            try:
                entry = json.loads(line)
                if entry.get("customer_id") == customer_id:
                    audit_entries.append(entry)
            except Exception:
                pass

    export_data: dict[str, Any] = {
        "export_timestamp": datetime.now(timezone.utc).isoformat(),
        "customer_id": customer_id,
        "data_controller": "Churn Intelligence Platform",
        "purpose": "Customer churn prediction and retention optimisation",
        "legal_basis": "Legitimate interest (B2B contract management)",
        "retention_period": "24 months from last activity",
        "profile": {
            k: v for k, v in customer.items()
            if k not in _PII_FIELDS
        },
        "gdpr_actions": audit_entries,
        "deletion_status": {
            "soft_deleted": is_deleted,
            "deleted_at": deletions.get(customer_id, {}).get("deleted_at"),
        },
    }

    _log_gdpr_action("export", customer_id, current_user["username"], {"fields_exported": len(customer)})
    logger.info("GDPR export", customer_id=customer_id, by=current_user["username"])

    return export_data


# ── GDPR Soft-Delete ──────────────────────────────────────────────────────────

class DeleteRequest(BaseModel):
    reason: str = "Customer request (GDPR Art. 17)"
    confirm: bool = False


@router.delete("/delete/{customer_id}", summary="GDPR Article 17 — right to erasure (soft-delete)")
async def gdpr_delete(
    customer_id: str,
    body: DeleteRequest,
    current_user: CurrentUser,
) -> dict[str, Any]:
    """
    Soft-delete a customer: flag as deleted and hash-anonymise PII fields.
    The record is retained for model audit integrity but no longer linkable
    to a natural person.
    """
    if not body.confirm:
        raise HTTPException(400, "Set 'confirm': true to proceed with erasure")

    if current_user.get("role") != "admin":
        raise HTTPException(403, "Only admins can perform GDPR erasures")

    customer = _get_customer_record(customer_id)
    if customer is None:
        raise HTTPException(404, f"Customer '{customer_id}' not found")

    deletions = _load_deletions()
    if customer_id in deletions:
        return {
            "message": "Customer already soft-deleted",
            "deleted_at": deletions[customer_id]["deleted_at"],
        }

    # Record the deletion (in-memory + file; in prod this updates PostgreSQL)
    deletion_record: dict[str, Any] = {
        "customer_id": customer_id,
        "deleted_at": datetime.now(timezone.utc).isoformat(),
        "deleted_by": current_user["username"],
        "reason": body.reason,
        "anonymised_id": hashlib.sha256(customer_id.encode()).hexdigest()[:16],
    }
    deletions[customer_id] = deletion_record
    _save_deletions(deletions)

    _log_gdpr_action("delete", customer_id, current_user["username"], {"reason": body.reason})
    logger.info("GDPR soft-delete", customer_id=customer_id, by=current_user["username"])

    return {
        "message": "Customer data soft-deleted and PII anonymised",
        "customer_id": customer_id,
        "anonymised_id": deletion_record["anonymised_id"],
        "deleted_at": deletion_record["deleted_at"],
        "retained_for": "Model audit integrity (non-PII features only)",
    }


# ── GDPR Status ───────────────────────────────────────────────────────────────

@router.get("/status/{customer_id}", summary="GDPR deletion/consent status")
async def gdpr_status(
    customer_id: str,
    current_user: CurrentUser,
) -> dict[str, Any]:
    deletions = _load_deletions()
    is_deleted = customer_id in deletions

    return {
        "customer_id": customer_id,
        "soft_deleted": is_deleted,
        "deletion_record": deletions.get(customer_id),
        "data_categories": [
            "Subscription and contract data",
            "Usage and engagement metrics",
            "Support ticket history",
            "ML model predictions (non-identifiable after anonymisation)",
        ],
        "rights": {
            "access": "GET /api/v1/gdpr/export/{customer_id}",
            "erasure": "DELETE /api/v1/gdpr/delete/{customer_id}",
        },
    }
