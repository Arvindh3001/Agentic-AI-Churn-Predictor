"""
Customers Router
=================
GET   /api/v1/customers              — paginated customer list from CSV
GET   /api/v1/customers/{id}         — single customer + last Redis analysis
GET   /api/v1/customers/high-risk    — pre-filtered high-risk watchlist
PATCH /api/v1/customers/{id}         — update editable fields (in-memory overlay)

Edits are stored in an in-memory dict and merged on top of the CSV baseline.
They persist until the server restarts.  In production, back this with Postgres.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.middleware.auth import CurrentUser

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["customers"])

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CSV_PATH = _PROJECT_ROOT / "data" / "synthetic" / "customers.csv"

# Columns to expose — never expose raw target label in a real API
_SAFE_COLS = [
    "customer_id",
    "monthly_charges",
    "tenure_months",
    "contract_type",
    "num_support_tickets_30d",
    "feature_adoption_rate",
    "nps_score",
    "login_frequency_30d",
    "churn",
]

# In-memory edits overlay — survives request boundaries, lost on server restart
_CUSTOMER_OVERRIDES: dict[str, dict[str, Any]] = {}

# Which fields the UI is allowed to change
_EDITABLE_COLS = frozenset({
    "contract_type",
    "monthly_charges",
    "num_support_tickets_30d",
    "feature_adoption_rate",
    "nps_score",
})


# ------------------------------------------------------------------ #
# Pydantic schema for PATCH body
# ------------------------------------------------------------------ #

class CustomerUpdate(BaseModel):
    contract_type: str | None = None
    monthly_charges: float | None = None
    num_support_tickets_30d: int | None = None
    feature_adoption_rate: float | None = None
    nps_score: float | None = None


# ------------------------------------------------------------------ #
# Internal helpers
# ------------------------------------------------------------------ #

def _load_csv() -> "pd.DataFrame":
    import pandas as pd
    if not _CSV_PATH.exists():
        raise HTTPException(status_code=503, detail="Customer data not available")
    return pd.read_csv(_CSV_PATH)


def _safe_row(row: "pd.Series") -> dict[str, Any]:
    """Convert a CSV row to a safe API-friendly dict."""
    result: dict[str, Any] = {}
    for col in _SAFE_COLS:
        if col in row.index:
            val = row[col]
            # Convert numpy types to Python natives
            if hasattr(val, "item"):
                val = val.item()
            result[col] = val
    return result


def _apply_overrides(record: dict[str, Any]) -> dict[str, Any]:
    """Merge any in-memory edits on top of the base CSV record."""
    cid = record.get("customer_id")
    if cid and cid in _CUSTOMER_OVERRIDES:
        return {**record, **_CUSTOMER_OVERRIDES[cid]}
    return record


# ------------------------------------------------------------------ #
# Endpoints
# ------------------------------------------------------------------ #

@router.get("/customers")
async def list_customers(
    current_user: CurrentUser,
    status_filter: str | None = Query(
        None,
        description="Filter customers: 'churned' | 'active'",
    ),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    """
    Return a paginated list of customers.

    Use status_filter=churned to get the at-risk watchlist.
    In-memory edits are merged on top of the CSV baseline.
    """
    df = _load_csv()
    churn_col = "churned" if "churned" in df.columns else "churn"

    if status_filter == "churned":
        df = df[df[churn_col] == 1]
    elif status_filter == "active":
        df = df[df[churn_col] == 0]

    total = len(df)
    page = df.iloc[offset: offset + limit]

    customers = [_apply_overrides(_safe_row(row)) for _, row in page.iterrows()]

    logger.debug(
        "Customers listed",
        total=total,
        returned=len(customers),
        filter=status_filter,
        user=current_user["username"],
    )

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "customers": customers,
    }


@router.get("/customers/high-risk")
async def high_risk_watchlist(
    current_user: CurrentUser,
    limit: int = Query(20, ge=1, le=100),
) -> dict[str, Any]:
    """
    Return the top churned customers as a quick watchlist.
    Sorted by monthly_charges descending (highest CLV risk first).
    In-memory edits are merged on top of the CSV baseline.
    """
    df = _load_csv()
    churn_col = "churned" if "churned" in df.columns else "churn"
    at_risk = df[df[churn_col] == 1].sort_values(
        "monthly_charges", ascending=False
    ).head(limit)

    return {
        "total": len(at_risk),
        "customers": [_apply_overrides(_safe_row(row)) for _, row in at_risk.iterrows()],
    }


@router.get("/customers/{customer_id}")
async def get_customer(
    customer_id: str,
    current_user: CurrentUser,
) -> dict[str, Any]:
    """
    Return a single customer's profile and their last pipeline analysis
    result (if available in Redis).  In-memory edits are merged in.
    """
    df = _load_csv()
    matches = df[df["customer_id"] == customer_id]
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"Customer '{customer_id}' not found")

    record = _apply_overrides(_safe_row(matches.iloc[0]))

    # Attach last analysis from Redis if present
    last_analysis: dict[str, Any] | None = None
    try:
        from memory.redis_state import RedisStateManager
        last_analysis = RedisStateManager().get_customer_context(customer_id)
    except Exception:
        pass

    return {
        "customer": record,
        "last_analysis": last_analysis,
    }


@router.patch("/customers/{customer_id}")
async def update_customer(
    customer_id: str,
    updates: CustomerUpdate,
    current_user: CurrentUser,
) -> dict[str, Any]:
    """
    Update editable fields for a customer.

    Changes are stored in an in-memory overlay on top of the CSV baseline.
    After saving, the update is broadcast to all connected /ws/customers clients
    so the dashboard refreshes in real-time without a page reload.
    """
    df = _load_csv()
    matches = df[df["customer_id"] == customer_id]
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"Customer '{customer_id}' not found")

    # Build a fully merged record: CSV base → existing overrides → new changes
    record = _apply_overrides(_safe_row(matches.iloc[0]))

    patch_data = {k: v for k, v in updates.model_dump().items() if v is not None}
    for key, val in patch_data.items():
        if key in _EDITABLE_COLS:
            record[key] = val

    # Persist only the editable subset in the overlay
    _CUSTOMER_OVERRIDES[customer_id] = {
        k: record[k] for k in _EDITABLE_COLS if k in record
    }

    logger.info(
        "Customer updated",
        customer_id=customer_id,
        changes=list(patch_data.keys()),
        user=current_user["username"],
    )

    # Broadcast real-time update to all connected dashboard clients
    try:
        from app.websockets.customer_broadcast import broadcaster
        await broadcaster.broadcast({"type": "customer_updated", "customer": record})
    except Exception as exc:
        logger.warning("WS broadcast failed", error=str(exc))

    return {"customer": record, "updated_by": current_user["username"]}
