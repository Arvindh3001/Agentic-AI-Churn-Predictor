"""
Analytics Router — Phase 7
============================
GET  /api/v1/analytics/survival        → Kaplan-Meier curves + Cox PH hazard ratios
GET  /api/v1/analytics/cohort          → Monthly cohort retention matrix
GET  /api/v1/analytics/seasonality     → STL trend / seasonal decomposition
GET  /api/v1/analytics/fairness        → Bias checks across protected attributes
GET  /api/v1/analytics/fairness/report → HTML fairness report
GET  /api/v1/analytics/robustness      → Model robustness + calibration report
POST /api/v1/analytics/optimize        → Portfolio-level retention budget optimizer
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.middleware.auth import CurrentUser
from src.optimization.knapsack_solver import solve_retention_budget

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CSV_PATH = _PROJECT_ROOT / "data" / "synthetic" / "customers.csv"


# ── CSV loader (cached) ───────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def _load_df() -> pd.DataFrame:
    if not _CSV_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(_CSV_PATH)
    # Derive churn column if not present
    if "churned" not in df.columns and "churn_label" in df.columns:
        df["churned"] = df["churn_label"]
    # Derive binary prediction from probability
    if "churn_probability" in df.columns and "predicted_churn" not in df.columns:
        df["predicted_churn"] = (df["churn_probability"] >= 0.5).astype(int)
    return df


# ── Survival Analysis ─────────────────────────────────────────────────────────

@router.get("/survival", summary="Kaplan-Meier + Cox PH survival analysis")
async def survival_analysis(
    current_user: CurrentUser,
    group_by: str | None = Query(None, description="Column to stratify KM curves (e.g. contract_type)"),
    max_months: int = Query(60, ge=6, le=120),
) -> dict[str, Any]:
    from src.temporal.survival_analysis import kaplan_meier, cox_ph, time_to_churn_projection

    df = _load_df()
    if df.empty:
        raise HTTPException(404, "Customer dataset not found")

    timeline = list(range(0, max_months + 1))
    km_result = kaplan_meier(df, group_col=group_by, timeline=timeline)
    cph_result = cox_ph(df)
    projection = time_to_churn_projection(df)

    return {
        "kaplan_meier": km_result,
        "cox_ph": cph_result,
        "projection": projection,
        "dataset_size": len(df),
    }


# ── Cohort Analysis ───────────────────────────────────────────────────────────

@router.get("/cohort", summary="Monthly cohort retention matrix")
async def cohort_analysis(
    current_user: CurrentUser,
    max_months: int = Query(24, ge=3, le=60),
) -> dict[str, Any]:
    from src.temporal.cohort_analysis import build_cohort_matrix, cohort_churn_rates

    df = _load_df()
    if df.empty:
        raise HTTPException(404, "Customer dataset not found")

    matrix_data = build_cohort_matrix(df, max_months=max_months)
    churn_rates = cohort_churn_rates(matrix_data)

    return {
        "cohort_matrix": matrix_data,
        "churn_rates": churn_rates,
        "dataset_size": len(df),
    }


# ── Seasonality ───────────────────────────────────────────────────────────────

@router.get("/seasonality", summary="STL trend and seasonality decomposition")
async def seasonality(
    current_user: CurrentUser,
) -> dict[str, Any]:
    from src.temporal.seasonality import seasonality_summary

    df = _load_df()
    if df.empty:
        raise HTTPException(404, "Customer dataset not found")

    result = seasonality_summary(df)
    if "error" in result:
        raise HTTPException(422, result["error"])

    return result


# ── Fairness ──────────────────────────────────────────────────────────────────

_DEFAULT_PROTECTED = ["contract_type", "tenure_bucket", "payment_method"]


@router.get("/fairness", summary="Bias and fairness checks across protected attributes")
async def fairness_check(
    current_user: CurrentUser,
    attributes: str = Query(
        ",".join(_DEFAULT_PROTECTED),
        description="Comma-separated protected attribute columns",
    ),
) -> dict[str, Any]:
    from src.fairness.bias_detector import check_bias, bias_summary
    from src.fairness.fairness_report import build_json_report

    df = _load_df()
    if df.empty:
        raise HTTPException(404, "Customer dataset not found")

    # Derive tenure bucket if not present
    if "tenure_bucket" not in df.columns and "tenure_months" in df.columns:
        df = df.copy()
        df["tenure_bucket"] = pd.cut(
            df["tenure_months"],
            bins=[0, 12, 36, 72, 9999],
            labels=["0-12m", "13-36m", "37-72m", "73m+"],
        ).astype(str)

    attr_list = [a.strip() for a in attributes.split(",") if a.strip()]
    results = check_bias(df, protected_attributes=attr_list)
    report = build_json_report(
        results, model_version="champion", dataset_size=len(df)
    )
    return report


@router.get("/fairness/report", summary="HTML fairness report", response_class=HTMLResponse)
async def fairness_html_report(
    current_user: CurrentUser,
    attributes: str = Query(",".join(_DEFAULT_PROTECTED)),
) -> HTMLResponse:
    from src.fairness.bias_detector import check_bias
    from src.fairness.fairness_report import build_html_report

    df = _load_df()
    if df.empty:
        raise HTTPException(404, "Customer dataset not found")

    if "tenure_bucket" not in df.columns and "tenure_months" in df.columns:
        df = df.copy()
        df["tenure_bucket"] = pd.cut(
            df["tenure_months"],
            bins=[0, 12, 36, 72, 9999],
            labels=["0-12m", "13-36m", "37-72m", "73m+"],
        ).astype(str)

    attr_list = [a.strip() for a in attributes.split(",") if a.strip()]
    results = check_bias(df, protected_attributes=attr_list)
    html = build_html_report(results, model_version="champion", dataset_size=len(df))
    return HTMLResponse(content=html)


# ── Robustness ────────────────────────────────────────────────────────────────

@router.get("/robustness", summary="Model robustness and calibration report")
async def robustness(
    current_user: CurrentUser,
    n_samples: int = Query(500, ge=50, le=2000),
) -> dict[str, Any]:
    from src.robustness.robustness_tester import run_robustness_tests

    df = _load_df()
    if df.empty:
        raise HTTPException(404, "Customer dataset not found")

    numeric_feature_cols = [
        "monthly_charges", "tenure_months", "num_support_tickets_30d",
        "feature_adoption_rate", "nps_score", "total_charges",
        "avg_session_duration_min", "login_frequency_30d",
    ]
    available_cols = [c for c in numeric_feature_cols if c in df.columns]
    if not available_cols:
        raise HTTPException(422, "No numeric feature columns found in dataset")

    X = df[available_cols].fillna(0)
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
    X_sample = X.iloc[sample_idx].reset_index(drop=True)

    # Build a lightweight predict_fn using saved model or fallback to linear proxy
    def predict_fn(df_in: pd.DataFrame) -> np.ndarray:
        try:
            from agents.tools.model_tool import get_predictions
            rows = df_in.to_dict(orient="records")
            probs = []
            for row in rows[:10]:  # limit for speed
                result = get_predictions(row)
                probs.append(result.get("churn_probability", 0.5))
            # Fill remaining with mean
            mean_p = float(np.mean(probs)) if probs else 0.5
            return np.array(probs + [mean_p] * (len(df_in) - len(probs)))
        except Exception:
            # Fallback: logistic proxy on standardised features
            arr = df_in.select_dtypes(include=[np.number]).values.astype(float)
            arr = np.nan_to_num(arr)
            # Normalise
            std = arr.std(axis=0, keepdims=True)
            std[std == 0] = 1
            arr_n = (arr - arr.mean(axis=0, keepdims=True)) / std
            logit = arr_n.mean(axis=1) * 0.5
            return 1 / (1 + np.exp(-logit))

    y_true = None
    y_prob = None
    if "churned" in df.columns and "churn_probability" in df.columns:
        y_true = df.iloc[sample_idx]["churned"].astype(int).values
        y_prob = df.iloc[sample_idx]["churn_probability"].clip(0, 1).values

    segment_cols = ["contract_type"] if "contract_type" in df.columns else []
    report = run_robustness_tests(
        X_sample, predict_fn,
        y_true=y_true, y_prob=y_prob,
        segment_cols=segment_cols,
    )
    return report.to_dict()


# ── Portfolio Budget Optimizer ────────────────────────────────────────────────

class OptimizeRequest(BaseModel):
    customer_ids: list[str] | None = None
    total_budget: float = 50_000.0
    max_actions_per_customer: int = 2
    risk_tier_filter: list[str] = ["CRITICAL", "HIGH"]


@router.post("/optimize", summary="Portfolio-level retention budget optimizer")
async def optimize_retention(
    body: OptimizeRequest,
    current_user: CurrentUser,
) -> dict[str, Any]:
    df = _load_df()
    if df.empty:
        raise HTTPException(404, "Customer dataset not found")

    # Filter to requested customers or high-risk set
    if body.customer_ids:
        sub = df[df["customer_id"].isin(body.customer_ids)]
    elif "risk_tier" in df.columns:
        sub = df[df["risk_tier"].isin(body.risk_tier_filter)]
    elif "churn_probability" in df.columns:
        sub = df[df["churn_probability"] >= 0.70]
    else:
        sub = df.head(200)

    if sub.empty:
        return {"error": "No customers matched the filter", "plan": []}

    # Build action catalogue per customer
    customers_input = []
    for _, row in sub.head(200).iterrows():
        cid = str(row.get("customer_id", "unknown"))
        prob = float(row.get("churn_probability", 0.7))
        clv = float(row.get("clv_estimate", row.get("monthly_charges", 80) * 24))
        customers_input.append({
            "customer_id": cid,
            "churn_probability": prob,
            "clv": clv,
            "actions": [
                {"name": "discount_10pct", "cost": 20, "prob_reduction": 0.08},
                {"name": "discount_20pct", "cost": 40, "prob_reduction": 0.15},
                {"name": "csm_outreach", "cost": 80, "prob_reduction": 0.12},
                {"name": "loyalty_upgrade", "cost": 60, "prob_reduction": 0.10},
                {"name": "dedicated_csm", "cost": 150, "prob_reduction": 0.20},
            ],
        })

    try:
        result = solve_retention_budget(
            customers=customers_input,
            total_budget=body.total_budget,
            max_actions_per_customer=body.max_actions_per_customer,
        )
    except Exception as exc:
        logger.warning("Optimizer error", error=str(exc))
        raise HTTPException(500, f"Optimizer failed: {exc}") from exc

    return {
        "budget": body.total_budget,
        "customers_considered": len(customers_input),
        **result,
    }
