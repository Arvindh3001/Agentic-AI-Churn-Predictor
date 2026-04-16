"""
Cohort retention and churn analysis.

Builds monthly cohort heatmaps showing what fraction of customers from each
acquisition cohort are still active (or have churned) in subsequent months.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ── Cohort matrix builder ────────────────────────────────────────────────────

def build_cohort_matrix(
    df: pd.DataFrame,
    signup_col: str = "signup_date",
    tenure_col: str = "tenure_months",
    churn_col: str = "churned",
    max_months: int = 24,
) -> dict[str, Any]:
    """
    Build a cohort retention matrix.

    Args:
        df: Customer dataframe.
        signup_col: Column with signup/acquisition date.
        tenure_col: Column with observed tenure in months.
        churn_col: Column indicating churn (1) or active (0).
        max_months: Maximum number of months tracked per cohort.

    Returns:
        Dict with ``matrix`` (cohort × month retention rates), ``cohorts``
        (list of cohort labels), and ``months`` (0..max_months-1).
    """
    df = df.copy()

    # Build synthetic cohort groups if no signup date
    if signup_col not in df.columns or df[signup_col].isna().all():
        return _synthetic_cohort_matrix(df, tenure_col, churn_col, max_months)

    df[signup_col] = pd.to_datetime(df[signup_col], errors="coerce")
    df = df.dropna(subset=[signup_col])
    df["cohort"] = df[signup_col].dt.to_period("M")

    cohort_groups = sorted(df["cohort"].unique())
    matrix_rows: list[list[float | None]] = []
    cohort_labels: list[str] = []

    for cohort in cohort_groups:
        sub = df[df["cohort"] == cohort]
        cohort_size = len(sub)
        if cohort_size == 0:
            continue

        row: list[float | None] = []
        for month in range(max_months):
            # Customers still active at `month` = those with tenure > month
            # (or churned after month)
            still_active = sub[
                (sub[tenure_col] > month) | (sub[churn_col] == 0)
            ].shape[0]
            # More precisely: customers observed for at least `month` months
            observed = sub[sub[tenure_col] >= month].shape[0]
            if observed == 0:
                row.append(None)
            else:
                churned_by_month = sub[
                    (sub[churn_col] == 1) & (sub[tenure_col] <= month)
                ].shape[0]
                retention = 1.0 - churned_by_month / cohort_size
                row.append(round(retention, 4))

        matrix_rows.append(row)
        cohort_labels.append(str(cohort))

    return {
        "matrix": matrix_rows,
        "cohorts": cohort_labels,
        "months": list(range(max_months)),
        "method": "actual",
    }


def _synthetic_cohort_matrix(
    df: pd.DataFrame,
    tenure_col: str,
    churn_col: str,
    max_months: int,
) -> dict[str, Any]:
    """Fallback: bucket customers into tenure-based pseudo-cohorts."""
    if tenure_col not in df.columns:
        return {"error": f"Column '{tenure_col}' not found", "matrix": [], "cohorts": [], "months": []}

    df = df.copy()
    bucket_size = 6
    df["cohort_bucket"] = (df[tenure_col] // bucket_size) * bucket_size
    cohort_labels: list[str] = []
    matrix_rows: list[list[float | None]] = []

    for bucket, sub in sorted(df.groupby("cohort_bucket")):
        cohort_size = len(sub)
        label = f"Cohort T+{int(bucket)}"
        cohort_labels.append(label)
        row: list[float | None] = []
        for month in range(max_months):
            observed = sub[sub[tenure_col] >= (int(bucket) + month)].shape[0]
            if observed == 0:
                row.append(None)
            else:
                churned = sub[
                    (sub[churn_col] == 1) & (sub[tenure_col] <= (int(bucket) + month))
                ].shape[0]
                retention = 1.0 - churned / cohort_size
                row.append(round(retention, 4))
        matrix_rows.append(row)

    return {
        "matrix": matrix_rows,
        "cohorts": cohort_labels,
        "months": list(range(max_months)),
        "method": "tenure_buckets",
    }


# ── Churn rate by cohort month ────────────────────────────────────────────────

def cohort_churn_rates(
    cohort_matrix: dict[str, Any],
) -> dict[str, Any]:
    """
    Derive period-over-period churn rates from a retention matrix.

    Returns the average churn rate per month-of-life across all cohorts.
    """
    matrix = cohort_matrix.get("matrix", [])
    months = cohort_matrix.get("months", [])
    if not matrix:
        return {"avg_churn_by_month": [], "months": months}

    arr = np.array([[v if v is not None else np.nan for v in row] for row in matrix])
    # Period churn = retention[t-1] - retention[t]
    period_churn = np.diff(arr, axis=1) * -1  # positive = churned this period
    avg_churn = np.nanmean(period_churn, axis=0).tolist()

    return {
        "avg_churn_by_month": [round(v, 4) for v in avg_churn],
        "months": months[1:],
        "cohort_count": len(matrix),
    }
