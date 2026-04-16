"""
Temporal survival analysis: Kaplan-Meier curves + Cox Proportional Hazards.

Uses the `lifelines` library. Outputs serialisable dicts so results can be
returned directly from FastAPI endpoints.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Optional lifelines import ────────────────────────────────────────────────
try:
    from lifelines import KaplanMeierFitter, CoxPHFitter  # type: ignore
    _LIFELINES_OK = True
except ImportError:
    _LIFELINES_OK = False
    logger.warning("lifelines not installed — survival analysis will use scipy fallback")


# ── Kaplan-Meier ─────────────────────────────────────────────────────────────

def kaplan_meier(
    df: pd.DataFrame,
    duration_col: str = "tenure_months",
    event_col: str = "churned",
    group_col: str | None = None,
    timeline: list[float] | None = None,
) -> dict[str, Any]:
    """
    Fit Kaplan-Meier survival curves.

    Args:
        df: Customer dataframe.
        duration_col: Column with observed time (months).
        event_col: Boolean/int column; 1 = churned, 0 = censored.
        group_col: Optional column for stratified curves.
        timeline: Evaluation time-points (default: 0–max with step 1).

    Returns:
        Dict with keys ``curves`` (list of {label, timeline, survival}) and
        ``median_survival`` per group.
    """
    if df.empty:
        return {"curves": [], "median_survival": {}}

    if timeline is None:
        max_t = int(df[duration_col].max()) if duration_col in df.columns else 60
        timeline = list(range(0, max_t + 1))

    tl = np.array(timeline)
    curves: list[dict[str, Any]] = []
    median_survival: dict[str, float] = {}

    def _fit_group(sub: pd.DataFrame, label: str) -> None:
        T = sub[duration_col].clip(lower=0)
        E = sub[event_col].astype(int)

        if _LIFELINES_OK:
            kmf = KaplanMeierFitter()
            kmf.fit(T, event_observed=E, timeline=tl, label=label)
            sf = kmf.survival_function_at_times(tl).values.tolist()
            ci_lower = kmf.confidence_interval_survival_function_["KM_estimate_lower_0.95"]
            ci_upper = kmf.confidence_interval_survival_function_["KM_estimate_upper_0.95"]
            ci_lower_vals = np.interp(tl, ci_lower.index, ci_lower.values).tolist()
            ci_upper_vals = np.interp(tl, ci_upper.index, ci_upper.values).tolist()
            median_t = float(kmf.median_survival_time_) if not np.isinf(kmf.median_survival_time_) else None
        else:
            # Scipy fallback — simple Nelson-Åalen estimator
            from scipy.stats import kaplan_meier_estimator  # type: ignore
            event_mask = E.astype(bool)
            km_time, km_sf = kaplan_meier_estimator(event_mask, T)
            sf = np.interp(tl, km_time, km_sf, right=km_sf[-1]).tolist()
            ci_lower_vals = ci_upper_vals = sf
            idx = np.searchsorted(sf[::-1], 0.5)
            median_t = float(km_time[-(idx + 1)]) if idx < len(km_time) else None

        curves.append({
            "label": label,
            "timeline": tl.tolist(),
            "survival": sf,
            "ci_lower": ci_lower_vals,
            "ci_upper": ci_upper_vals,
        })
        median_survival[label] = median_t  # type: ignore[assignment]

    if group_col and group_col in df.columns:
        for grp, sub in df.groupby(group_col):
            _fit_group(sub, str(grp))
    else:
        _fit_group(df, "overall")

    return {"curves": curves, "median_survival": median_survival}


# ── Cox Proportional Hazards ─────────────────────────────────────────────────

def cox_ph(
    df: pd.DataFrame,
    duration_col: str = "tenure_months",
    event_col: str = "churned",
    covariates: list[str] | None = None,
) -> dict[str, Any]:
    """
    Fit Cox Proportional Hazards model.

    Returns hazard ratios, confidence intervals, p-values, and concordance.
    """
    if not _LIFELINES_OK:
        return {"error": "lifelines not installed", "hazard_ratios": {}}

    default_covariates = [
        "monthly_charges", "tenure_months", "num_support_tickets_30d",
        "feature_adoption_rate", "nps_score",
    ]
    cols = covariates or default_covariates
    available = [c for c in cols if c in df.columns]
    if not available:
        return {"error": "No covariate columns found", "hazard_ratios": {}}

    sub = df[[duration_col, event_col, *available]].dropna()
    if len(sub) < 20:
        return {"error": "Insufficient data for Cox PH (< 20 rows)", "hazard_ratios": {}}

    cph = CoxPHFitter(penalizer=0.1)
    try:
        cph.fit(sub, duration_col=duration_col, event_col=event_col)
    except Exception as exc:
        return {"error": str(exc), "hazard_ratios": {}}

    summary = cph.summary
    hazard_ratios: dict[str, dict[str, float]] = {}
    for feature in summary.index:
        hazard_ratios[str(feature)] = {
            "hazard_ratio": float(np.exp(summary.loc[feature, "coef"])),
            "ci_lower": float(np.exp(summary.loc[feature, "coef lower 95%"])),
            "ci_upper": float(np.exp(summary.loc[feature, "coef upper 95%"])),
            "p_value": float(summary.loc[feature, "p"]),
            "significant": bool(summary.loc[feature, "p"] < 0.05),
        }

    return {
        "concordance_index": float(cph.concordance_index_),
        "log_likelihood": float(cph.log_likelihood_),
        "hazard_ratios": hazard_ratios,
        "n_observations": int(len(sub)),
    }


# ── Churn-risk time-to-event projections ─────────────────────────────────────

def time_to_churn_projection(
    df: pd.DataFrame,
    churn_prob_col: str = "churn_probability",
    tenure_col: str = "tenure_months",
    n_months: int = 12,
) -> dict[str, Any]:
    """
    Project expected churn counts over the next N months using
    the empirical churn-probability distribution.
    """
    if churn_prob_col not in df.columns:
        return {"error": f"Column '{churn_prob_col}' not found"}

    probs = df[churn_prob_col].clip(0, 1)
    monthly_hazard = 1 - (1 - probs) ** (1 / max(n_months, 1))

    projection: list[dict[str, Any]] = []
    active = np.ones(len(df))
    for month in range(1, n_months + 1):
        expected_churns = float((active * monthly_hazard).sum())
        active = active * (1 - monthly_hazard)
        projection.append({
            "month": month,
            "expected_churns": round(expected_churns, 1),
            "expected_active": round(float(active.sum()), 1),
        })

    return {
        "projection": projection,
        "total_expected_churns": sum(r["expected_churns"] for r in projection),
        "horizon_months": n_months,
    }
