"""
Counterfactual Tool
====================
LangChain-compatible tool that generates ranked retention interventions
for a high-risk customer using the DiCE-backed CounterfactualEngine.

The tool loads the production model (same path as model_tool) and the
preprocessed training data, then runs DiCE or the perturbation fallback
to produce ≥ 3 actionable, business-constraint-valid counterfactuals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog
from langchain_core.tools import tool
from pydantic import BaseModel

logger = structlog.get_logger(__name__)

# Maximum discount allowed (30 % price cut)
MAX_DISCOUNT_PCT: float = 0.30
# Maximum per-customer spend on a retention action
MAX_COST_USD: float = 300.0


class CounterfactualInput(BaseModel):
    customer_features: dict[str, Any]
    current_churn_prob: float
    top_k: int = 5


@tool("counterfactual_tool", args_schema=CounterfactualInput)
def counterfactual_tool(
    customer_features: dict[str, Any],
    current_churn_prob: float,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Generate ranked counterfactual retention interventions for a customer.

    Args:
        customer_features:  Raw customer feature dict.
        current_churn_prob: Baseline churn probability from the prediction agent.
        top_k:              Number of top interventions to return.

    Returns:
        Dict with 'interventions' list and 'current_churn_prob'.
        Each intervention has: action, new_churn_prob, cost_usd,
        feasibility_score, effort, days_to_effect.
    """
    from agents.tools.model_tool import _load_model, _load_pipeline

    model = _load_model()
    pipeline = _load_pipeline()

    if model is None:
        return {"error": "Model unavailable for counterfactual generation"}

    # Try the full CounterfactualEngine; fall back to rule-based perturbation
    try:
        interventions = _run_engine(model, pipeline, customer_features, current_churn_prob, top_k)
    except Exception as exc:
        logger.warning("CounterfactualEngine failed — using rule-based fallback", error=str(exc))
        interventions = _rule_based_interventions(customer_features, current_churn_prob, model, pipeline)

    # Apply business constraints and rank
    feasible = _apply_constraints(interventions, customer_features)
    ranked = _rank_interventions(feasible)[:top_k]

    logger.info(
        "Counterfactual interventions generated",
        n_interventions=len(ranked),
        current_prob=round(current_churn_prob, 3),
    )

    return {
        "current_churn_prob": round(current_churn_prob, 4),
        "interventions": ranked,
        "n_feasible": len(ranked),
    }


# ------------------------------------------------------------------ #
# CounterfactualEngine wrapper
# ------------------------------------------------------------------ #

def _run_engine(
    model: Any,
    pipeline: Any,
    features: dict[str, Any],
    current_prob: float,
    top_k: int,
) -> list[dict[str, Any]]:
    """Run the DiCE-backed engine from src/explainability/counterfactual.py."""
    import pandas as pd
    from src.explainability.counterfactual import CounterfactualEngine

    # Load training data for DiCE background
    X_train = _load_training_data(pipeline)

    engine = CounterfactualEngine(
        model=model,
        X_train=X_train,
        feature_names=list(X_train.columns) if hasattr(X_train, "columns") else None,
    )

    customer_df = pd.DataFrame([features])
    result = engine.generate(customer_df, customer_id="current")

    raw_interventions = result.get("interventions", [])
    return raw_interventions


def _load_training_data(pipeline: Any) -> Any:
    """Load a sample of the training data for DiCE background distribution."""
    import pandas as pd

    csv_path = Path("data/synthetic/customers.csv")
    if csv_path.exists():
        df = pd.read_csv(csv_path).head(500)
        # Drop non-feature columns
        for col in ("customer_id", "churned", "churn_label"):
            if col in df.columns:
                df = df.drop(columns=[col])
        return df

    raise FileNotFoundError("Training data not found for DiCE background")


# ------------------------------------------------------------------ #
# Rule-based perturbation fallback
# ------------------------------------------------------------------ #

def _rule_based_interventions(
    features: dict[str, Any],
    current_prob: float,
    model: Any,
    pipeline: Any,
) -> list[dict[str, Any]]:
    """
    Generate counterfactuals by perturbing individual features according to
    the INTERVENTION_CATALOGUE, scoring each modified version with the model.
    """
    import pandas as pd
    import numpy as np

    CATALOGUE = [
        {
            "feature": "monthly_charges",
            "delta_pct": -0.15,
            "action_template": "Reduce monthly price by 15%",
            "cost_usd": 45.0,
            "effort": 1,
            "days_to_effect": 7,
        },
        {
            "feature": "monthly_charges",
            "delta_pct": -0.25,
            "action_template": "Reduce monthly price by 25%",
            "cost_usd": 75.0,
            "effort": 1,
            "days_to_effect": 7,
        },
        {
            "feature": "feature_adoption_rate",
            "delta_abs": 0.30,
            "action_template": "Assign dedicated onboarding / CSM support",
            "cost_usd": 120.0,
            "effort": 3,
            "days_to_effect": 30,
        },
        {
            "feature": "num_support_tickets_30d",
            "delta_abs": -2,
            "action_template": "Proactive support outreach to resolve open issues",
            "cost_usd": 60.0,
            "effort": 2,
            "days_to_effect": 14,
        },
        {
            "feature": "nps_score",
            "delta_abs": 2,
            "action_template": "Offer loyalty upgrade with annual contract incentive",
            "cost_usd": 80.0,
            "effort": 2,
            "days_to_effect": 21,
        },
        {
            "feature": "login_frequency_30d",
            "delta_abs": 5,
            "action_template": "Personalised re-engagement campaign (email + in-app)",
            "cost_usd": 20.0,
            "effort": 1,
            "days_to_effect": 10,
        },
    ]

    interventions = []
    for entry in CATALOGUE:
        modified = dict(features)
        feat = entry["feature"]
        original_val = float(modified.get(feat, 0) or 0)

        if "delta_pct" in entry:
            modified[feat] = original_val * (1 + entry["delta_pct"])
        elif "delta_abs" in entry:
            modified[feat] = original_val + entry["delta_abs"]

        # Score modified customer
        new_prob = _score_features(modified, model, pipeline, current_prob)
        prob_reduction = current_prob - new_prob

        if prob_reduction > 0.02:  # only keep interventions with meaningful impact
            interventions.append({
                "action": entry["action_template"],
                "new_churn_prob": round(new_prob, 4),
                "prob_reduction": round(prob_reduction, 4),
                "cost_usd": entry["cost_usd"],
                "effort": entry["effort"],
                "days_to_effect": entry["days_to_effect"],
                "feature_changed": feat,
            })

    return interventions


def _score_features(
    features: dict[str, Any],
    model: Any,
    pipeline: Any,
    fallback_prob: float,
) -> float:
    """Score a modified feature dict with the trained model."""
    try:
        import pandas as pd
        df = pd.DataFrame([features])
        X = pipeline.transform(df)
        return float(model.predict_proba(X)[0, 1])
    except Exception:
        return fallback_prob  # can't score — assume no change


# ------------------------------------------------------------------ #
# Constraint filtering and ranking
# ------------------------------------------------------------------ #

def _apply_constraints(
    interventions: list[dict[str, Any]],
    features: dict[str, Any],
) -> list[dict[str, Any]]:
    """Filter interventions that violate hard business rules."""
    monthly_charges = float(features.get("monthly_charges", 100) or 100)
    feasible = []

    for iv in interventions:
        cost = iv.get("cost_usd", 0)
        feat = iv.get("feature_changed", "")

        # Hard cap on per-customer spend
        if cost > MAX_COST_USD:
            continue

        # Discount cannot exceed 30 % of monthly charges
        if feat == "monthly_charges":
            original = monthly_charges
            new_charge = iv.get("new_monthly_charges", original * 0.75)
            discount_pct = (original - new_charge) / max(original, 1e-9)
            if discount_pct > MAX_DISCOUNT_PCT:
                continue

        feasible.append(iv)

    return feasible


def _rank_interventions(interventions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Score each intervention:
        score = prob_reduction × (1 / max(cost_usd, 1)) × (1 / days_to_effect)
    Higher is better.
    """
    for iv in interventions:
        reduction = iv.get("prob_reduction", iv.get("churn_prob_delta", 0.01))
        cost = max(iv.get("cost_usd", 1), 1)
        days = max(iv.get("days_to_effect", 14), 1)
        iv["feasibility_score"] = round(reduction / cost / days * 1000, 6)

    return sorted(interventions, key=lambda x: x["feasibility_score"], reverse=True)
