"""
SHAP Extractor Tool
====================
LangChain-compatible tool that computes SHAP values for a single customer
and returns a structured explanation dict ready for the Explanation Agent.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import structlog
from langchain_core.tools import tool
from pydantic import BaseModel

from agents.tools.model_tool import _load_model, _load_pipeline

logger = structlog.get_logger(__name__)

_shap_explainer_cache: dict[str, Any] = {}


class SHAPInput(BaseModel):
    customer_features: dict[str, Any]
    top_k: int = 5


@tool("shap_explanation_tool", args_schema=SHAPInput)
def shap_explanation_tool(
    customer_features: dict[str, Any],
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Compute SHAP values for a single customer and return a structured
    explanation with top risk drivers and a base value.

    Args:
        customer_features: Raw customer feature dict.
        top_k: Number of top drivers to return.

    Returns:
        Dict with shap_contributions, top_positive_drivers,
        top_negative_drivers, base_value, churn_probability.
    """
    model = _load_model()
    pipeline = _load_pipeline()

    if model is None:
        return {"error": "Model not available for SHAP computation"}

    df = pd.DataFrame([customer_features])

    try:
        if pipeline is not None:
            X = pipeline.transform(df)
            feature_names = _get_feature_names(pipeline, df)
        else:
            X = df.select_dtypes(include="number").fillna(0).values
            feature_names = list(df.select_dtypes(include="number").columns)
    except Exception as exc:
        logger.warning("SHAP pipeline transform failed", error=str(exc))
        X = df.select_dtypes(include="number").fillna(0).values
        feature_names = list(df.select_dtypes(include="number").columns)

    explainer = _get_shap_explainer(model, X)

    try:
        raw = explainer.shap_values(X)
        # Handle all SHAP return shapes:
        #   list[class0_array, class1_array]  — SHAP <0.46 TreeExplainer
        #   3-D ndarray (n_samples, n_features, n_classes) — SHAP 0.46+ binary
        #   2-D ndarray (n_samples, n_features)             — KernelExplainer / linear
        if isinstance(raw, list):
            values = raw[1][0]  # class 1, sample 0
        elif isinstance(raw, np.ndarray) and raw.ndim == 3:
            values = raw[0, :, 1]  # sample 0, all features, class 1
        else:
            values = raw[0]  # sample 0 (2-D)

        ev = explainer.expected_value
        if isinstance(ev, (list, np.ndarray)) and np.asarray(ev).ndim >= 1:
            base_value = float(np.asarray(ev).flat[-1])  # last entry = class-1 base
        else:
            base_value = float(ev)
    except Exception as exc:
        logger.error("SHAP value computation failed", error=str(exc))
        return {"error": f"SHAP failed: {exc}"}

    # Align feature names to SHAP output length
    n = len(values)
    if len(feature_names) != n:
        feature_names = [f"feature_{i}" for i in range(n)]

    contributions = {
        feat: round(float(val), 6)
        for feat, val in zip(feature_names, values)
    }

    sorted_contribs = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
    churn_prob = float(model.predict_proba(X)[0, 1])

    top_positive = [
        {"feature": k, "shap_value": v, "label": _humanise(k)}
        for k, v in sorted_contribs if v > 0
    ][:top_k]

    top_negative = [
        {"feature": k, "shap_value": v, "label": _humanise(k)}
        for k, v in sorted_contribs if v < 0
    ][:top_k]

    result = {
        "churn_probability": round(churn_prob, 4),
        "base_value": round(base_value, 4),
        "shap_contributions": contributions,
        "top_positive_drivers": top_positive,
        "top_negative_drivers": top_negative,
        "top_risk_factors": [
            {"feature": k, "shap_value": v, "label": _humanise(k), "direction": "increases_risk" if v > 0 else "reduces_risk"}
            for k, v in sorted_contribs[:top_k]
        ],
    }

    logger.info(
        "SHAP explanation computed",
        churn_prob=round(churn_prob, 4),
        top_driver=sorted_contribs[0][0] if sorted_contribs else "n/a",
    )
    return result


def _get_shap_explainer(model: Any, X_background: np.ndarray) -> Any:
    """Get or create a SHAP explainer, cached in-process."""
    import shap

    cache_key = type(model).__name__
    if cache_key in _shap_explainer_cache:
        return _shap_explainer_cache[cache_key]

    model_type = type(model).__name__
    tree_types = ("XGBClassifier", "LGBMClassifier", "RandomForestClassifier", "StackingClassifier")

    try:
        if any(t in model_type for t in tree_types):
            explainer = shap.TreeExplainer(model)
        else:
            bg = shap.sample(X_background, min(100, len(X_background)))
            explainer = shap.KernelExplainer(
                lambda x: model.predict_proba(x)[:, 1], bg
            )
        _shap_explainer_cache[cache_key] = explainer
        return explainer
    except Exception:
        bg = shap.sample(X_background, min(50, len(X_background)))
        explainer = shap.KernelExplainer(
            lambda x: model.predict_proba(x)[:, 1], bg
        )
        _shap_explainer_cache[cache_key] = explainer
        return explainer


def _get_feature_names(pipeline: Any, df: pd.DataFrame) -> list[str]:
    """Extract feature names from the fitted pipeline's final output."""
    try:
        # After OHE encoding, feature names change — try to get them
        encoder_step = None
        for name, step in pipeline.named_steps.items():
            if hasattr(step, "feature_names_out_"):
                encoder_step = step
        if encoder_step:
            return list(encoder_step.feature_names_out_)
    except Exception:
        pass
    return [f"feature_{i}" for i in range(df.shape[1])]


def _humanise(feature_name: str) -> str:
    """Convert snake_case feature name to Title Case human label."""
    return feature_name.replace("_", " ").title()
