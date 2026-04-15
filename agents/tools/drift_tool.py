"""
Drift Checker Tool
===================
LangChain-compatible tool that checks whether a customer's feature vector
deviates significantly from the training distribution — a per-instance
anomaly signal complementing the dataset-level drift detector.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog
from langchain_core.tools import tool
from pydantic import BaseModel
from scipy import stats

logger = structlog.get_logger(__name__)

_baseline_stats_cache: dict[str, Any] = {}

NUMERIC_FEATURES = [
    "age", "tenure_months", "monthly_charges", "total_charges",
    "num_support_tickets_30d", "login_frequency_30d", "feature_adoption_rate",
    "nps_score", "usage_30d", "usage_60d", "usage_90d", "days_since_last_login",
]

Z_SCORE_THRESHOLD = 3.0   # flag if any feature is > 3 std devs from mean


class DriftCheckInput(BaseModel):
    customer_features: dict[str, Any]


@tool("drift_check_tool", args_schema=DriftCheckInput)
def drift_check_tool(customer_features: dict[str, Any]) -> dict[str, Any]:
    """
    Check if a customer's feature values are anomalous relative to the
    training distribution. Returns a quality report with flagged features
    and an overall anomaly score.

    Args:
        customer_features: Raw customer feature dict.

    Returns:
        Dict with passed (bool), anomaly_score, anomalous_features list,
        and per-feature z-scores.
    """
    baseline = _load_baseline_stats()

    if not baseline:
        # No baseline available — return neutral result
        return {
            "passed": True,
            "anomaly_score": 0.0,
            "anomalous_features": [],
            "z_scores": {},
            "warning": "No baseline statistics available; skipping drift check.",
        }

    z_scores: dict[str, float] = {}
    anomalous: list[str] = []

    for feat in NUMERIC_FEATURES:
        val = customer_features.get(feat)
        if val is None:
            continue
        mean = baseline.get(f"{feat}_mean")
        std = baseline.get(f"{feat}_std")
        if mean is None or std is None or std < 1e-9:
            continue

        z = abs((float(val) - mean) / std)
        z_scores[feat] = round(z, 3)
        if z > Z_SCORE_THRESHOLD:
            anomalous.append(feat)

    anomaly_score = round(len(anomalous) / max(len(NUMERIC_FEATURES), 1), 4)
    passed = anomaly_score < 0.3  # < 30% of features anomalous

    result = {
        "passed": passed,
        "anomaly_score": anomaly_score,
        "anomalous_features": anomalous,
        "z_scores": z_scores,
        "z_threshold": Z_SCORE_THRESHOLD,
    }

    if anomalous:
        logger.warning(
            "Customer feature anomaly detected",
            anomalous_features=anomalous,
            anomaly_score=anomaly_score,
        )

    return result


def _load_baseline_stats() -> dict[str, float]:
    """Load training distribution statistics from disk or cache."""
    if _baseline_stats_cache:
        return _baseline_stats_cache

    # Try loading from saved pipeline baseline
    stats_path = Path("models/artifacts/baseline_stats.pkl")
    if stats_path.exists():
        with open(stats_path, "rb") as f:
            stats_dict = pickle.load(f)
        _baseline_stats_cache.update(stats_dict)
        return _baseline_stats_cache

    # Fallback: compute from synthetic data if available
    csv_path = Path("data/synthetic/customers.csv")
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        for feat in NUMERIC_FEATURES:
            if feat in df.columns:
                _baseline_stats_cache[f"{feat}_mean"] = float(df[feat].mean())
                _baseline_stats_cache[f"{feat}_std"] = float(df[feat].std())
        # Save for next run
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "wb") as f:
            pickle.dump(dict(_baseline_stats_cache), f)
        logger.info("Baseline stats computed and cached", features=len(NUMERIC_FEATURES))

    return _baseline_stats_cache
