"""
Feature Engineering Module
===========================
Scikit-learn compatible transformer that creates all derived features
from the raw customer DataFrame before encoding and scaling.

Derived Features:
    - usage_decline_rate      : (usage_90d - usage_30d) / (usage_90d + ε)
    - support_escalation_flag : 1 if num_support_tickets_30d >= 5
    - charge_per_month_normalised : monthly_charges / (tenure_months + 1)
    - tenure_bin              : ordinal bucket (0–4) based on tenure_months
    - login_trend             : login_frequency_30d / (days_since_last_login + 1)
    - nps_segment             : numeric encoding of NPS promoter/passive/detractor
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import structlog
from sklearn.base import BaseEstimator, TransformerMixin

logger = structlog.get_logger(__name__)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Compute all derived and temporal features.
    Expects a DataFrame with the raw churn schema columns.
    """

    # Tenure buckets: 0–6m, 7–12m, 13–24m, 25–48m, 49–84m
    TENURE_BINS = [0, 6, 12, 24, 48, 85]
    TENURE_LABELS = [0, 1, 2, 3, 4]

    def fit(self, X: pd.DataFrame, y: Any = None) -> "FeatureEngineer":
        # Nothing to learn from training data — purely deterministic transforms
        return self

    def transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """Apply all feature engineering transforms and return enriched DataFrame."""
        X = X.copy()

        X = self._usage_features(X)
        X = self._support_features(X)
        X = self._charge_features(X)
        X = self._tenure_features(X)
        X = self._login_features(X)
        X = self._nps_features(X)

        logger.debug(
            "Feature engineering complete",
            original_cols=len(X.columns) - 6,  # rough count of added cols
            final_cols=len(X.columns),
        )
        return X

    # ------------------------------------------------------------------ #
    # Private transform methods
    # ------------------------------------------------------------------ #

    def _usage_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Temporal usage trend features."""
        if all(c in X.columns for c in ["usage_90d", "usage_30d"]):
            X["usage_decline_rate"] = (
                (X["usage_90d"] - X["usage_30d"]) / (X["usage_90d"].clip(lower=1e-6))
            ).clip(-2.0, 2.0)
        else:
            X["usage_decline_rate"] = 0.0

        if all(c in X.columns for c in ["usage_30d", "usage_60d", "usage_90d"]):
            # Linear trend slope via simple 3-point approximation
            X["usage_trend_slope"] = (X["usage_30d"] - X["usage_90d"]) / 60.0
        else:
            X["usage_trend_slope"] = 0.0

        return X

    def _support_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Support ticket escalation flag."""
        if "num_support_tickets_30d" in X.columns:
            X["support_escalation_flag"] = (X["num_support_tickets_30d"] >= 5).astype(int)
        else:
            X["support_escalation_flag"] = 0
        return X

    def _charge_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Billing-derived features."""
        if all(c in X.columns for c in ["monthly_charges", "tenure_months"]):
            X["charge_per_month_normalised"] = X["monthly_charges"] / (X["tenure_months"] + 1)
        else:
            X["charge_per_month_normalised"] = X.get("monthly_charges", 0.0)

        if all(c in X.columns for c in ["total_charges", "monthly_charges"]):
            expected_total = X["monthly_charges"] * X.get("tenure_months", 1)
            X["charge_anomaly"] = (
                (X["total_charges"] - expected_total) / (expected_total.clip(lower=1e-6))
            ).clip(-3.0, 3.0)
        else:
            X["charge_anomaly"] = 0.0

        return X

    def _tenure_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Tenure bucket + contract stability score."""
        if "tenure_months" in X.columns:
            X["tenure_bin"] = pd.cut(
                X["tenure_months"],
                bins=self.TENURE_BINS,
                labels=self.TENURE_LABELS,
                right=True,
                include_lowest=True,
            ).astype(float)
        else:
            X["tenure_bin"] = 0.0

        return X

    def _login_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Login behaviour features."""
        if all(c in X.columns for c in ["login_frequency_30d", "days_since_last_login"]):
            X["login_trend"] = X["login_frequency_30d"] / (
                X["days_since_last_login"].clip(lower=1) + 1
            )
        else:
            X["login_trend"] = 0.0

        return X

    def _nps_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """NPS promoter/passive/detractor segment."""
        if "nps_score" in X.columns:
            X["nps_segment"] = pd.cut(
                X["nps_score"],
                bins=[-101, 0, 50, 101],
                labels=[0, 1, 2],  # detractor, passive, promoter
                right=True,
            ).astype(float)
        else:
            X["nps_segment"] = 1.0

        return X


def get_feature_names() -> list[str]:
    """Return the full list of expected feature columns after engineering."""
    raw_numeric = [
        "age",
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "num_support_tickets_30d",
        "login_frequency_30d",
        "feature_adoption_rate",
        "nps_score",
        "usage_30d",
        "usage_60d",
        "usage_90d",
        "days_since_last_login",
    ]
    derived = [
        "usage_decline_rate",
        "usage_trend_slope",
        "support_escalation_flag",
        "charge_per_month_normalised",
        "charge_anomaly",
        "tenure_bin",
        "login_trend",
        "nps_segment",
    ]
    categorical = ["contract_type", "plan_tier", "payment_method"]
    return raw_numeric + derived + categorical
