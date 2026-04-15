"""
Stacking Ensemble Builder
==========================
Constructs a stacking classifier with XGBoost + LightGBM + Random Forest
as base learners and Logistic Regression as the meta-learner.

Stacking strategy:
    - Base learners output cross-validated probability estimates (out-of-fold)
    - Meta-learner trains on these probability estimates
    - Final predictions use the full base learners + meta-learner stack

Usage:
    from src.models.ensemble import build_stacking_ensemble

    ensemble = build_stacking_ensemble(xgb_model, lgbm_model, rf_model)
    ensemble.fit(X_train, y_train)
    proba = ensemble.predict_proba(X_test)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

logger = structlog.get_logger(__name__)


def build_stacking_ensemble(
    xgb_model: XGBClassifier | None = None,
    lgbm_model: LGBMClassifier | None = None,
    rf_model: RandomForestClassifier | None = None,
    cv_folds: int = 5,
    random_state: int = 42,
) -> StackingClassifier:
    """
    Build and return a StackingClassifier.

    Args:
        xgb_model: Fitted or unfitted XGBoost classifier. Defaults to sensible config.
        lgbm_model: Fitted or unfitted LightGBM classifier.
        rf_model: Fitted or unfitted Random Forest classifier.
        cv_folds: Number of cross-validation folds for stacking.
        random_state: Controls reproducibility of the CV splits.

    Returns:
        sklearn StackingClassifier (not yet fitted).
    """
    _xgb = xgb_model or XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=2.7,
        eval_metric="logloss",
        random_state=random_state,
        verbosity=0,
    )

    _lgbm = lgbm_model or LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=63,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )

    _rf = rf_model or RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    meta_learner = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        solver="lbfgs",
        random_state=random_state,
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    ensemble = StackingClassifier(
        estimators=[
            ("xgboost", _xgb),
            ("lightgbm", _lgbm),
            ("random_forest", _rf),
        ],
        final_estimator=meta_learner,
        cv=cv,
        stack_method="predict_proba",
        passthrough=False,  # meta-learner only sees base probabilities
        n_jobs=-1,
    )

    logger.info(
        "Stacking ensemble created",
        base_learners=["xgboost", "lightgbm", "random_forest"],
        meta_learner="logistic_regression",
        cv_folds=cv_folds,
    )

    return ensemble


class EnsemblePredictor:
    """
    Thin wrapper around a fitted StackingClassifier that exposes
    a typed, agent-friendly interface for predictions.
    """

    def __init__(self, ensemble: StackingClassifier) -> None:
        self._ensemble = ensemble

    def predict(
        self,
        X: np.ndarray,
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        """
        Predict churn for a batch of customers.

        Returns:
            Dict with churn_probabilities, predictions, and base_learner_probas.
        """
        churn_proba = self._ensemble.predict_proba(X)[:, 1]
        predictions = (churn_proba >= threshold).astype(int)

        # Extract base learner probabilities for transparency
        base_probas: dict[str, np.ndarray] = {}
        for name, estimator in self._ensemble.named_estimators_.items():
            try:
                base_probas[name] = estimator.predict_proba(X)[:, 1]
            except Exception:
                pass

        return {
            "churn_probabilities": churn_proba.tolist(),
            "predictions": predictions.tolist(),
            "base_learner_probabilities": {k: v.tolist() for k, v in base_probas.items()},
            "threshold": threshold,
        }

    def predict_single(
        self,
        x: np.ndarray,
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        """Predict churn for a single customer feature vector."""
        result = self.predict(x.reshape(1, -1), threshold=threshold)
        return {
            "churn_probability": result["churn_probabilities"][0],
            "prediction": result["predictions"][0],
            "base_learner_probabilities": {
                k: v[0] for k, v in result["base_learner_probabilities"].items()
            },
            "threshold": threshold,
        }
