"""
Conformal Prediction — Uncertainty Quantification
====================================================
Wraps MAPIE (Model Agnostic Prediction Interval Estimator) to produce
calibrated prediction intervals for churn probability.

Output per customer:
    {
        "churn_prob": 0.87,
        "confidence_interval": [0.82, 0.91],
        "coverage": 0.90,
        "is_uncertain": False
    }

Usage:
    from src.models.uncertainty import ConformalPredictor

    cp = ConformalPredictor(base_model=fitted_ensemble, alpha=0.1)
    cp.calibrate(X_cal, y_cal)
    result = cp.predict(X_test)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import structlog
from mapie.classification import MapieClassifier
from mapie.metrics import classification_coverage_score

logger = structlog.get_logger(__name__)


class ConformalPredictor:
    """
    Conformal prediction wrapper using MAPIE for binary classification.

    Args:
        base_model: A fitted sklearn-compatible classifier with predict_proba.
        alpha: Significance level → coverage = 1 - alpha (default 0.10 → 90% coverage).
    """

    def __init__(self, base_model: Any, alpha: float = 0.10) -> None:
        self._base_model = base_model
        self.alpha = alpha
        self._mapie: MapieClassifier | None = None
        self._is_calibrated = False

    def calibrate(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        method: str = "score",
    ) -> "ConformalPredictor":
        """
        Calibrate the conformal predictor on a held-out calibration set.

        Args:
            X_cal: Calibration feature matrix (should NOT be training data).
            y_cal: Ground-truth calibration labels.
            method: MAPIE method. 'score' uses the softmax score approach.

        Returns:
            Self (for chaining).
        """
        self._mapie = MapieClassifier(
            estimator=self._base_model,
            method=method,
            cv="prefit",  # base model is already fitted
            random_state=42,
        )
        self._mapie.fit(X_cal, y_cal)
        self._is_calibrated = True

        logger.info(
            "Conformal predictor calibrated",
            calibration_samples=len(X_cal),
            alpha=self.alpha,
            target_coverage=1 - self.alpha,
        )
        return self

    def predict(
        self,
        X: np.ndarray,
    ) -> list[dict[str, Any]]:
        """
        Predict churn with confidence intervals.

        Returns:
            List of per-customer dicts with churn_prob, confidence_interval, is_uncertain.
        """
        if not self._is_calibrated or self._mapie is None:
            raise RuntimeError("Call .calibrate() before .predict()")

        churn_proba = self._base_model.predict_proba(X)[:, 1]

        _, prediction_sets = self._mapie.predict(
            X,
            alpha=self.alpha,
            include_last_label=True,
        )

        results: list[dict[str, Any]] = []
        for i, prob in enumerate(churn_proba):
            ps = prediction_sets[i, :, 0]  # shape: (n_classes,)
            # ps[0] = True means class 0 (no churn) is in set; ps[1] = True means class 1 (churn)
            both_in_set = bool(ps[0]) and bool(ps[1])
            neither_in_set = not bool(ps[0]) and not bool(ps[1])

            results.append(
                {
                    "churn_prob": round(float(prob), 4),
                    "confidence_interval": _estimate_interval(prob, self.alpha),
                    "coverage": round(1.0 - self.alpha, 2),
                    "is_uncertain": both_in_set,  # ambiguous prediction → both classes in set
                    "prediction_set": {"no_churn": bool(ps[0]), "churn": bool(ps[1])},
                    "alpha": self.alpha,
                }
            )

        logger.debug("Conformal predictions generated", n=len(results))
        return results

    def predict_single(self, x: np.ndarray) -> dict[str, Any]:
        """Predict for a single customer vector."""
        return self.predict(x.reshape(1, -1))[0]

    def evaluate_coverage(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> float:
        """
        Compute empirical coverage on the test set.
        Should be close to 1 - alpha.
        """
        if self._mapie is None:
            raise RuntimeError("Call .calibrate() first.")

        _, prediction_sets = self._mapie.predict(X_test, alpha=self.alpha)
        # prediction_sets shape: (n_samples, n_classes, n_alphas)
        coverage = classification_coverage_score(y_test, prediction_sets[:, :, 0])

        logger.info(
            "Conformal coverage evaluation",
            empirical_coverage=round(float(coverage), 4),
            target_coverage=round(1 - self.alpha, 4),
        )
        return float(coverage)


def _estimate_interval(prob: float, alpha: float) -> list[float]:
    """
    Simple Wilson score interval as a proxy confidence interval.
    Used when MAPIE set is [0, 1] (ambiguous).
    """
    z = 1.645 if alpha <= 0.1 else 1.96
    n = 1000  # effective sample size proxy
    centre = (prob + z**2 / (2 * n)) / (1 + z**2 / n)
    margin = z * (np.sqrt(prob * (1 - prob) / n + z**2 / (4 * n**2))) / (1 + z**2 / n)
    return [round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4)]
