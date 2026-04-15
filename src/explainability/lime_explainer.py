"""
LIME Explainer — Local Model-Agnostic Explanations
====================================================
Wraps LIME's LimeTabularExplainer to produce local explanations
for individual churn predictions.

Also computes a SHAP vs LIME agreement score per instance — measuring
how consistently the two methods rank the top-K features.

Usage:
    from src.explainability.lime_explainer import LimeExplainer

    explainer = LimeExplainer(model=fitted_model, X_train=X_train,
                               feature_names=feature_names)
    result = explainer.explain_instance(X_test[0])
    explainer.plot_explanation(result, save_path="reports/explainability/lime_0.png")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import structlog
from lime.lime_tabular import LimeTabularExplainer

logger = structlog.get_logger(__name__)


class LimeExplainer:
    """
    LIME tabular explainer for binary churn classification.

    Args:
        model: Fitted sklearn-compatible classifier with predict_proba.
        X_train: Training feature matrix (background distribution).
        feature_names: Column names matching the feature matrix.
        categorical_features: Indices of categorical columns (post-encoding
                              these are usually all numeric, pass [] if so).
        num_samples: Number of perturbed samples LIME generates per explanation.
    """

    def __init__(
        self,
        model: Any,
        X_train: np.ndarray,
        feature_names: list[str],
        categorical_features: list[int] | None = None,
        num_samples: int = 5_000,
        random_state: int = 42,
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        self.num_samples = num_samples

        self._explainer = LimeTabularExplainer(
            training_data=X_train,
            feature_names=feature_names,
            class_names=["Stay", "Churn"],
            categorical_features=categorical_features or [],
            mode="classification",
            discretize_continuous=True,
            random_state=random_state,
        )
        logger.info(
            "LIME explainer initialised",
            n_features=len(feature_names),
            num_samples=num_samples,
        )

    def explain_instance(
        self,
        x: np.ndarray,
        num_features: int = 10,
    ) -> dict[str, Any]:
        """
        Generate LIME explanation for a single customer.

        Args:
            x: 1-D feature vector (shape: n_features,).
            num_features: Number of top features to include in explanation.

        Returns:
            Dict with feature_weights, churn_probability, local_prediction,
            intercept, and raw LIME explanation object.
        """
        explanation = self._explainer.explain_instance(
            data_row=x,
            predict_fn=self.model.predict_proba,
            num_features=num_features,
            num_samples=self.num_samples,
            labels=(1,),  # explain churn class (index 1)
        )

        churn_prob = float(self.model.predict_proba(x.reshape(1, -1))[0, 1])
        weights = dict(explanation.as_list(label=1))
        intercept = float(explanation.intercept[1])
        local_pred = float(explanation.local_pred[0])

        # Sort by absolute weight
        sorted_weights = sorted(weights.items(), key=lambda kv: abs(kv[1]), reverse=True)

        result: dict[str, Any] = {
            "churn_probability": round(churn_prob, 4),
            "local_prediction": round(local_pred, 4),
            "intercept": round(intercept, 4),
            "feature_weights": {k: round(v, 6) for k, v in weights.items()},
            "top_drivers": [
                {"condition": k, "weight": round(v, 6)} for k, v in sorted_weights
            ],
            "top_positive_drivers": [
                {"condition": k, "weight": round(v, 6)}
                for k, v in sorted_weights if v > 0
            ][:5],
            "top_negative_drivers": [
                {"condition": k, "weight": round(v, 6)}
                for k, v in sorted_weights if v < 0
            ][:5],
            "_explanation": explanation,  # raw LIME object for plotting
        }

        logger.debug(
            "LIME explanation computed",
            churn_prob=result["churn_probability"],
            top_driver=sorted_weights[0][0] if sorted_weights else "n/a",
        )
        return result

    def plot_explanation(
        self,
        result: dict[str, Any],
        save_path: str | Path | None = None,
        title: str = "LIME Local Explanation — Churn",
    ) -> None:
        """Render horizontal bar chart of LIME feature weights."""
        explanation = result["_explanation"]
        fig = explanation.as_pyplot_figure(label=1)
        fig.suptitle(title, fontsize=12, fontweight="bold")
        fig.set_size_inches(10, 6)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=130, bbox_inches="tight")
            logger.info("LIME plot saved", path=str(save_path))
        plt.close(fig)

    def save_html(
        self,
        result: dict[str, Any],
        save_path: str | Path = "reports/explainability/lime_explanation.html",
    ) -> Path:
        """Save LIME explanation as interactive HTML."""
        explanation = result["_explanation"]
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        explanation.save_to_file(str(save_path))
        logger.info("LIME HTML saved", path=str(save_path))
        return save_path


def compute_shap_lime_agreement(
    shap_contributions: dict[str, float],
    lime_weights: dict[str, float],
    top_k: int = 5,
) -> float:
    """
    Compute agreement score between SHAP and LIME top-K feature rankings.

    Uses Rank-Biased Overlap (RBO) approximation — 1.0 = perfect agreement.

    Args:
        shap_contributions: Dict of feature → SHAP value.
        lime_weights: Dict of feature/condition → LIME weight.
        top_k: Number of top features to compare.

    Returns:
        Agreement score in [0, 1].
    """
    shap_top = [
        f for f, _ in sorted(shap_contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
    ][:top_k]

    # LIME conditions are strings like "0.50 < feature_adoption_rate <= 0.80"
    # Extract feature name from the condition string
    lime_top_features: list[str] = []
    for condition in sorted(lime_weights.items(), key=lambda kv: abs(kv[1]), reverse=True):
        cond_str = condition[0]
        matched = next(
            (f for f in shap_contributions if f in cond_str),
            cond_str,
        )
        lime_top_features.append(matched)
        if len(lime_top_features) >= top_k:
            break

    # Overlap at each rank position
    overlaps = [
        len(set(shap_top[:k]) & set(lime_top_features[:k])) / k
        for k in range(1, top_k + 1)
    ]
    agreement = float(np.mean(overlaps))

    logger.debug(
        "SHAP-LIME agreement computed",
        agreement=round(agreement, 4),
        shap_top=shap_top,
        lime_top=lime_top_features,
    )
    return round(agreement, 4)
