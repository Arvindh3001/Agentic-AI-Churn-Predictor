"""
SHAP Explainer — Global & Local
=================================
Provides global feature importance (summary plots, PDP, interaction heatmap)
and local per-customer explanations (waterfall / force plots).

Works with any sklearn-compatible model. Uses TreeExplainer for tree-based
models (XGBoost, LightGBM, RandomForest) and LinearExplainer for LR.
Falls back to KernelExplainer for unsupported model types.

Usage:
    from src.explainability.shap_explainer import ShapExplainer

    explainer = ShapExplainer(model=fitted_model, feature_names=feature_names)
    explainer.fit(X_train_sample)

    # Global
    global_vals = explainer.global_importance()
    explainer.plot_summary(X_test, save_path="reports/explainability/shap_summary.png")

    # Local
    local = explainer.explain_instance(X_test[0])
    explainer.plot_waterfall(X_test[0], save_path="reports/explainability/shap_waterfall_0.png")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import structlog

logger = structlog.get_logger(__name__)

# Models using TreeExplainer
_TREE_TYPES = ("XGBClassifier", "LGBMClassifier", "RandomForestClassifier",
               "GradientBoostingClassifier", "StackingClassifier")
_LINEAR_TYPES = ("LogisticRegression", "LinearSVC")


class ShapExplainer:
    """
    Unified SHAP wrapper supporting tree, linear, and generic models.

    Args:
        model: Fitted sklearn-compatible classifier.
        feature_names: Column names matching the feature matrix columns.
        background_samples: Number of background samples for KernelExplainer.
    """

    def __init__(
        self,
        model: Any,
        feature_names: list[str],
        background_samples: int = 100,
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        self.background_samples = background_samples
        self._explainer: Any = None
        self._shap_values: np.ndarray | None = None

    def fit(self, X_background: np.ndarray) -> "ShapExplainer":
        """
        Initialise the appropriate SHAP explainer on background data.

        Args:
            X_background: Representative sample of training data (used as
                          background distribution for KernelExplainer).
        """
        model_type = type(self.model).__name__

        if any(t in model_type for t in _TREE_TYPES):
            # TreeExplainer: fast, exact Shapley values for tree ensembles
            try:
                self._explainer = shap.TreeExplainer(self.model)
                logger.info("Using TreeExplainer", model_type=model_type)
            except Exception:
                self._explainer = self._make_kernel_explainer(X_background)
        elif any(t in model_type for t in _LINEAR_TYPES):
            bg = shap.sample(X_background, min(self.background_samples, len(X_background)))
            self._explainer = shap.LinearExplainer(self.model, bg)
            logger.info("Using LinearExplainer", model_type=model_type)
        else:
            self._explainer = self._make_kernel_explainer(X_background)

        return self

    def _make_kernel_explainer(self, X_background: np.ndarray) -> shap.KernelExplainer:
        bg = shap.sample(X_background, min(self.background_samples, len(X_background)))
        logger.info("Using KernelExplainer (slow — consider TreeExplainer for tree models)")
        return shap.KernelExplainer(
            lambda x: self.model.predict_proba(x)[:, 1], bg
        )

    def compute_shap_values(self, X: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for a feature matrix.

        Returns:
            Array of shape (n_samples, n_features) — SHAP values for churn class.
        """
        if self._explainer is None:
            raise RuntimeError("Call .fit() before computing SHAP values.")

        raw = self._explainer.shap_values(X)

        # TreeExplainer on multi-output models returns list [class0, class1]
        if isinstance(raw, list):
            values = raw[1]
        else:
            values = raw

        self._shap_values = values
        logger.debug("SHAP values computed", shape=values.shape)
        return values

    # ------------------------------------------------------------------ #
    # Global explainability
    # ------------------------------------------------------------------ #

    def global_importance(self, X: np.ndarray | None = None) -> pd.Series:
        """
        Compute mean absolute SHAP value per feature (global importance).

        Args:
            X: Feature matrix. Required if SHAP values not yet computed.

        Returns:
            pd.Series sorted descending by importance.
        """
        if self._shap_values is None:
            if X is None:
                raise ValueError("Provide X or call compute_shap_values() first.")
            self.compute_shap_values(X)

        mean_abs = np.abs(self._shap_values).mean(axis=0)
        importance = pd.Series(mean_abs, index=self.feature_names).sort_values(ascending=False)
        return importance

    def plot_summary(
        self,
        X: np.ndarray,
        max_display: int = 20,
        save_path: str | Path | None = None,
        plot_type: str = "dot",
    ) -> None:
        """Generate SHAP beeswarm / bar summary plot."""
        shap_values = self.compute_shap_values(X)
        X_df = pd.DataFrame(X, columns=self.feature_names)

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X_df,
            plot_type=plot_type,
            max_display=max_display,
            show=False,
        )
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=130, bbox_inches="tight")
            logger.info("SHAP summary plot saved", path=str(save_path))
        plt.close()

    def plot_interaction_heatmap(
        self,
        X: np.ndarray,
        top_n: int = 10,
        save_path: str | Path | None = None,
    ) -> None:
        """Plot SHAP feature interaction heatmap (top N features)."""
        importance = self.global_importance(X)
        top_features = importance.head(top_n).index.tolist()
        top_idx = [self.feature_names.index(f) for f in top_features if f in self.feature_names]

        shap_top = self._shap_values[:, top_idx]
        corr = pd.DataFrame(shap_top, columns=top_features).corr()

        import seaborn as sns
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        ax.set_title("SHAP Feature Interaction Heatmap")
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=130, bbox_inches="tight")
            logger.info("SHAP interaction heatmap saved", path=str(save_path))
        plt.close()

    # ------------------------------------------------------------------ #
    # Local explainability
    # ------------------------------------------------------------------ #

    def explain_instance(self, x: np.ndarray) -> dict[str, Any]:
        """
        Compute local SHAP explanation for a single customer.

        Args:
            x: 1-D feature vector (shape: n_features,).

        Returns:
            Dict with feature_contributions, base_value, prediction,
            top_positive_drivers, top_negative_drivers.
        """
        if self._explainer is None:
            raise RuntimeError("Call .fit() first.")

        x2d = x.reshape(1, -1)
        raw = self._explainer.shap_values(x2d)
        values = raw[1][0] if isinstance(raw, list) else raw[0]

        base_value = (
            self._explainer.expected_value[1]
            if isinstance(self._explainer.expected_value, (list, np.ndarray))
            else float(self._explainer.expected_value)
        )

        contributions = {
            feat: round(float(val), 6)
            for feat, val in zip(self.feature_names, values)
        }

        sorted_contribs = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)

        churn_prob = float(self.model.predict_proba(x2d)[0, 1])

        return {
            "churn_probability": round(churn_prob, 4),
            "base_value": round(float(base_value), 4),
            "feature_contributions": contributions,
            "top_positive_drivers": [
                {"feature": k, "shap_value": v}
                for k, v in sorted_contribs
                if v > 0
            ][:5],
            "top_negative_drivers": [
                {"feature": k, "shap_value": v}
                for k, v in sorted_contribs
                if v < 0
            ][:5],
        }

    def plot_waterfall(
        self,
        x: np.ndarray,
        save_path: str | Path | None = None,
    ) -> None:
        """Waterfall chart for a single customer's SHAP values."""
        if self._explainer is None:
            raise RuntimeError("Call .fit() first.")

        x2d = x.reshape(1, -1)
        raw = self._explainer.shap_values(x2d)
        values = raw[1][0] if isinstance(raw, list) else raw[0]

        base_value = (
            self._explainer.expected_value[1]
            if isinstance(self._explainer.expected_value, (list, np.ndarray))
            else float(self._explainer.expected_value)
        )

        explanation = shap.Explanation(
            values=values,
            base_values=float(base_value),
            data=x,
            feature_names=self.feature_names,
        )

        fig, _ = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(explanation, show=False)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=130, bbox_inches="tight")
            logger.info("SHAP waterfall plot saved", path=str(save_path))
        plt.close()
