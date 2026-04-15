"""
Explainability Benchmarking
=============================
Evaluates explanation quality across three dimensions:

    1. Fidelity    — how well does the local explanation approximate the model
                     in the neighbourhood of the instance?
    2. Stability   — are explanations consistent for similar customers?
    3. SHAP-LIME Agreement — rank overlap between the two methods.

Usage:
    from src.explainability.benchmarking import ExplainabilityBenchmark

    bench = ExplainabilityBenchmark(model, shap_explainer, lime_explainer)
    report = bench.run(X_sample, n_instances=50)
    print(report)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import structlog
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from src.explainability.lime_explainer import LimeExplainer, compute_shap_lime_agreement
from src.explainability.shap_explainer import ShapExplainer

logger = structlog.get_logger(__name__)


class ExplainabilityBenchmark:
    """
    Runs fidelity, stability, and agreement benchmarks.

    Args:
        model: Fitted production model.
        shap_explainer: Fitted ShapExplainer instance.
        lime_explainer: Fitted LimeExplainer instance.
    """

    def __init__(
        self,
        model: Any,
        shap_explainer: ShapExplainer,
        lime_explainer: LimeExplainer,
    ) -> None:
        self.model = model
        self.shap = shap_explainer
        self.lime = lime_explainer

    def run(
        self,
        X_sample: np.ndarray,
        n_instances: int = 30,
        top_k: int = 5,
        n_neighbours: int = 10,
        seed: int = 42,
    ) -> dict[str, Any]:
        """
        Run the full benchmark suite on a sample of instances.

        Args:
            X_sample: Feature matrix to sample from.
            n_instances: Number of instances to benchmark.
            top_k: Top-K features to use in agreement metric.
            n_neighbours: Neighbours for stability test.
            seed: Random seed for sampling.

        Returns:
            Dict with per-metric mean scores and per-instance details.
        """
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(X_sample), size=min(n_instances, len(X_sample)), replace=False)
        subset = X_sample[indices]

        fidelity_scores: list[float] = []
        stability_scores: list[float] = []
        agreement_scores: list[float] = []
        per_instance: list[dict[str, Any]] = []

        for i, x in enumerate(subset):
            # SHAP explanation
            shap_result = self.shap.explain_instance(x)
            shap_contribs: dict[str, float] = shap_result["feature_contributions"]

            # LIME explanation
            lime_result = self.lime.explain_instance(x, num_features=top_k * 2)
            lime_weights: dict[str, float] = lime_result["feature_weights"]

            # 1. Fidelity (LIME local model)
            fid = _lime_fidelity(self.model, self.lime._explainer, x, lime_result["_explanation"])
            fidelity_scores.append(fid)

            # 2. Stability (SHAP rank consistency over neighbourhood)
            stab = _shap_stability(self.shap, x, X_sample, n_neighbours=n_neighbours)
            stability_scores.append(stab)

            # 3. SHAP-LIME agreement
            agree = compute_shap_lime_agreement(shap_contribs, lime_weights, top_k=top_k)
            agreement_scores.append(agree)

            per_instance.append(
                {
                    "index": int(indices[i]),
                    "fidelity": round(fid, 4),
                    "stability": round(stab, 4),
                    "agreement": round(agree, 4),
                    "churn_prob": shap_result["churn_probability"],
                }
            )

            if (i + 1) % 10 == 0:
                logger.info("Benchmarking progress", completed=i + 1, total=len(subset))

        report: dict[str, Any] = {
            "n_instances": len(subset),
            "top_k": top_k,
            "mean_fidelity": round(float(np.mean(fidelity_scores)), 4),
            "mean_stability": round(float(np.mean(stability_scores)), 4),
            "mean_agreement": round(float(np.mean(agreement_scores)), 4),
            "std_fidelity": round(float(np.std(fidelity_scores)), 4),
            "std_stability": round(float(np.std(stability_scores)), 4),
            "std_agreement": round(float(np.std(agreement_scores)), 4),
            "per_instance": per_instance,
        }

        logger.info(
            "Explainability benchmark complete",
            mean_fidelity=report["mean_fidelity"],
            mean_stability=report["mean_stability"],
            mean_agreement=report["mean_agreement"],
        )
        _print_report(report)
        return report


# ------------------------------------------------------------------ #
# Metric implementations
# ------------------------------------------------------------------ #

def _lime_fidelity(
    model: Any,
    lime_explainer: Any,
    x: np.ndarray,
    explanation: Any,
    neighbourhood_size: int = 500,
) -> float:
    """
    Fidelity: AUC between LIME surrogate predictions and true model probabilities
    in the local neighbourhood of x.

    Higher = surrogate better approximates the black-box model locally.
    """
    try:
        # Sample neighbourhood using LIME's internal perturbation
        neighbourhood, _ = lime_explainer.data_inverse(x, neighbourhood_size, "gaussian")
        true_probs = model.predict_proba(neighbourhood)[:, 1]
        lime_probs = explanation.local_model.predict(
            lime_explainer.scaler.transform(neighbourhood)
        )
        # Convert LIME linear predictions to [0,1] range for AUC
        lime_scaled = 1 / (1 + np.exp(-lime_probs))
        auc = roc_auc_score((true_probs > 0.5).astype(int), lime_scaled)
        return float(auc)
    except Exception:
        # Fallback: R² between LIME approximation and model in neighbourhood
        return _lime_fidelity_r2(model, lime_explainer, x, neighbourhood_size)


def _lime_fidelity_r2(
    model: Any,
    lime_explainer: Any,
    x: np.ndarray,
    neighbourhood_size: int,
) -> float:
    """R² fidelity fallback when AUC computation fails."""
    try:
        neighbourhood, _ = lime_explainer.data_inverse(x, neighbourhood_size, "gaussian")
        true_probs = model.predict_proba(neighbourhood)[:, 1]
        lime_probs = lime_explainer.scaler.transform(neighbourhood).dot(
            lime_explainer.predict_fn(neighbourhood)
        )
        ss_res = np.sum((true_probs - lime_probs) ** 2)
        ss_tot = np.sum((true_probs - true_probs.mean()) ** 2)
        return float(max(0.0, 1 - ss_res / (ss_tot + 1e-9)))
    except Exception:
        return 0.5  # neutral fallback


def _shap_stability(
    shap_explainer: ShapExplainer,
    x: np.ndarray,
    X_pool: np.ndarray,
    n_neighbours: int = 10,
    top_k: int = 5,
) -> float:
    """
    Stability: average rank correlation of SHAP top-K feature rankings
    across nearest neighbours of x.

    Score of 1.0 = perfectly stable explanations in the neighbourhood.
    """
    nbrs = NearestNeighbors(n_neighbors=n_neighbours + 1, metric="euclidean")
    nbrs.fit(X_pool)
    _, indices = nbrs.kneighbors(x.reshape(1, -1))
    neighbour_indices = indices[0][1:]  # exclude x itself

    x_result = shap_explainer.explain_instance(x)
    x_ranking = _top_k_ranking(x_result["feature_contributions"], top_k)

    correlations: list[float] = []
    for idx in neighbour_indices:
        neighbour = X_pool[idx]
        n_result = shap_explainer.explain_instance(neighbour)
        n_ranking = _top_k_ranking(n_result["feature_contributions"], top_k)

        # Spearman rank correlation over shared features
        corr = _rank_correlation(x_ranking, n_ranking, top_k)
        correlations.append(corr)

    return float(np.mean(correlations)) if correlations else 0.5


def _top_k_ranking(contributions: dict[str, float], top_k: int) -> list[str]:
    return [
        f for f, _ in sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
    ][:top_k]


def _rank_correlation(ranking_a: list[str], ranking_b: list[str], top_k: int) -> float:
    """Simple positional overlap correlation between two ranked lists."""
    pos_a = {feat: i for i, feat in enumerate(ranking_a)}
    pos_b = {feat: i for i, feat in enumerate(ranking_b)}
    common = set(ranking_a) & set(ranking_b)
    if not common:
        return 0.0
    diffs = [(pos_a[f] - pos_b[f]) ** 2 for f in common]
    n = len(common)
    spearman = 1 - (6 * sum(diffs)) / (n * (n**2 - 1) + 1e-9)
    return float(max(-1.0, min(1.0, spearman)))


def _print_report(report: dict[str, Any]) -> None:
    print("\n" + "=" * 55)
    print("  Explainability Benchmark Report")
    print("=" * 55)
    print(f"  Instances tested : {report['n_instances']}")
    print(f"  Top-K features   : {report['top_k']}")
    print(f"  Fidelity (mean)  : {report['mean_fidelity']:.4f} ± {report['std_fidelity']:.4f}")
    print(f"  Stability (mean) : {report['mean_stability']:.4f} ± {report['std_stability']:.4f}")
    print(f"  Agreement (mean) : {report['mean_agreement']:.4f} ± {report['std_agreement']:.4f}")
    print("=" * 55 + "\n")
