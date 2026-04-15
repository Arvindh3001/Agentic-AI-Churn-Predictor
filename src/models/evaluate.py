"""
Full Evaluation Suite
======================
Computes all model performance metrics including:
    - AUC-ROC (primary)
    - F1-Score (macro)
    - Brier Score (calibration)
    - Confusion Matrix
    - Calibration Curve data
    - Lift Curve data
    - Per-threshold precision/recall

Usage:
    from src.models.evaluate import evaluate_model, plot_calibration_curve

    metrics = evaluate_model(model, X_test, y_test)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import structlog
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = structlog.get_logger(__name__)


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "model",
    threshold: float = 0.5,
    save_plots: bool = False,
    plot_dir: str | Path = "reports/explainability",
) -> dict[str, float]:
    """
    Compute the full evaluation suite for a binary classification model.

    Args:
        model: Fitted sklearn-compatible model with predict_proba.
        X_test: Test feature matrix.
        y_test: Ground-truth binary labels.
        model_name: Used in plot titles and filenames.
        threshold: Decision threshold for class predictions.
        save_plots: Whether to save calibration/ROC plots to disk.
        plot_dir: Directory for saved plots.

    Returns:
        Dict of metric_name → scalar value (MLflow-loggable).
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # Core metrics
    auc_roc = roc_auc_score(y_test, y_prob)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_positive = f1_score(y_test, y_pred, average="binary")
    brier = brier_score_loss(y_test, y_prob)
    ll = log_loss(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    specificity = tn / (tn + fp + 1e-9)

    metrics: dict[str, float] = {
        "auc_roc": float(auc_roc),
        "f1_macro": float(f1_macro),
        "f1_positive": float(f1_positive),
        "brier_score": float(brier),
        "log_loss": float(ll),
        "avg_precision": float(avg_precision),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "threshold": float(threshold),
    }

    logger.info(
        "Model evaluated",
        model=model_name,
        auc_roc=round(auc_roc, 4),
        f1_macro=round(f1_macro, 4),
        brier=round(brier, 4),
    )

    print(f"\n{'=' * 60}")
    print(f"  {model_name.upper()} — Evaluation Report")
    print(f"{'=' * 60}")
    print(f"  AUC-ROC:      {auc_roc:.4f}")
    print(f"  F1 (macro):   {f1_macro:.4f}")
    print(f"  F1 (churn):   {f1_positive:.4f}")
    print(f"  Brier Score:  {brier:.4f}")
    print(f"  Log Loss:     {ll:.4f}")
    print(f"  Avg Precision:{avg_precision:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TP={tp}  FP={fp}")
    print(f"    FN={fn}  TN={tn}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Stay', 'Churn'])}")

    if save_plots:
        plot_dir = Path(plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        _save_roc_curve(y_test, y_prob, auc_roc, model_name, plot_dir)
        _save_calibration_curve(y_test, y_prob, model_name, plot_dir)
        _save_lift_curve(y_test, y_prob, model_name, plot_dir)

    return metrics


def _save_roc_curve(
    y_test: np.ndarray,
    y_prob: np.ndarray,
    auc: float,
    model_name: str,
    plot_dir: Path,
) -> None:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / f"roc_{model_name}.png", dpi=120)
    plt.close(fig)
    logger.debug("ROC curve saved", model=model_name)


def _save_calibration_curve(
    y_test: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    plot_dir: Path,
) -> None:
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, y_prob, n_bins=10
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(mean_predicted_value, fraction_of_positives, "s-", label=model_name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfectly calibrated")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"Calibration Curve — {model_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / f"calibration_{model_name}.png", dpi=120)
    plt.close(fig)


def _save_lift_curve(
    y_test: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    plot_dir: Path,
) -> None:
    """Cumulative lift curve — how much better than random."""
    sorted_idx = np.argsort(y_prob)[::-1]
    y_sorted = y_test[sorted_idx]
    cumulative_churn = np.cumsum(y_sorted) / y_test.sum()
    population_pct = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
    lift = cumulative_churn / population_pct

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(population_pct * 100, lift, label=model_name)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="Baseline")
    ax.set_xlabel("% of Population Targeted")
    ax.set_ylabel("Cumulative Lift")
    ax.set_title(f"Lift Curve — {model_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / f"lift_{model_name}.png", dpi=120)
    plt.close(fig)


def find_optimal_threshold(
    y_test: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
) -> float:
    """
    Find the probability threshold that maximises a given metric.

    Args:
        metric: One of 'f1', 'precision', 'recall', 'gmean'.

    Returns:
        Optimal threshold float.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

    if metric == "f1":
        f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
        best_idx = np.argmax(f1_scores[:-1])
        return float(thresholds[best_idx])
    elif metric == "precision":
        best_idx = np.argmax(precisions[:-1])
        return float(thresholds[best_idx])
    elif metric == "recall":
        best_idx = np.argmax(recalls[:-1])
        return float(thresholds[best_idx])

    # geometric mean of sensitivity and specificity
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
    gmean = np.sqrt(tpr * (1 - fpr))
    best_idx = np.argmax(gmean)
    return float(roc_thresholds[best_idx])
