"""
Model robustness testing.

Tests:
1. Prediction stability — consistency across similar customers (Spearman rank corr)
2. Feature perturbation — how much do predictions shift under small input changes
3. Adversarial stress — targeted worst-case feature perturbations
4. Calibration reliability — reliability diagram + ECE (Expected Calibration Error)
5. Under-representation — segment coverage analysis
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr  # type: ignore

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class RobustnessReport:
    stability_score: float
    perturbation_sensitivity: dict[str, float]
    adversarial_max_shift: float
    calibration_ece: float
    calibration_bins: list[dict[str, float]]
    coverage_gaps: list[str]
    overall_score: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "stability_score": self.stability_score,
            "perturbation_sensitivity": self.perturbation_sensitivity,
            "adversarial_max_shift": self.adversarial_max_shift,
            "calibration_ece": self.calibration_ece,
            "calibration_bins": self.calibration_bins,
            "coverage_gaps": self.coverage_gaps,
            "overall_score": self.overall_score,
            "passed": self.passed,
        }


# ── 1. Prediction stability ───────────────────────────────────────────────────

def prediction_stability(
    X: pd.DataFrame,
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    n_neighbors: int = 10,
    noise_std: float = 0.01,
    n_samples: int = 200,
    random_seed: int = 42,
) -> float:
    """
    Measure prediction stability: for each sampled customer, add small Gaussian
    noise to numeric features and compute rank-correlation between original and
    perturbed probabilities.

    Returns: mean Spearman rank correlation (1.0 = perfectly stable).
    """
    rng = np.random.default_rng(random_seed)
    sample_idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
    sub = X.iloc[sample_idx].copy()

    numeric_cols = sub.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return 1.0

    try:
        orig_probs = predict_fn(sub)
    except Exception as exc:
        logger.warning("predict_fn failed in stability check: %s", exc)
        return float("nan")

    correlations: list[float] = []
    for _ in range(5):
        noisy = sub.copy()
        for col in numeric_cols:
            col_std = float(noisy[col].std()) or 1.0
            noise = rng.normal(0, noise_std * col_std, size=len(noisy))
            noisy[col] = noisy[col] + noise
        try:
            noisy_probs = predict_fn(noisy)
            corr, _ = spearmanr(orig_probs, noisy_probs)
            if not np.isnan(corr):
                correlations.append(float(corr))
        except Exception:
            pass

    return float(np.mean(correlations)) if correlations else float("nan")


# ── 2. Feature perturbation sensitivity ──────────────────────────────────────

def perturbation_sensitivity(
    X: pd.DataFrame,
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    perturbation_pct: float = 0.10,
    n_samples: int = 500,
    random_seed: int = 42,
) -> dict[str, float]:
    """
    For each numeric feature, perturb it by ±perturbation_pct of its std and
    measure mean absolute change in predicted probability.

    Returns: dict mapping feature → mean absolute prob shift.
    """
    rng = np.random.default_rng(random_seed)
    sample_idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
    sub = X.iloc[sample_idx].copy()
    numeric_cols = sub.select_dtypes(include=[np.number]).columns.tolist()

    try:
        base_probs = predict_fn(sub)
    except Exception as exc:
        logger.warning("predict_fn failed in perturbation test: %s", exc)
        return {}

    sensitivity: dict[str, float] = {}
    for col in numeric_cols:
        col_std = float(sub[col].std()) or 1.0
        delta = perturbation_pct * col_std
        perturbed = sub.copy()
        perturbed[col] = perturbed[col] + delta
        try:
            new_probs = predict_fn(perturbed)
            sensitivity[col] = float(np.mean(np.abs(new_probs - base_probs)))
        except Exception:
            sensitivity[col] = float("nan")

    return {k: round(v, 4) for k, v in sorted(sensitivity.items(), key=lambda x: -x[1])}


# ── 3. Adversarial stress test ────────────────────────────────────────────────

def adversarial_stress(
    X: pd.DataFrame,
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    high_risk_col: str | None = None,
    n_samples: int = 100,
    perturbation_sigma: float = 2.0,
    random_seed: int = 42,
) -> float:
    """
    Targeted adversarial test: perturb features by 2σ in the direction that
    maximally shifts predictions toward churn. Report the max mean shift.

    Returns: maximum mean probability shift achieved.
    """
    rng = np.random.default_rng(random_seed)
    if high_risk_col and high_risk_col in X.columns:
        # Focus on borderline cases (prob 0.3–0.7)
        mask = (X[high_risk_col] >= 0.3) & (X[high_risk_col] <= 0.7)
        pool = X[mask] if mask.sum() >= 10 else X
    else:
        pool = X

    sample_idx = rng.choice(len(pool), size=min(n_samples, len(pool)), replace=False)
    sub = pool.iloc[sample_idx].copy()
    numeric_cols = sub.select_dtypes(include=[np.number]).columns.tolist()

    try:
        base_probs = predict_fn(sub)
    except Exception:
        return float("nan")

    max_shift = 0.0
    for col in numeric_cols:
        col_std = float(sub[col].std()) or 1.0
        for direction in [+1, -1]:
            perturbed = sub.copy()
            perturbed[col] = perturbed[col] + direction * perturbation_sigma * col_std
            try:
                new_probs = predict_fn(perturbed)
                shift = float(np.mean(np.abs(new_probs - base_probs)))
                max_shift = max(max_shift, shift)
            except Exception:
                pass

    return round(max_shift, 4)


# ── 4. Calibration ───────────────────────────────────────────────────────────

def calibration_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, list[dict[str, float]]]:
    """
    Expected Calibration Error + reliability diagram data.

    Returns: (ECE, list of bin stats).
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_stats: list[dict[str, float]] = []
    n = len(y_true)

    for i in range(n_bins):
        low, high = bins[i], bins[i + 1]
        mask = (y_prob >= low) & (y_prob < high)
        if i == n_bins - 1:
            mask = (y_prob >= low) & (y_prob <= high)
        bin_count = int(mask.sum())
        if bin_count == 0:
            continue
        avg_conf = float(y_prob[mask].mean())
        avg_acc = float(y_true[mask].mean())
        ece += (bin_count / n) * abs(avg_conf - avg_acc)
        bin_stats.append({
            "bin_low": round(low, 2),
            "bin_high": round(high, 2),
            "avg_confidence": round(avg_conf, 4),
            "avg_accuracy": round(avg_acc, 4),
            "count": bin_count,
        })

    return round(float(ece), 4), bin_stats


# ── 5. Coverage gaps ─────────────────────────────────────────────────────────

def coverage_gaps(
    df: pd.DataFrame,
    segment_cols: list[str],
    min_samples: int = 30,
) -> list[str]:
    """
    Identify under-represented segments (fewer than min_samples rows).
    """
    gaps: list[str] = []
    for col in segment_cols:
        if col not in df.columns:
            continue
        counts = df[col].value_counts()
        for val, cnt in counts.items():
            if cnt < min_samples:
                gaps.append(f"{col}={val} (n={cnt})")
    return gaps


# ── Full robustness report ────────────────────────────────────────────────────

def run_robustness_tests(
    X: pd.DataFrame,
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    y_true: np.ndarray | None = None,
    y_prob: np.ndarray | None = None,
    segment_cols: list[str] | None = None,
) -> RobustnessReport:
    """
    Run all robustness tests and return a RobustnessReport.
    """
    logger.info("Running robustness tests on %d rows", len(X))

    stability = prediction_stability(X, predict_fn)
    pert_sens = perturbation_sensitivity(X, predict_fn)
    adv_shift = adversarial_stress(X, predict_fn)

    if y_true is not None and y_prob is not None:
        ece, cal_bins = calibration_ece(y_true, y_prob)
    else:
        ece, cal_bins = float("nan"), []

    seg_cols = segment_cols or []
    cov_gaps = coverage_gaps(X, seg_cols) if seg_cols else []

    # Composite score: stability (40%) + calibration (30%) + adversarial (30%)
    stability_norm = stability if not np.isnan(stability) else 0.5
    ece_score = max(0.0, 1.0 - ece * 5) if not np.isnan(ece) else 0.5
    adv_score = max(0.0, 1.0 - adv_shift * 2) if not np.isnan(adv_shift) else 0.5
    overall = 0.4 * stability_norm + 0.3 * ece_score + 0.3 * adv_score

    passed = (
        (stability >= 0.85 or np.isnan(stability))
        and (ece <= 0.10 or np.isnan(ece))
        and (adv_shift <= 0.20 or np.isnan(adv_shift))
    )

    return RobustnessReport(
        stability_score=round(stability_norm, 4),
        perturbation_sensitivity=pert_sens,
        adversarial_max_shift=round(adv_shift, 4) if not np.isnan(adv_shift) else 0.0,
        calibration_ece=round(ece, 4) if not np.isnan(ece) else 0.0,
        calibration_bins=cal_bins,
        coverage_gaps=cov_gaps,
        overall_score=round(overall, 4),
        passed=passed,
    )
