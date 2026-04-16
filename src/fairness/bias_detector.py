"""
Fairness and bias detection for churn predictions.

Checks:
- Demographic parity difference (via fairlearn)
- Equalized odds difference (via fairlearn)
- Disparate impact ratio (custom — EEOC 4/5ths rule)
- Per-segment churn rate analysis
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from fairlearn.metrics import (  # type: ignore
        demographic_parity_difference,
        equalized_odds_difference,
        MetricFrame,
    )
    import sklearn.metrics as skm
    _FAIRLEARN_OK = True
except ImportError:
    _FAIRLEARN_OK = False
    logger.warning("fairlearn not installed — bias detection will use manual calculations")


# ── Thresholds ────────────────────────────────────────────────────────────────
DEMOGRAPHIC_PARITY_THRESHOLD = 0.10   # max allowed difference
EQUALIZED_ODDS_THRESHOLD = 0.10
DISPARATE_IMPACT_THRESHOLD = 0.80    # EEOC 4/5ths rule


@dataclass
class BiasCheckResult:
    attribute: str
    demographic_parity_diff: float | None
    equalized_odds_diff: float | None
    disparate_impact_ratio: float | None
    segment_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    passed: bool = True
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attribute": self.attribute,
            "demographic_parity_diff": self.demographic_parity_diff,
            "equalized_odds_diff": self.equalized_odds_diff,
            "disparate_impact_ratio": self.disparate_impact_ratio,
            "segment_stats": self.segment_stats,
            "passed": self.passed,
            "failures": self.failures,
        }


# ── Per-segment statistics ────────────────────────────────────────────────────

def _segment_stats(
    df: pd.DataFrame,
    attribute: str,
    y_pred_col: str,
    y_true_col: str | None,
    prob_col: str | None,
) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for val, sub in df.groupby(attribute):
        entry: dict[str, float] = {
            "count": int(len(sub)),
            "predicted_churn_rate": float(sub[y_pred_col].mean()),
        }
        if prob_col and prob_col in sub.columns:
            entry["avg_churn_probability"] = float(sub[prob_col].mean())
        if y_true_col and y_true_col in sub.columns:
            entry["actual_churn_rate"] = float(sub[y_true_col].mean())
            true_pos = ((sub[y_pred_col] == 1) & (sub[y_true_col] == 1)).sum()
            actual_pos = (sub[y_true_col] == 1).sum()
            entry["true_positive_rate"] = float(true_pos / actual_pos) if actual_pos else float("nan")
        stats[str(val)] = entry
    return stats


# ── Disparate impact ─────────────────────────────────────────────────────────

def _disparate_impact(
    df: pd.DataFrame,
    attribute: str,
    y_pred_col: str,
) -> float | None:
    """
    Disparate impact ratio = min(P(Y=1|A=a)) / max(P(Y=1|A=a))
    EEOC 4/5ths rule: ratio < 0.8 indicates adverse impact.
    """
    rates = df.groupby(attribute)[y_pred_col].mean()
    if rates.empty or rates.max() == 0:
        return None
    return float(rates.min() / rates.max())


# ── Main detector ─────────────────────────────────────────────────────────────

def check_bias(
    df: pd.DataFrame,
    protected_attributes: list[str],
    y_pred_col: str = "predicted_churn",
    y_true_col: str | None = "churned",
    prob_col: str | None = "churn_probability",
    prediction_threshold: float = 0.5,
) -> list[BiasCheckResult]:
    """
    Run fairness checks for a list of protected attributes.

    Args:
        df: Customer dataframe with predictions and ground-truth labels.
        protected_attributes: List of column names to check (e.g. ["gender", "region"]).
        y_pred_col: Binary prediction column (0/1). Created from prob_col if missing.
        y_true_col: Ground-truth churn column (0/1). Optional.
        prob_col: Churn probability column. Used to derive y_pred if needed.
        prediction_threshold: Threshold to binarise probabilities.

    Returns:
        List of BiasCheckResult, one per attribute.
    """
    df = df.copy()

    # Derive binary prediction from probability if needed
    if y_pred_col not in df.columns:
        if prob_col and prob_col in df.columns:
            df[y_pred_col] = (df[prob_col] >= prediction_threshold).astype(int)
        else:
            return []

    results: list[BiasCheckResult] = []

    for attr in protected_attributes:
        if attr not in df.columns:
            logger.warning("Protected attribute '%s' not in dataframe — skipping", attr)
            continue

        # Drop rows with NaN in this attribute
        sub = df.dropna(subset=[attr])
        if len(sub) < 50:
            logger.warning("Too few rows (%d) for attribute '%s' — skipping", len(sub), attr)
            continue

        y_pred = sub[y_pred_col].astype(int)
        sensitive = sub[attr]
        has_truth = y_true_col and y_true_col in sub.columns
        y_true = sub[y_true_col].astype(int) if has_truth else None

        # Demographic parity difference
        dp_diff: float | None = None
        eq_odds_diff: float | None = None

        if _FAIRLEARN_OK:
            try:
                dp_diff = float(demographic_parity_difference(
                    y_true=y_pred,  # use prediction as "truth" when no labels
                    y_pred=y_pred,
                    sensitive_features=sensitive,
                ))
                if has_truth and y_true is not None:
                    dp_diff = float(demographic_parity_difference(
                        y_true=y_true,
                        y_pred=y_pred,
                        sensitive_features=sensitive,
                    ))
                    eq_odds_diff = float(equalized_odds_difference(
                        y_true=y_true,
                        y_pred=y_pred,
                        sensitive_features=sensitive,
                    ))
            except Exception as exc:
                logger.warning("fairlearn error for '%s': %s", attr, exc)
        else:
            # Manual demographic parity: max - min predicted positive rate
            rates = sub.groupby(attr)[y_pred_col].mean()
            dp_diff = float(rates.max() - rates.min()) if len(rates) > 1 else 0.0

        di_ratio = _disparate_impact(sub, attr, y_pred_col)
        seg_stats = _segment_stats(sub, attr, y_pred_col, y_true_col, prob_col)

        # Evaluate pass/fail
        failures: list[str] = []
        if dp_diff is not None and abs(dp_diff) > DEMOGRAPHIC_PARITY_THRESHOLD:
            failures.append(
                f"Demographic parity diff {dp_diff:.3f} exceeds threshold {DEMOGRAPHIC_PARITY_THRESHOLD}"
            )
        if eq_odds_diff is not None and abs(eq_odds_diff) > EQUALIZED_ODDS_THRESHOLD:
            failures.append(
                f"Equalized odds diff {eq_odds_diff:.3f} exceeds threshold {EQUALIZED_ODDS_THRESHOLD}"
            )
        if di_ratio is not None and di_ratio < DISPARATE_IMPACT_THRESHOLD:
            failures.append(
                f"Disparate impact ratio {di_ratio:.3f} below EEOC threshold {DISPARATE_IMPACT_THRESHOLD}"
            )

        results.append(BiasCheckResult(
            attribute=attr,
            demographic_parity_diff=round(dp_diff, 4) if dp_diff is not None else None,
            equalized_odds_diff=round(eq_odds_diff, 4) if eq_odds_diff is not None else None,
            disparate_impact_ratio=round(di_ratio, 4) if di_ratio is not None else None,
            segment_stats=seg_stats,
            passed=len(failures) == 0,
            failures=failures,
        ))

    return results


# ── Summary helper ────────────────────────────────────────────────────────────

def bias_summary(results: list[BiasCheckResult]) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    return {
        "total_checks": total,
        "passed": passed,
        "failed": total - passed,
        "overall_pass": passed == total,
        "attributes_checked": [r.attribute for r in results],
        "failed_attributes": [r.attribute for r in results if not r.passed],
    }
