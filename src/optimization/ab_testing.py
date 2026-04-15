"""
A/B Testing Framework — Retention Action Experiments
======================================================
Provides deterministic group assignment, outcome logging, and
statistical analysis for retention intervention experiments.

Design:
  - Assignment is deterministic: same (customer_id, experiment_id) always
    maps to the same group, across restarts, using a seeded hash.
  - Outcomes are stored in Redis (with in-memory fallback).
  - Analysis uses Mann-Whitney U (continuous outcomes) and Chi-squared
    (binary outcomes: churned / retained).

Usage:
    from src.optimization.ab_testing import ABTestingManager

    ab = ABTestingManager()
    group = ab.assign_group("C-4821", experiment_id="retention_v1")
    # → "treatment" or "control"

    ab.log_outcome("C-4821", "retention_v1", group, outcome_value=1)

    report = ab.analyze_experiment("retention_v1")
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# In-memory fallback store: {experiment_id: [outcome_record, ...]}
_memory_store: dict[str, list[dict[str, Any]]] = {}


class ABTestingManager:
    """
    Manages A/B experiment assignment and analysis for retention actions.

    Args:
        redis_client:  Optional Redis client. Falls back to in-memory store.
        treatment_pct: Fraction of customers assigned to treatment (default 0.5).
    """

    def __init__(
        self,
        redis_client: Any | None = None,
        treatment_pct: float = 0.5,
    ) -> None:
        self._redis = redis_client
        self.treatment_pct = treatment_pct

    # ------------------------------------------------------------------ #
    # Group Assignment
    # ------------------------------------------------------------------ #

    def assign_group(self, customer_id: str, experiment_id: str) -> str:
        """
        Deterministically assign a customer to 'treatment' or 'control'.

        The assignment is stable: the same (customer_id, experiment_id) pair
        always produces the same group, ensuring consistent user experience
        across multiple pipeline runs.

        Returns:
            "treatment" or "control"
        """
        key = f"{experiment_id}:{customer_id}"
        digest = int(hashlib.md5(key.encode()).hexdigest(), 16)
        bucket = (digest % 100) / 100.0
        group = "treatment" if bucket < self.treatment_pct else "control"
        logger.debug(
            "A/B group assigned",
            customer_id=customer_id,
            experiment_id=experiment_id,
            group=group,
        )
        return group

    # ------------------------------------------------------------------ #
    # Outcome Logging
    # ------------------------------------------------------------------ #

    def log_outcome(
        self,
        customer_id: str,
        experiment_id: str,
        group: str,
        outcome_value: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Record the observed outcome for a customer in an experiment.

        Args:
            customer_id:    Customer being tracked.
            experiment_id:  Experiment this outcome belongs to.
            group:          "treatment" | "control"
            outcome_value:  e.g. 1 = retained, 0 = churned; or CLV delta.
            metadata:       Extra context (action taken, cost, etc.).
        """
        record = {
            "customer_id": customer_id,
            "experiment_id": experiment_id,
            "group": group,
            "outcome_value": outcome_value,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metadata": metadata or {},
        }

        redis_key = f"churn:ab:{experiment_id}:outcomes"
        if self._redis:
            try:
                self._redis.rpush(redis_key, json.dumps(record))
                self._redis.expire(redis_key, 86400 * 90)  # 90-day TTL
                return
            except Exception as exc:
                logger.warning("Redis AB log failed — using in-memory", error=str(exc))

        _memory_store.setdefault(experiment_id, []).append(record)

    # ------------------------------------------------------------------ #
    # Statistical Analysis
    # ------------------------------------------------------------------ #

    def analyze_experiment(
        self,
        experiment_id: str,
        min_samples: int = 30,
    ) -> dict[str, Any]:
        """
        Compute uplift and statistical significance for an experiment.

        Uses:
          - Mann-Whitney U test  (non-parametric, continuous outcomes)
          - Chi-squared test     (binary outcomes: 0 / 1)

        Returns a report dict with:
            treatment_n, control_n, treatment_mean, control_mean,
            uplift, p_value_mannwhitney, p_value_chisq,
            is_significant (p < 0.05 on either test)
        """
        records = self._load_outcomes(experiment_id)
        if not records:
            return {"error": f"No outcomes logged for experiment '{experiment_id}'"}

        treatment = [r["outcome_value"] for r in records if r["group"] == "treatment"]
        control = [r["outcome_value"] for r in records if r["group"] == "control"]

        if len(treatment) < min_samples or len(control) < min_samples:
            return {
                "warning": f"Insufficient samples (treatment={len(treatment)}, control={len(control)}, min={min_samples})",
                "treatment_n": len(treatment),
                "control_n": len(control),
            }

        from scipy import stats

        # Mann-Whitney U (works for any ordinal/continuous outcome)
        u_stat, p_mw = stats.mannwhitneyu(treatment, control, alternative="two-sided")

        # Chi-squared for binary outcomes (0/1)
        t_retained = sum(1 for v in treatment if v >= 0.5)
        c_retained = sum(1 for v in control if v >= 0.5)
        contingency = [
            [t_retained, len(treatment) - t_retained],
            [c_retained, len(control) - c_retained],
        ]
        chi2, p_chi, _, _ = stats.chi2_contingency(contingency)

        t_mean = sum(treatment) / len(treatment)
        c_mean = sum(control) / len(control)
        uplift = round(t_mean - c_mean, 4)

        report = {
            "experiment_id": experiment_id,
            "treatment_n": len(treatment),
            "control_n": len(control),
            "treatment_mean": round(t_mean, 4),
            "control_mean": round(c_mean, 4),
            "uplift": uplift,
            "uplift_pct": round(uplift / max(c_mean, 1e-9) * 100, 2),
            "u_statistic": round(u_stat, 4),
            "p_value_mannwhitney": round(p_mw, 6),
            "chi2_statistic": round(chi2, 4),
            "p_value_chisq": round(p_chi, 6),
            "is_significant": bool(p_mw < 0.05 or p_chi < 0.05),
        }

        logger.info("A/B analysis complete", experiment_id=experiment_id, uplift=uplift)
        return report

    def get_experiment_summary(self, experiment_id: str) -> dict[str, Any]:
        """Quick count of outcomes per group without full analysis."""
        records = self._load_outcomes(experiment_id)
        treatment = [r for r in records if r["group"] == "treatment"]
        control = [r for r in records if r["group"] == "control"]
        return {
            "experiment_id": experiment_id,
            "total_outcomes": len(records),
            "treatment_n": len(treatment),
            "control_n": len(control),
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _load_outcomes(self, experiment_id: str) -> list[dict[str, Any]]:
        """Load all logged outcomes for an experiment from Redis or memory."""
        redis_key = f"churn:ab:{experiment_id}:outcomes"
        if self._redis:
            try:
                raw = self._redis.lrange(redis_key, 0, -1)
                return [json.loads(r) for r in raw]
            except Exception as exc:
                logger.warning("Redis AB load failed", error=str(exc))
        return _memory_store.get(experiment_id, [])
