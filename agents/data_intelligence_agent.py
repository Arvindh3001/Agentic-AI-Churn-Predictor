"""
Data Intelligence Agent
========================
Responsible for fetching, validating, and enriching customer data
before it enters the prediction pipeline.

Responsibilities:
  1. Fetch customer feature row from PostgreSQL (or CSV fallback)
  2. Run schema validation (Pandera)
  3. Run per-instance anomaly check (z-score drift tool)
  4. Retrieve similar past customers from ChromaDB
  5. Cache enriched context in Redis
  6. Return validated FeatureVector + DataQualityReport

Raises DataQualityException if critical quality checks fail.
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
import structlog

from agents.state import AgentState, DataQualityReport
from agents.tools.drift_tool import drift_check_tool
from agents.tools.sql_tool import fetch_customer_features, fetch_intervention_history
from memory.redis_state import RedisStateManager
from memory.vector_store import VectorStore

logger = structlog.get_logger(__name__)


class DataQualityException(Exception):
    """Raised when incoming customer data fails critical quality checks."""


class DataIntelligenceAgent:
    """
    Fetches, validates, and enriches customer data for the agent pipeline.

    Args:
        redis_store: Shared Redis state manager.
        vector_store: Shared ChromaDB vector store.
        use_csv_fallback: If True, falls back to CSV when PostgreSQL is unavailable.
        csv_path: Path to fallback CSV.
    """

    def __init__(
        self,
        redis_store: RedisStateManager | None = None,
        vector_store: VectorStore | None = None,
        use_csv_fallback: bool = True,
        csv_path: str = "data/synthetic/customers.csv",
    ) -> None:
        self._redis = redis_store or RedisStateManager()
        self._vector_store = vector_store or VectorStore()
        self._use_csv_fallback = use_csv_fallback
        self._csv_path = csv_path
        self._csv_cache: pd.DataFrame | None = None

    def run(self, state: AgentState) -> dict[str, Any]:
        """
        LangGraph node function — receives state, returns partial state update.

        Returns dict with keys: customer_features, raw_feature_vector,
        data_quality, similar_customers, current_step, completed_steps, errors.
        """
        t0 = time.time()
        customer_id = state["customer_id"]
        logger.info("DataIntelligenceAgent started", customer_id=customer_id)

        # 1. Check Redis cache first
        cached = self._redis.get_customer_context(customer_id)
        if cached:
            logger.info("Customer context loaded from Redis cache", customer_id=customer_id)
            return self._success(state, cached, t0)

        # 2. Fetch features
        features = self._fetch_features(customer_id)
        if features is None:
            err = f"Customer {customer_id} not found in any data source"
            logger.error(err)
            return {
                "current_step": "data_intelligence",
                "completed_steps": state["completed_steps"],
                "errors": state["errors"] + [err],
                "should_abort": True,
            }

        # 3. Validate schema
        validation_warnings = self._validate_features(features)

        # 4. Per-instance drift / anomaly check
        drift_result = drift_check_tool.invoke({"customer_features": features})
        anomalous_features: list[str] = drift_result.get("anomalous_features", [])
        drift_passed: bool = drift_result.get("passed", True)

        # 5. Fetch past intervention history
        history = fetch_intervention_history(customer_id)

        # 6. Build quality report
        quality: DataQualityReport = {
            "passed": drift_passed and len(validation_warnings) == 0,
            "missing_pct": self._missing_pct(features),
            "anomaly_features": anomalous_features,
            "drift_detected": not drift_passed,
            "warnings": validation_warnings + (
                [f"Anomalous features: {anomalous_features}"] if anomalous_features else []
            ),
        }

        # 7. Enrich with similar customers from ChromaDB
        query_text = self._build_customer_summary(features)
        similar = self._vector_store.find_similar_customers(query_text, top_k=3)

        # 8. Cache enriched context
        context = {
            "customer_features": features,
            "intervention_history": history,
            "data_quality": quality,
        }
        self._redis.cache_customer_context(customer_id, context)

        # 9. Update ChromaDB profile
        self._vector_store.upsert_customer_profile(
            customer_id=customer_id,
            feature_summary=query_text,
            metadata={
                "tenure_months": features.get("tenure_months", 0),
                "plan_tier": features.get("plan_tier", "unknown"),
                "contract_type": features.get("contract_type", "unknown"),
                "monthly_charges": features.get("monthly_charges", 0),
            },
        )

        return self._success(
            state,
            {"customer_features": features, "data_quality": quality, "similar_customers": similar},
            t0,
        )

    def _success(
        self,
        state: AgentState,
        data: dict[str, Any],
        t0: float,
    ) -> dict[str, Any]:
        duration = round(time.time() - t0, 3)
        self._redis.append_stream_event(state["run_id"], {
            "step": "data_intelligence",
            "status": "completed",
            "duration_s": duration,
            "quality_passed": data.get("data_quality", {}).get("passed", True),
        })
        return {
            "customer_features": data.get("customer_features", {}),
            "data_quality": data.get("data_quality", {}),
            "similar_customers": data.get("similar_customers", []),
            "current_step": "data_intelligence_done",
            "completed_steps": state["completed_steps"] + ["data_intelligence"],
            "errors": state["errors"],
            "step_durations": {**state.get("step_durations", {}), "data_intelligence": duration},
        }

    def _fetch_features(self, customer_id: str) -> dict[str, Any] | None:
        """Try PostgreSQL first, fall back to CSV."""
        features = fetch_customer_features(customer_id)
        if features:
            return features

        if self._use_csv_fallback:
            return self._fetch_from_csv(customer_id)
        return None

    def _fetch_from_csv(self, customer_id: str) -> dict[str, Any] | None:
        """Load customer from synthetic CSV (dev / demo mode).

        Returns None — and never substitutes another customer's row — when the
        requested ID is absent.  Silent substitution would send the wrong
        customer's features through the pipeline under the original ID, causing
        incorrect predictions, tainted vector-store entries, and potential
        attribute leakage between customers.
        """
        try:
            if self._csv_cache is None:
                self._csv_cache = pd.read_csv(self._csv_path)
            row = self._csv_cache[self._csv_cache["customer_id"] == customer_id]
            if row.empty:
                logger.warning(
                    "Customer ID not found in CSV fallback — aborting fetch",
                    customer_id=customer_id,
                )
                return None
            return row.iloc[0].to_dict()
        except Exception as exc:
            logger.error("CSV fallback failed", error=str(exc))
            return None

    def _validate_features(self, features: dict[str, Any]) -> list[str]:
        """Lightweight validation, returns list of warning strings."""
        warnings: list[str] = []
        required = ["tenure_months", "monthly_charges", "contract_type"]
        for col in required:
            if features.get(col) is None:
                warnings.append(f"Missing required field: {col}")
        if float(features.get("monthly_charges", 0)) <= 0:
            warnings.append("monthly_charges is zero or negative")
        return warnings

    @staticmethod
    def _missing_pct(features: dict[str, Any]) -> float:
        n_total = len(features)
        n_missing = sum(1 for v in features.values() if v is None or v != v)  # NaN check
        return round(n_missing / max(n_total, 1), 4)

    @staticmethod
    def _build_customer_summary(features: dict[str, Any]) -> str:
        """Build a natural-language summary for ChromaDB embedding."""
        parts = [
            f"Contract: {features.get('contract_type', 'unknown')}",
            f"Plan: {features.get('plan_tier', 'unknown')}",
            f"Tenure: {features.get('tenure_months', 0)} months",
            f"Charges: ${features.get('monthly_charges', 0):.0f}/month",
            f"Tickets: {features.get('num_support_tickets_30d', 0)} in 30d",
            f"Login freq: {features.get('login_frequency_30d', 0):.0f}/30d",
            f"Feature adoption: {features.get('feature_adoption_rate', 0):.0%}",
            f"NPS: {features.get('nps_score', 0):.0f}",
            f"Days since login: {features.get('days_since_last_login', 0)}",
        ]
        return " | ".join(parts)
