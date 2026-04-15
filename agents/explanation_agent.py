"""
Explanation Agent
==================
Generates a multi-layer explanation for a churn prediction:
  1. SHAP values (feature contributions) via shap_tool
  2. LLM-powered narrative (OpenAI GPT-4o / Anthropic Claude / template)

The agent uses a chain-of-thought prompt to ground the narrative in the
actual SHAP evidence — preventing hallucinations about why a customer
is at risk.

LangSmith traces every LLM call for full observability.
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from agents.state import AgentState, ExplanationResult
from agents.tools.shap_tool import shap_explanation_tool
from config.settings import settings
from memory.redis_state import RedisStateManager
from memory.vector_store import VectorStore
from src.explainability.narrative_generator import NarrativeGenerator

logger = structlog.get_logger(__name__)


class ExplanationAgent:
    """
    Computes SHAP explanations and generates LLM narratives.

    Args:
        redis_store: Shared Redis state manager.
        vector_store: For storing and retrieving past explanations.
        llm_provider: 'openai', 'anthropic', or None (template mode).
    """

    def __init__(
        self,
        redis_store: RedisStateManager | None = None,
        vector_store: VectorStore | None = None,
        llm_provider: str | None = None,
    ) -> None:
        self._redis = redis_store or RedisStateManager()
        self._vector_store = vector_store or VectorStore()

        # Use LLM if configured and key is available
        provider = llm_provider or settings.llm.default_llm_provider
        has_key = bool(
            (provider == "openai" and settings.llm.openai_api_key) or
            (provider == "anthropic" and settings.llm.anthropic_api_key) or
            (provider == "google" and settings.llm.google_api_key)
        )
        effective_provider = provider if has_key else None

        self._narrator = NarrativeGenerator(
            llm_provider=effective_provider,  # type: ignore[arg-type]
            model=settings.llm.default_llm_model,
        )
        self._llm_mode = effective_provider is not None
        logger.info(
            "ExplanationAgent initialised",
            llm_mode=self._llm_mode,
            provider=effective_provider or "template",
        )

    def run(self, state: AgentState) -> dict[str, Any]:
        """
        LangGraph node function. Expects customer_features and prediction in state.
        Returns partial state update with 'explanation' key.
        """
        t0 = time.time()
        customer_id = state["customer_id"]
        features = state.get("customer_features", {})
        prediction = state.get("prediction", {})

        logger.info("ExplanationAgent started", customer_id=customer_id)

        self._redis.append_stream_event(state["run_id"], {
            "step": "explanation",
            "status": "running",
            "message": "Computing SHAP explanations...",
        })

        # 1. SHAP values
        shap_result = shap_explanation_tool.invoke({"customer_features": features})

        if "error" in shap_result:
            logger.warning("SHAP failed, using minimal explanation", error=shap_result["error"])
            shap_result = {
                "shap_contributions": {},
                "top_positive_drivers": [],
                "top_negative_drivers": [],
                "top_risk_factors": [],
                "churn_probability": prediction.get("churn_probability", 0.5),
                "base_value": 0.27,
            }

        # 2. Retrieve similar past cases from ChromaDB for LLM context
        similar_cases = self._vector_store.search_explanations(
            query=f"customer churn risk {prediction.get('risk_tier', '')} "
                  f"contract {features.get('contract_type', '')}",
            top_k=2,
        )

        # 3. Build counterfactual context (minimal, without calling CF engine)
        cf_context = self._build_cf_stub(features, prediction)

        self._redis.append_stream_event(state["run_id"], {
            "step": "explanation",
            "status": "running",
            "message": "Generating narrative explanation...",
        })

        # 4. Generate narrative
        narrative = self._narrator.generate(
            shap_result=shap_result,
            customer_context={
                "customer_id": customer_id,
                "plan_tier": features.get("plan_tier"),
                "tenure_months": features.get("tenure_months"),
                "contract_type": features.get("contract_type"),
                "similar_cases_count": len(similar_cases),
            },
        )

        explanation: ExplanationResult = {
            "shap_contributions": shap_result.get("shap_contributions", {}),
            "lime_weights": {},  # LIME is called optionally in Phase 2 benchmarking
            "shap_lime_agreement": 0.0,
            "top_risk_factors": shap_result.get("top_risk_factors", []),
            "narrative_text": narrative,
            "plot_paths": {},
        }

        duration = round(time.time() - t0, 3)

        # 5. Persist explanation to ChromaDB for future retrieval
        self._vector_store.store_explanation(
            customer_id=customer_id,
            run_id=state["run_id"],
            narrative=narrative,
            metadata={
                "churn_prob": prediction.get("churn_probability", 0),
                "risk_tier": prediction.get("risk_tier", "UNKNOWN"),
                "llm_mode": self._llm_mode,
            },
        )

        self._redis.append_stream_event(state["run_id"], {
            "step": "explanation",
            "status": "completed",
            "duration_s": duration,
            "top_driver": (explanation["top_risk_factors"][0]["feature"]
                           if explanation["top_risk_factors"] else "n/a"),
        })

        logger.info(
            "ExplanationAgent completed",
            customer_id=customer_id,
            llm_mode=self._llm_mode,
            duration_s=duration,
        )

        return {
            "explanation": explanation,
            "current_step": "explanation_done",
            "completed_steps": state["completed_steps"] + ["explanation"],
            "errors": state["errors"],
            "step_durations": {
                **state.get("step_durations", {}),
                "explanation": duration,
            },
        }

    @staticmethod
    def _build_cf_stub(
        features: dict[str, Any],
        prediction: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Build a minimal counterfactual context hint for the narrative LLM.
        Full DiCE counterfactuals are generated in Phase 4 (CounterfactualAgent).
        """
        hints: list[str] = []
        prob = prediction.get("churn_probability", 0)

        if features.get("monthly_charges", 0) > 80:
            hints.append("a price reduction could meaningfully lower churn risk")
        if features.get("feature_adoption_rate", 1) < 0.4:
            hints.append("improving feature adoption through onboarding support may help")
        if features.get("contract_type") == "Month-to-Month":
            hints.append("upgrading to an annual contract would reduce churn risk")

        return {
            "customer_id": features.get("customer_id", "unknown"),
            "current_churn_prob": round(prob, 4),
            "intervention_hints": hints,
            "interventions": [],  # populated in Phase 4
        }
