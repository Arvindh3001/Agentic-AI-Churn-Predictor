"""
Prediction Agent
=================
Runs the production churn ensemble model on a validated customer feature
vector and returns a structured prediction result with:
  - Churn probability
  - Conformal confidence interval
  - Risk tier classification (LOW / MEDIUM / HIGH / CRITICAL)
  - Base learner breakdown (per-model probabilities for transparency)
  - Model version / name for audit trail
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from agents.state import AgentState, PredictionResult
from agents.tools.model_tool import churn_prediction_tool
from memory.redis_state import RedisStateManager

logger = structlog.get_logger(__name__)

# Risk tier thresholds (also defined in tools/model_tool.py — kept in sync)
RISK_THRESHOLDS = {
    "CRITICAL": 0.85,
    "HIGH": 0.70,
    "MEDIUM": 0.40,
    "LOW": 0.0,
}


class PredictionAgent:
    """
    Invokes the ML ensemble and returns a typed PredictionResult.

    Args:
        redis_store: Shared Redis state manager for streaming events.
    """

    def __init__(self, redis_store: RedisStateManager | None = None) -> None:
        self._redis = redis_store or RedisStateManager()

    def run(self, state: AgentState) -> dict[str, Any]:
        """
        LangGraph node function. Expects state to contain customer_features.
        Returns partial state update with 'prediction' key.
        """
        t0 = time.time()
        customer_id = state["customer_id"]
        features = state.get("customer_features")

        if not features:
            err = "PredictionAgent: no customer_features in state"
            logger.error(err, customer_id=customer_id)
            return {
                "current_step": "prediction",
                "completed_steps": state["completed_steps"],
                "errors": state["errors"] + [err],
                "should_abort": True,
            }

        logger.info("PredictionAgent started", customer_id=customer_id)

        # Stream progress event
        self._redis.append_stream_event(state["run_id"], {
            "step": "prediction",
            "status": "running",
            "message": "Running ensemble model...",
        })

        # Run prediction via tool
        raw_result = churn_prediction_tool.invoke({"customer_features": features})

        if "error" in raw_result:
            err = f"Prediction failed: {raw_result['error']}"
            logger.error(err)
            return {
                "current_step": "prediction",
                "completed_steps": state["completed_steps"],
                "errors": state["errors"] + [err],
                "should_abort": True,
            }

        prediction: PredictionResult = {
            "churn_probability": raw_result["churn_probability"],
            "confidence_interval": raw_result["confidence_interval"],
            "risk_tier": raw_result["risk_tier"],
            "model_version": raw_result["model_version"],
            "model_name": raw_result["model_name"],
            "is_uncertain": raw_result["is_uncertain"],
        }

        duration = round(time.time() - t0, 3)

        self._redis.append_stream_event(state["run_id"], {
            "step": "prediction",
            "status": "completed",
            "churn_probability": prediction["churn_probability"],
            "risk_tier": prediction["risk_tier"],
            "confidence_interval": prediction["confidence_interval"],
            "duration_s": duration,
        })

        logger.info(
            "PredictionAgent completed",
            customer_id=customer_id,
            churn_prob=prediction["churn_probability"],
            risk_tier=prediction["risk_tier"],
            duration_s=duration,
        )

        return {
            "prediction": prediction,
            "current_step": "prediction_done",
            "completed_steps": state["completed_steps"] + ["prediction"],
            "errors": state["errors"],
            "step_durations": {
                **state.get("step_durations", {}),
                "prediction": duration,
            },
        }

    @staticmethod
    def should_explain(state: AgentState) -> str:
        """
        LangGraph conditional edge: decide next step based on risk tier.
        Routes HIGH/CRITICAL to explanation, skips for LOW risk.
        """
        prediction = state.get("prediction", {})
        risk_tier = prediction.get("risk_tier", "LOW")

        if state.get("should_abort"):
            return "abort"
        if risk_tier in ("HIGH", "CRITICAL", "MEDIUM"):
            return "explain"
        return "skip_explanation"
