"""
Counterfactual Agent
=====================
LangGraph node that generates ranked, business-constraint-valid retention
interventions for MEDIUM / HIGH / CRITICAL risk customers.

Responsibilities:
  1. Invoke the counterfactual_tool (DiCE or rule-based fallback)
  2. Ensure ≥ 1 feasible intervention is returned (abort is non-fatal)
  3. Pass ranked interventions to the Retention Strategist

Graph position:
    explanation → counterfactual → retention_strategist → END
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from agents.state import AgentState, CounterfactualResult
from agents.tools.counterfactual_tool import counterfactual_tool
from memory.redis_state import RedisStateManager

logger = structlog.get_logger(__name__)


class CounterfactualAgent:
    """
    Generates counterfactual interventions for at-risk customers.

    Args:
        redis_store: Shared Redis state manager for streaming events.
        top_k:       Maximum number of interventions to generate and rank.
    """

    def __init__(
        self,
        redis_store: RedisStateManager | None = None,
        top_k: int = 5,
    ) -> None:
        self._redis = redis_store or RedisStateManager()
        self.top_k = top_k

    def run(self, state: AgentState) -> dict[str, Any]:
        """
        LangGraph node function.

        Expects state to contain:
            customer_features   (from DataIntelligenceAgent)
            prediction          (from PredictionAgent)

        Returns partial state update with 'counterfactuals' key.
        """
        t0 = time.time()
        customer_id = state["customer_id"]
        features = state.get("customer_features", {})
        prediction = state.get("prediction", {})
        current_prob = prediction.get("churn_probability", 0.5)

        logger.info(
            "CounterfactualAgent started",
            customer_id=customer_id,
            churn_prob=round(current_prob, 3),
        )

        self._redis.append_stream_event(state["run_id"], {
            "step": "counterfactual",
            "status": "running",
            "message": "Generating retention interventions...",
        })

        # Call the counterfactual tool
        raw = counterfactual_tool.invoke({
            "customer_features": features,
            "current_churn_prob": current_prob,
            "top_k": self.top_k,
        })

        duration = round(time.time() - t0, 3)

        if "error" in raw:
            # Non-fatal: log warning and continue with empty interventions
            logger.warning(
                "CounterfactualAgent: tool returned error",
                error=raw["error"],
                customer_id=customer_id,
            )
            result: CounterfactualResult = {
                "current_churn_prob": current_prob,
                "interventions": [],
                "n_feasible": 0,
            }
            self._redis.append_stream_event(state["run_id"], {
                "step": "counterfactual",
                "status": "warning",
                "error": raw["error"],
                "duration_s": duration,
            })
        else:
            result = CounterfactualResult(
                current_churn_prob=raw["current_churn_prob"],
                interventions=raw["interventions"],
                n_feasible=raw["n_feasible"],
            )
            self._redis.append_stream_event(state["run_id"], {
                "step": "counterfactual",
                "status": "completed",
                "n_interventions": result["n_feasible"],
                "duration_s": duration,
            })

        logger.info(
            "CounterfactualAgent completed",
            customer_id=customer_id,
            n_interventions=result["n_feasible"],
            duration_s=duration,
        )

        return {
            "counterfactuals": result,
            "current_step": "counterfactual_done",
            "completed_steps": state["completed_steps"] + ["counterfactual"],
            "errors": state["errors"],
            "step_durations": {
                **state.get("step_durations", {}),
                "counterfactual": duration,
            },
        }
