"""
Retention Strategist Agent
===========================
LangGraph node that selects the optimal subset of retention interventions
within a per-customer budget, executes the chosen action in the CRM, and
assigns the customer to an A/B experiment group.

Responsibilities:
  1. Estimate customer lifetime value (CLV) from features
  2. Build a value-annotated item list from counterfactual interventions
  3. Run the KnapsackSolver to select the budget-optimal action set
  4. Assign the customer to treatment / control (A/B framework)
  5. Execute the selected action via crm_executor_tool
  6. Return a RetentionPlan to the pipeline state

Graph position:
    counterfactual → retention_strategist → END
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from agents.state import AgentState, RetentionPlan
from agents.tools.crm_executor_tool import crm_executor_tool
from memory.redis_state import RedisStateManager
from src.optimization.ab_testing import ABTestingManager
from src.optimization.knapsack_solver import KnapsackSolver

logger = structlog.get_logger(__name__)

# Default per-customer retention budget (USD)
DEFAULT_BUDGET_USD: float = 300.0
# Experiment ID used for A/B assignment
EXPERIMENT_ID: str = "retention_v1"
# Months of retained CLV we expect a successful intervention to recover
RETENTION_MONTHS: float = 12.0


class RetentionStrategistAgent:
    """
    Selects and executes the optimal retention plan for at-risk customers.

    Args:
        redis_store:    Shared Redis state manager.
        budget_usd:     Maximum per-customer spend on retention actions.
        experiment_id:  A/B experiment name for group assignment.
    """

    def __init__(
        self,
        redis_store: RedisStateManager | None = None,
        budget_usd: float = DEFAULT_BUDGET_USD,
        experiment_id: str = EXPERIMENT_ID,
    ) -> None:
        self._redis = redis_store or RedisStateManager()
        self._budget = budget_usd
        self._experiment_id = experiment_id
        self._ab = ABTestingManager()
        self._solver = KnapsackSolver(budget_usd=budget_usd)

    def run(self, state: AgentState) -> dict[str, Any]:
        """
        LangGraph node function.

        Expects state to contain:
            customer_features   (from DataIntelligenceAgent)
            prediction          (from PredictionAgent)
            counterfactuals     (from CounterfactualAgent)

        Returns partial state update with 'retention_plan' key.
        """
        t0 = time.time()
        customer_id = state["customer_id"]
        features = state.get("customer_features", {})
        prediction = state.get("prediction", {})
        cf_result = state.get("counterfactuals", {})

        logger.info("RetentionStrategistAgent started", customer_id=customer_id)

        self._redis.append_stream_event(state["run_id"], {
            "step": "retention_strategist",
            "status": "running",
            "message": "Optimising retention action plan...",
        })

        # --- A/B group assignment -------------------------------------------
        ab_group = self._ab.assign_group(customer_id, self._experiment_id)

        # --- Estimate CLV -------------------------------------------------------
        clv = self._estimate_clv(features)

        # --- Build knapsack items from interventions ----------------------------
        interventions = cf_result.get("interventions", [])
        items = self._build_items(interventions, clv)

        # --- Solve knapsack ---------------------------------------------------
        if items:
            knapsack_result = self._solver.solve(items)
            selected = knapsack_result["selected"]
            total_cost = knapsack_result["total_cost"]
            total_value = knapsack_result["total_value"]
        else:
            # No feasible interventions — produce a zero-cost holding plan
            selected = []
            total_cost = 0.0
            total_value = 0.0
            logger.warning("No feasible interventions for knapsack", customer_id=customer_id)

        estimated_roi = round((total_value - total_cost) / max(total_cost, 1.0), 4)

        # --- Execute CRM action ------------------------------------------------
        crm_action_id = ""
        if selected:
            best_action = selected[0]
            crm_result = crm_executor_tool.invoke({
                "customer_id": customer_id,
                "action_type": _action_type_from_label(best_action.get("label", "")),
                "action_description": best_action.get("label", best_action.get("action", "Retention action")),
                "estimated_cost_usd": best_action.get("cost_usd", 0.0),
                "run_id": state["run_id"],
                "ab_group": ab_group,
            })
            crm_action_id = crm_result.get("action_id", "")

        # --- Estimate confidence -----------------------------------------------
        churn_prob = prediction.get("churn_probability", 0.5)
        confidence = self._estimate_confidence(selected, churn_prob)

        duration = round(time.time() - t0, 3)

        plan: RetentionPlan = {
            "selected_actions": selected,
            "total_cost_usd": total_cost,
            "estimated_revenue_saved_usd": round(total_value, 2),
            "estimated_roi": estimated_roi,
            "ab_group": ab_group,
            "crm_action_id": crm_action_id,
            "confidence": confidence,
        }

        self._redis.append_stream_event(state["run_id"], {
            "step": "retention_strategist",
            "status": "completed",
            "ab_group": ab_group,
            "n_actions": len(selected),
            "total_cost_usd": total_cost,
            "estimated_roi": estimated_roi,
            "duration_s": duration,
        })

        logger.info(
            "RetentionStrategistAgent completed",
            customer_id=customer_id,
            ab_group=ab_group,
            n_actions=len(selected),
            roi=estimated_roi,
            duration_s=duration,
        )

        return {
            "retention_plan": plan,
            "current_step": "retention_strategist_done",
            "completed_steps": state["completed_steps"] + ["retention_strategist"],
            "errors": state["errors"],
            "step_durations": {
                **state.get("step_durations", {}),
                "retention_strategist": duration,
            },
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _estimate_clv(features: dict[str, Any]) -> float:
        """
        Approximate annual CLV from monthly charges and expected tenure.

        CLV = monthly_charges × RETENTION_MONTHS × (1 - discount_factor)
        """
        monthly = float(features.get("monthly_charges", 50) or 50)
        return round(monthly * RETENTION_MONTHS, 2)

    @staticmethod
    def _build_items(
        interventions: list[dict[str, Any]],
        clv: float,
    ) -> list[dict[str, Any]]:
        """
        Convert interventions into knapsack items.

        value_usd is the fraction of CLV we expect to recover, scaled by
        the probability reduction each intervention achieves.
        """
        items = []
        for iv in interventions:
            prob_reduction = iv.get("prob_reduction", iv.get("feasibility_score", 0.01))
            value_usd = round(clv * prob_reduction, 2)

            items.append({
                "id": iv.get("action", iv.get("label", "action"))[:40],
                "label": iv.get("action", iv.get("label", "Retention action")),
                "cost_usd": float(iv.get("cost_usd", 50)),
                "value_usd": max(value_usd, 1.0),  # floor at $1 to keep solver happy
                "prob_reduction": prob_reduction,
                "new_churn_prob": iv.get("new_churn_prob", 0.3),
                "effort": iv.get("effort", 2),
                "days_to_effect": iv.get("days_to_effect", 14),
            })
        return items

    @staticmethod
    def _estimate_confidence(
        selected_actions: list[dict[str, Any]],
        churn_probability: float,
    ) -> float:
        """
        Rough confidence that the plan will prevent churn.
        Based on combined probability reduction of selected actions.
        """
        if not selected_actions:
            return 0.0
        total_reduction = sum(a.get("prob_reduction", 0.05) for a in selected_actions)
        remaining_prob = max(churn_probability - total_reduction, 0.05)
        # Confidence = 1 - remaining risk
        return round(min(1.0 - remaining_prob, 0.99), 4)


def _action_type_from_label(label: str) -> str:
    """Map a human-readable action label to a CRM action_type slug."""
    label_lower = label.lower()
    if "price" in label_lower or "discount" in label_lower:
        return "price_reduction"
    if "csm" in label_lower or "onboarding" in label_lower or "support" in label_lower:
        return "csm_assignment"
    if "loyalty" in label_lower or "upgrade" in label_lower or "annual" in label_lower:
        return "loyalty_upgrade"
    if "outreach" in label_lower or "ticket" in label_lower:
        return "support_outreach"
    return "engagement_campaign"
