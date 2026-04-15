"""
Agent State Schema
==================
The single shared state TypedDict that flows through every node
in the LangGraph StateGraph. All agents read from and write to this state.

Design principle: immutable-by-convention — each agent returns a *partial*
dict with only the keys it modified; LangGraph merges it into the full state.
"""

from __future__ import annotations

from typing import Any, Literal
from typing_extensions import TypedDict, NotRequired


RiskTier = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]


class DataQualityReport(TypedDict):
    passed: bool
    missing_pct: float
    anomaly_features: list[str]
    drift_detected: bool
    warnings: list[str]


class PredictionResult(TypedDict):
    churn_probability: float
    confidence_interval: list[float]   # [lower, upper]
    risk_tier: RiskTier
    model_version: str
    model_name: str
    is_uncertain: bool


class ExplanationResult(TypedDict):
    shap_contributions: dict[str, float]    # feature → SHAP value
    lime_weights: dict[str, float]           # condition → LIME weight
    shap_lime_agreement: float               # 0–1 agreement score
    top_risk_factors: list[dict[str, Any]]  # top-5 drivers with labels
    narrative_text: str                      # LLM or template narrative
    plot_paths: dict[str, str]              # saved chart file paths


class CounterfactualResult(TypedDict):
    current_churn_prob: float
    interventions: list[dict[str, Any]]   # ranked, constraint-valid interventions
    n_feasible: int


class RetentionPlan(TypedDict):
    selected_actions: list[dict[str, Any]]  # knapsack-selected interventions
    total_cost_usd: float
    estimated_revenue_saved_usd: float
    estimated_roi: float
    ab_group: str                           # "treatment" | "control"
    crm_action_id: str                      # action_id from CRM executor
    confidence: float                       # probability the plan reduces churn


class AgentState(TypedDict):
    # ---- Input ----
    customer_id: str
    run_id: str                              # unique per pipeline execution
    triggered_by: str                        # "api" | "celery" | "batch"

    # ---- Data Intelligence Agent output ----
    customer_features: NotRequired[dict[str, Any]]
    raw_feature_vector: NotRequired[list[float]]   # processed array for models
    data_quality: NotRequired[DataQualityReport]
    similar_customers: NotRequired[list[dict[str, Any]]]  # from ChromaDB

    # ---- Prediction Agent output ----
    prediction: NotRequired[PredictionResult]

    # ---- Explanation Agent output ----
    explanation: NotRequired[ExplanationResult]

    # ---- Counterfactual Agent output (Phase 4) ----
    counterfactuals: NotRequired[CounterfactualResult]

    # ---- Retention Strategist Agent output (Phase 4) ----
    retention_plan: NotRequired[RetentionPlan]

    # ---- Control flow ----
    current_step: str
    completed_steps: list[str]
    errors: list[str]
    should_abort: bool                       # set True on unrecoverable error

    # ---- Metadata ----
    start_time: NotRequired[str]             # ISO timestamp
    step_durations: NotRequired[dict[str, float]]  # step → seconds
    llm_tokens_used: NotRequired[int]


def initial_state(
    customer_id: str,
    run_id: str,
    triggered_by: str = "api",
) -> AgentState:
    """Return a blank starting state for a new pipeline run."""
    return AgentState(
        customer_id=customer_id,
        run_id=run_id,
        triggered_by=triggered_by,
        current_step="init",
        completed_steps=[],
        errors=[],
        should_abort=False,
    )
