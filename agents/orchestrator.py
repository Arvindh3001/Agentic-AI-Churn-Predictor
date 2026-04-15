"""
LangGraph Orchestrator — Churn Intelligence Pipeline
======================================================
Defines the StateGraph that wires together all Phase 3 agents into
a directed execution graph with conditional routing.

Graph topology:

    [START]
       │
       ▼
  data_intelligence ──(abort?)──► [END:error]
       │
       ▼
   prediction ────(abort?)──────► [END:error]
       │
       ├─(LOW risk)──────────────► [END:low_risk]
       │
       └─(MEDIUM/HIGH/CRITICAL)──►
       │
       ▼
   explanation
       │
       ▼
    [END:complete]

Async execution:
  - `run_async(customer_id)` submits a Celery task and returns a task_id
  - `run_sync(customer_id)` runs inline (dev/testing)
  - Results streamed to Redis in real-time

LangSmith tracing is enabled via LANGCHAIN_TRACING_V2 env var.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any

import structlog
from langgraph.graph import END, StateGraph

from agents.counterfactual_agent import CounterfactualAgent
from agents.data_intelligence_agent import DataIntelligenceAgent
from agents.explanation_agent import ExplanationAgent
from agents.prediction_agent import PredictionAgent
from agents.retention_strategist import RetentionStrategistAgent
from agents.state import AgentState, initial_state
from config.settings import settings
from memory.redis_state import RedisStateManager
from memory.vector_store import VectorStore

logger = structlog.get_logger(__name__)

# Configure LangSmith tracing
if settings.llm.langchain_tracing_v2:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.llm.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.llm.langchain_project
    os.environ["LANGCHAIN_ENDPOINT"] = settings.llm.langchain_endpoint


# ------------------------------------------------------------------ #
# Graph node labels
# ------------------------------------------------------------------ #
NODE_DATA_INTEL = "data_intelligence"
NODE_PREDICTION = "prediction"
NODE_EXPLANATION = "explanation"
NODE_COUNTERFACTUAL = "counterfactual"
NODE_RETENTION = "retention_strategist"
NODE_ABORT = "abort"


def _make_abort_node(reason: str):
    """Factory: returns a terminal error node function."""
    def abort_fn(state: AgentState) -> dict[str, Any]:
        logger.error("Pipeline aborted", reason=reason, customer_id=state["customer_id"])
        return {
            "current_step": f"aborted:{reason}",
            "completed_steps": state["completed_steps"],
            "errors": state["errors"],
            "should_abort": True,
        }
    return abort_fn


def _low_risk_terminal(state: AgentState) -> dict[str, Any]:
    """Terminal node for LOW risk customers — skip expensive explanation."""
    logger.info(
        "Low-risk customer — skipping full explanation",
        customer_id=state["customer_id"],
        churn_prob=state.get("prediction", {}).get("churn_probability"),
    )
    return {
        "current_step": "complete:low_risk",
        "completed_steps": state["completed_steps"] + ["low_risk_terminal"],
        "errors": state["errors"],
    }


# ------------------------------------------------------------------ #
# Routing conditions
# ------------------------------------------------------------------ #

def _route_after_data_intel(state: AgentState) -> str:
    if state.get("should_abort"):
        return "abort"
    return NODE_PREDICTION


def _route_after_prediction(state: AgentState) -> str:
    if state.get("should_abort"):
        return "abort"
    return PredictionAgent.should_explain(state)


# ------------------------------------------------------------------ #
# Graph builder
# ------------------------------------------------------------------ #

def build_graph(
    redis_store: RedisStateManager | None = None,
    vector_store: VectorStore | None = None,
    llm_provider: str | None = None,
    budget_usd: float = 300.0,
) -> Any:
    """
    Construct and compile the LangGraph StateGraph.

    Phase 4 graph topology:

        [START]
           │
           ▼
      data_intelligence ──(abort?)──► [END:error]
           │
           ▼
        prediction ────(abort?)──────► [END:error]
           │
           ├─(LOW risk)──────────────► [END:low_risk]
           │
           └─(MEDIUM/HIGH/CRITICAL)──►
           │
           ▼
        explanation
           │
           ▼
        counterfactual
           │
           ▼
        retention_strategist
           │
           ▼
         [END:complete]

    Args:
        redis_store:  Shared Redis state manager.
        vector_store: Shared ChromaDB vector store.
        llm_provider: LLM to use for explanation narratives.
        budget_usd:   Per-customer retention budget for the knapsack solver.

    Returns:
        Compiled LangGraph app (callable).
    """
    _redis = redis_store or RedisStateManager()
    _vector = vector_store or VectorStore()

    # Instantiate agents
    data_agent = DataIntelligenceAgent(redis_store=_redis, vector_store=_vector)
    pred_agent = PredictionAgent(redis_store=_redis)
    expl_agent = ExplanationAgent(
        redis_store=_redis,
        vector_store=_vector,
        llm_provider=llm_provider,
    )
    cf_agent = CounterfactualAgent(redis_store=_redis)
    ret_agent = RetentionStrategistAgent(redis_store=_redis, budget_usd=budget_usd)

    # Build graph
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node(NODE_DATA_INTEL, data_agent.run)
    graph.add_node(NODE_PREDICTION, pred_agent.run)
    graph.add_node(NODE_EXPLANATION, expl_agent.run)
    graph.add_node(NODE_COUNTERFACTUAL, cf_agent.run)
    graph.add_node(NODE_RETENTION, ret_agent.run)
    graph.add_node("abort", _make_abort_node("pipeline_error"))
    graph.add_node("low_risk_terminal", _low_risk_terminal)

    # Entry point
    graph.set_entry_point(NODE_DATA_INTEL)

    # Edges
    graph.add_conditional_edges(
        NODE_DATA_INTEL,
        _route_after_data_intel,
        {
            "abort": "abort",
            NODE_PREDICTION: NODE_PREDICTION,
        },
    )

    graph.add_conditional_edges(
        NODE_PREDICTION,
        _route_after_prediction,
        {
            "abort": "abort",
            "explain": NODE_EXPLANATION,
            "skip_explanation": "low_risk_terminal",
        },
    )

    # After explanation → counterfactual → retention strategist → done
    graph.add_edge(NODE_EXPLANATION, NODE_COUNTERFACTUAL)
    graph.add_edge(NODE_COUNTERFACTUAL, NODE_RETENTION)
    graph.add_edge(NODE_RETENTION, END)
    graph.add_edge("abort", END)
    graph.add_edge("low_risk_terminal", END)

    app = graph.compile()
    logger.info("LangGraph pipeline compiled", nodes=list(graph.nodes))
    return app


# ------------------------------------------------------------------ #
# Pipeline runner
# ------------------------------------------------------------------ #

class ChurnOrchestrator:
    """
    High-level interface for running the churn analysis pipeline.

    Usage:
        orchestrator = ChurnOrchestrator()
        result = orchestrator.run_sync("C-4821")
        task_id = orchestrator.run_async("C-4821")  # Celery
    """

    def __init__(
        self,
        llm_provider: str | None = None,
    ) -> None:
        self._redis = RedisStateManager()
        self._vector = VectorStore()
        self._app = build_graph(
            redis_store=self._redis,
            vector_store=self._vector,
            llm_provider=llm_provider,
        )

    def run_sync(
        self,
        customer_id: str,
        triggered_by: str = "api",
        run_id: str | None = None,
    ) -> AgentState:
        """
        Run the full pipeline synchronously.

        Args:
            customer_id: Customer to analyse.
            triggered_by: Source of the request for audit logging.
            run_id: Explicit run ID to use for Redis state keys. When called
                    from a Celery task, pass ``self.request.id`` so that
                    callers polling ``/agent/status/{task_id}`` resolve to
                    the correct Redis key.  Defaults to a random short UUID.

        Returns:
            Final AgentState after all nodes have executed.
        """
        run_id = run_id or str(uuid.uuid4())[:8]
        t0 = time.time()

        state = initial_state(
            customer_id=customer_id,
            run_id=run_id,
            triggered_by=triggered_by,
        )
        state["start_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # Save initial state to Redis for polling
        self._redis.save_state(run_id, dict(state))
        self._redis.append_stream_event(run_id, {
            "step": "pipeline",
            "status": "started",
            "customer_id": customer_id,
        })

        logger.info("Pipeline started", customer_id=customer_id, run_id=run_id)

        try:
            final_state = self._app.invoke(state)
        except Exception as exc:
            logger.error("Pipeline crashed", error=str(exc), run_id=run_id)
            final_state = {
                **state,
                "errors": state["errors"] + [str(exc)],
                "current_step": "crashed",
                "should_abort": True,
            }

        total_duration = round(time.time() - t0, 3)
        final_state["step_durations"] = {
            **final_state.get("step_durations", {}),
            "total": total_duration,
        }

        # Persist final state
        self._redis.save_state(run_id, dict(final_state))
        self._redis.append_stream_event(run_id, {
            "step": "pipeline",
            "status": "complete" if not final_state.get("should_abort") else "error",
            "total_duration_s": total_duration,
            "completed_steps": final_state.get("completed_steps", []),
        })

        # Store interaction in ChromaDB memory
        self._persist_interaction(customer_id, run_id, final_state)

        logger.info(
            "Pipeline complete",
            customer_id=customer_id,
            run_id=run_id,
            steps=final_state.get("completed_steps"),
            duration_s=total_duration,
        )

        return final_state

    def run_async(self, customer_id: str, triggered_by: str = "api") -> str:
        """
        Submit pipeline as a Celery background task.

        Returns:
            task_id string — use to poll status via /agent/status/{task_id}
        """
        from celery_app import celery_app  # imported lazily to avoid circular deps

        task = celery_app.send_task(
            "tasks.run_churn_pipeline",
            args=[customer_id],
            kwargs={"triggered_by": triggered_by},
        )
        logger.info("Pipeline submitted as Celery task", customer_id=customer_id, task_id=task.id)
        return task.id

    def get_status(self, run_id: str) -> dict[str, Any]:
        """Poll current status of a running or completed pipeline."""
        status = self._redis.get_status(run_id)
        events = self._redis.get_all_stream_events(run_id)
        state = self._redis.load_state(run_id)

        return {
            "run_id": run_id,
            "status": status,
            "stream_events": events,
            "result": {
                "prediction": state.get("prediction") if state else None,
                "explanation": {
                    "narrative_text": state.get("explanation", {}).get("narrative_text"),
                    "top_risk_factors": state.get("explanation", {}).get("top_risk_factors"),
                } if state and state.get("explanation") else None,
                "completed_steps": state.get("completed_steps", []) if state else [],
                "errors": state.get("errors", []) if state else [],
            },
        }

    def _persist_interaction(
        self,
        customer_id: str,
        run_id: str,
        state: dict[str, Any],
    ) -> None:
        """Store final pipeline result in ChromaDB interaction memory."""
        prediction = state.get("prediction", {})
        explanation = state.get("explanation", {})
        self._vector.upsert_customer_interaction(
            customer_id=customer_id,
            interaction={
                "customer_id": customer_id,
                "run_id": run_id,
                "timestamp": state.get("start_time", ""),
                "churn_prob": prediction.get("churn_probability", 0),
                "risk_tier": prediction.get("risk_tier", "UNKNOWN"),
                "top_risk_factors": explanation.get("top_risk_factors", []),
                "action_taken": "pending_hitl",
                "outcome": "pending",
            },
        )


# ------------------------------------------------------------------ #
# CLI entry point — run a single customer analysis
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Run churn pipeline for a customer")
    parser.add_argument("--customer-id", default="demo-customer-001")
    parser.add_argument("--llm", default=None, help="openai | anthropic | None")
    args = parser.parse_args()

    orchestrator = ChurnOrchestrator(llm_provider=args.llm)
    result = orchestrator.run_sync(args.customer_id)

    print("\n" + "=" * 60)
    print("  CHURN ANALYSIS RESULT")
    print("=" * 60)
    if result.get("prediction"):
        p = result["prediction"]
        print(f"  Customer:     {result['customer_id']}")
        print(f"  Churn Prob:   {p['churn_probability']:.0%}")
        print(f"  Risk Tier:    {p['risk_tier']}")
        print(f"  CI:           {p['confidence_interval']}")
    if result.get("explanation"):
        e = result["explanation"]
        print(f"\n  Narrative:\n  {e['narrative_text']}")
        print(f"\n  Top Risk Factors:")
        for f in e["top_risk_factors"][:3]:
            print(f"    • {f.get('label', f.get('feature'))}: {f.get('shap_value', 0):+.3f}")
    print(f"\n  Steps: {result.get('completed_steps')}")
    print(f"  Errors: {result.get('errors')}")
    print("=" * 60)
