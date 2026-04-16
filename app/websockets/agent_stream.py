"""
WebSocket — Agent Pipeline Event Stream
==========================================
Real-time streaming of LangGraph pipeline progress events to the browser.

The pipeline writes events to Redis via:
    redis_store.append_stream_event(run_id, event_dict)

This WebSocket handler polls Redis at 1-second intervals, forwards new
events to the connected client, and closes when the pipeline reaches a
terminal state.

Connection:
    ws://localhost:8000/ws/agent/{run_id}

Event shape (sent to client):
    {"step": "prediction", "status": "running", "message": "..."}
    {"step": "pipeline", "status": "final", "prediction": {...}, ...}
"""

from __future__ import annotations

import asyncio

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["websocket"])

# Status values that mean the pipeline has finished
_TERMINAL_PREFIXES = (
    "complete",
    "low_risk",
    "crashed",
    "aborted",
    "not_found",
)

# Max polling iterations (1s each) — 5 min timeout
_MAX_POLLS = 300


@router.websocket("/ws/agent/{run_id}")
async def stream_pipeline_events(websocket: WebSocket, run_id: str) -> None:
    """
    Stream Redis pipeline events to the connected WebSocket client.

    Closes automatically when:
      - Pipeline reaches a terminal state (complete / crashed / aborted)
      - Max polling timeout reached (5 min)
      - Client disconnects
    """
    await websocket.accept()
    logger.info("WebSocket connected", run_id=run_id)

    from memory.redis_state import RedisStateManager
    redis = RedisStateManager()

    try:
        for _ in range(_MAX_POLLS):
            # Drain any new events
            events = redis.pop_stream_events(run_id, max_events=50)
            for event in events:
                await websocket.send_json(event)

            # Check terminal state
            status = redis.get_status(run_id)
            if any(status.startswith(t) for t in _TERMINAL_PREFIXES):
                # Send consolidated final payload
                state = redis.load_state(run_id)
                if state:
                    explanation = state.get("explanation") or {}
                    await websocket.send_json({
                        "step": "pipeline",
                        "status": "final",
                        "prediction": state.get("prediction"),
                        "explanation": {
                            "narrative_text": explanation.get("narrative_text"),
                            "top_risk_factors": explanation.get("top_risk_factors", []),
                        },
                        "retention_plan": state.get("retention_plan"),
                        "hitl_decision": state.get("hitl_decision"),
                        "completed_steps": state.get("completed_steps", []),
                        "errors": state.get("errors", []),
                    })
                break

            await asyncio.sleep(1.0)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected", run_id=run_id)
    except Exception as exc:
        logger.error("WebSocket error", run_id=run_id, error=str(exc))
        try:
            await websocket.send_json({"step": "error", "status": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info("WebSocket closed", run_id=run_id)
