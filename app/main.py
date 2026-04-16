"""
FastAPI Application — Churn Intelligence Platform
===================================================
Production-grade REST API + WebSocket server.

Start with:
    uvicorn app.main:app --reload --port 8000

API docs (auto-generated):
    http://localhost:8000/docs    (Swagger UI)
    http://localhost:8000/redoc   (ReDoc)

Endpoints summary:
    POST  /auth/token                    → JWT access token
    GET   /auth/verify                   → validate current token
    GET   /api/v1/customers              → paginated customer list
    GET   /api/v1/customers/high-risk    → at-risk watchlist
    GET   /api/v1/customers/{id}         → customer profile + last analysis
    POST  /api/v1/agent/analyse          → trigger pipeline (sync or async)
    GET   /api/v1/agent/status/{run_id}  → poll pipeline progress
    POST  /api/v1/agent/batch            → batch analysis via Celery
    GET   /api/v1/agent/batch/{task_id}  → batch task progress
    WS    /ws/agent/{run_id}             → real-time pipeline event stream
    POST  /hitl/decision                 → HITL approve / reject
    POST  /hitl/slack/interactive        → Slack button callback
    POST  /hitl/feedback                 → CSM outcome feedback
    GET   /hitl/status/{run_id}          → HITL decision status
    GET   /hitl/feedback/stats           → feedback / retrain stats
    GET   /hitl/audit                    → recent audit log
    GET   /health                        → service health check
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.hitl_webhook import router as hitl_router
from app.routers import agent, auth, customers, analytics
from app.websockets.agent_stream import router as ws_router
from app.websockets.customer_broadcast import router as customer_ws_router
from config.settings import settings

logger = structlog.get_logger(__name__)

_start_time = time.time()


# ------------------------------------------------------------------ #
# Application lifespan
# ------------------------------------------------------------------ #

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm model cache on startup so first request is fast."""
    logger.info("API starting up", env=settings.app_env, port=settings.app_port)
    try:
        from agents.tools.model_tool import _load_model, _load_pipeline
        model = _load_model()
        pipeline = _load_pipeline()
        if model and pipeline:
            logger.info("Model + pipeline cache warmed on startup")
        else:
            logger.warning("Model or pipeline not found — predictions will fail")
    except Exception as exc:
        logger.warning("Startup model pre-warm failed", error=str(exc))
    yield
    logger.info("API shutting down")


# ------------------------------------------------------------------ #
# App factory
# ------------------------------------------------------------------ #

app = FastAPI(
    title="Churn Intelligence Platform",
    description=(
        "Agentic AI customer churn prediction and retention platform. "
        "6-agent LangGraph pipeline: data intelligence → prediction → "
        "explanation → counterfactual → retention strategy → HITL review."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ---- Middleware --------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# ---- Routers ----------------------------------------------------------

app.include_router(auth.router)
app.include_router(customers.router)
app.include_router(agent.router)
app.include_router(analytics.router)
app.include_router(hitl_router)
app.include_router(ws_router)
app.include_router(customer_ws_router)


# ---- System endpoints -------------------------------------------------

@app.get("/health", tags=["system"])
async def health_check() -> dict:
    """Service health check — used by load balancers and monitoring."""
    from memory.redis_state import RedisStateManager
    redis_ok = RedisStateManager().is_connected
    return {
        "status": "healthy",
        "version": "1.0.0",
        "uptime_seconds": round(time.time() - _start_time, 1),
        "redis": "connected" if redis_ok else "unavailable",
        "env": settings.app_env,
    }


@app.get("/", tags=["system"])
async def root() -> dict:
    return {
        "service": "Churn Intelligence Platform API",
        "version": "1.0.0",
        "docs": "/docs",
    }
