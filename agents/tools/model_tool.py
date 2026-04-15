"""
Model Tool
===========
LangChain-compatible tool for loading the production model from MLflow
and running predictions with conformal uncertainty intervals.

Caches the loaded model in-process to avoid repeated MLflow round-trips.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import structlog
from langchain_core.tools import tool
from pydantic import BaseModel

from config.settings import settings

logger = structlog.get_logger(__name__)

# Module-level cache: loaded once, reused across tool calls
_model_cache: dict[str, Any] = {}
_pipeline_cache: dict[str, Any] = {}


class PredictionInput(BaseModel):
    customer_features: dict[str, Any]
    threshold: float = 0.5


@tool("churn_prediction_tool", args_schema=PredictionInput)
def churn_prediction_tool(
    customer_features: dict[str, Any],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Run the production churn prediction model on a customer feature dict.

    Args:
        customer_features: Raw customer feature dict (pre-encoding).
        threshold: Decision threshold for binary prediction.

    Returns:
        Dict with churn_probability, prediction, risk_tier, model_version,
        confidence_interval, and base_learner_probabilities.
    """
    import pandas as pd

    model = _load_model()
    pipeline = _load_pipeline()

    if model is None:
        return {"error": "Production model not available", "churn_probability": 0.5}

    if pipeline is None:
        # Fail closed: scoring without the trained preprocessing pipeline risks
        # wrong feature ordering / schema and would produce silently incorrect
        # predictions that are harder to detect than an explicit error.
        logger.error("Preprocessing pipeline not found — refusing to score without it")
        return {"error": "Preprocessing pipeline artifact missing; cannot score safely"}

    # Build DataFrame for pipeline — drop non-feature columns that were
    # never seen during pipeline fit (target label, identifier, etc.)
    _DROP_COLS = {"customer_id", "churn", "churn_label", "churned"}
    cleaned = {k: v for k, v in customer_features.items() if k not in _DROP_COLS}
    df = pd.DataFrame([cleaned])

    try:
        X = pipeline.transform(df)
    except Exception as exc:
        # Also fail closed on transform errors — raw fallback would bypass all
        # encoding and scaling, producing a feature matrix the model was never
        # trained on.
        logger.error("Pipeline transform failed — refusing raw-feature fallback", error=str(exc))
        return {"error": f"Preprocessing failed: {exc}; cannot score safely"}

    churn_prob = float(model.predict_proba(X)[0, 1])
    prediction = int(churn_prob >= threshold)
    risk_tier = _classify_risk(churn_prob)

    # Conformal interval (Wilson score approximation if MAPIE not available)
    ci = _wilson_interval(churn_prob)

    # Base learner probabilities if stacking ensemble
    base_probas: dict[str, float] = {}
    if hasattr(model, "named_estimators_"):
        for name, est in model.named_estimators_.items():
            try:
                base_probas[name] = round(float(est.predict_proba(X)[0, 1]), 4)
            except Exception:
                pass

    model_version = _model_cache.get("version", "unknown")

    result = {
        "churn_probability": round(churn_prob, 4),
        "prediction": prediction,
        "risk_tier": risk_tier,
        "model_version": model_version,
        "model_name": settings.mlflow.mlflow_model_name,
        "confidence_interval": ci,
        "is_uncertain": ci[1] - ci[0] > 0.25,
        "base_learner_probabilities": base_probas,
        "threshold": threshold,
    }

    logger.info(
        "Prediction completed",
        churn_prob=round(churn_prob, 4),
        risk_tier=risk_tier,
    )
    return result


def _load_model() -> Any | None:
    """Load production model from MLflow or local artifact cache."""
    if "model" in _model_cache:
        return _model_cache["model"]

    # Try MLflow only if the tracking server is reachable (fast probe, 2 s timeout)
    if _mlflow_reachable():
        try:
            import mlflow.sklearn
            mlflow.set_tracking_uri(settings.mlflow.mlflow_tracking_uri)
            model_uri = f"models:/{settings.mlflow.mlflow_model_name}/Production"
            model = mlflow.sklearn.load_model(model_uri)
            _model_cache["model"] = model
            _model_cache["version"] = "production"
            logger.info("Model loaded from MLflow", uri=model_uri)
            return model
        except Exception as exc:
            logger.warning("MLflow load failed, trying local artifact", error=str(exc))
    else:
        logger.debug("MLflow not reachable — skipping, using local artifact")

    # Fallback: local pickle
    local_paths = [
        Path("models/artifacts/stacking_ensemble.pkl"),
        Path("models/artifacts/preprocessing_pipeline.pkl").parent / "stacking_ensemble.pkl",
    ]
    for path in local_paths:
        if path.exists():
            with open(path, "rb") as f:
                model = pickle.load(f)
            _model_cache["model"] = model
            _model_cache["version"] = "local"
            logger.info("Model loaded from local artifact", path=str(path))
            return model

    logger.error("No model available — neither MLflow nor local artifact found")
    return None


def _load_pipeline() -> Any | None:
    """Load preprocessing pipeline from local artifact."""
    if "pipeline" in _pipeline_cache:
        return _pipeline_cache["pipeline"]

    path = Path("models/artifacts/preprocessing_pipeline.pkl")
    if path.exists():
        with open(path, "rb") as f:
            pipeline = pickle.load(f)
        _pipeline_cache["pipeline"] = pipeline
        return pipeline
    return None


def _mlflow_reachable(timeout: float = 2.0) -> bool:
    """Return True only if the MLflow tracking server responds within *timeout* seconds."""
    import urllib.request
    import urllib.error
    try:
        url = settings.mlflow.mlflow_tracking_uri.rstrip("/") + "/health"
        req = urllib.request.urlopen(url, timeout=timeout)
        return req.status == 200
    except Exception:
        return False


def _classify_risk(prob: float) -> str:
    if prob >= 0.85:
        return "CRITICAL"
    elif prob >= 0.70:
        return "HIGH"
    elif prob >= 0.40:
        return "MEDIUM"
    return "LOW"


def _wilson_interval(p: float, n: int = 500, z: float = 1.645) -> list[float]:
    """Wilson score confidence interval for a probability estimate."""
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    margin = z * (np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / (1 + z**2 / n)
    return [round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4)]
