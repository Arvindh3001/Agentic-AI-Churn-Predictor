"""
MLflow Model Registry Helpers
==============================
Utilities for registering, promoting, and loading models from the
MLflow Model Registry with a champion-challenger pattern.

Lifecycle stages: Staging → Validation → Production

Usage:
    from src.models.registry import log_and_register_model, get_champion_model

    log_and_register_model(run_id="abc123", model_name="churn-ensemble", stage="Staging")
    model = get_champion_model("churn-ensemble")
"""

from __future__ import annotations

from typing import Any

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import structlog
from mlflow.tracking import MlflowClient

logger = structlog.get_logger(__name__)

MODEL_REGISTRY_NAME = "churn-ensemble"
PRODUCTION_STAGE = "Production"
STAGING_STAGE = "Staging"
ARCHIVED_STAGE = "Archived"


def get_mlflow_client() -> MlflowClient:
    """Return a configured MLflow tracking client."""
    return MlflowClient()


def log_and_register_model(
    run_id: str,
    model_name: str = MODEL_REGISTRY_NAME,
    stage: str = STAGING_STAGE,
    tags: dict[str, str] | None = None,
    description: str = "",
) -> str:
    """
    Register an MLflow run's model artifact into the Model Registry.

    Args:
        run_id: MLflow run ID containing the logged model.
        model_name: Registry model name.
        stage: Lifecycle stage to set after registration.
        tags: Optional key-value tags to attach to the model version.
        description: Human-readable description for this version.

    Returns:
        Model version string (e.g. "3").
    """
    client = get_mlflow_client()
    model_uri = f"runs:/{run_id}/model"

    registered = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = registered.version

    if description:
        client.update_model_version(
            name=model_name,
            version=version,
            description=description,
        )

    if tags:
        for key, value in tags.items():
            client.set_model_version_tag(name=model_name, version=version, key=key, value=value)

    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=(stage == PRODUCTION_STAGE),
    )

    logger.info(
        "Model registered",
        name=model_name,
        version=version,
        stage=stage,
        run_id=run_id,
    )
    return str(version)


def promote_to_production(
    model_name: str = MODEL_REGISTRY_NAME,
    version: str | None = None,
    archive_existing: bool = True,
) -> None:
    """
    Promote a model version to Production.

    If version is None, promotes the latest Staging version.
    """
    client = get_mlflow_client()

    if version is None:
        staging_versions = client.get_latest_versions(model_name, stages=[STAGING_STAGE])
        if not staging_versions:
            raise ValueError(f"No model in Staging stage for '{model_name}'")
        version = staging_versions[0].version

    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=PRODUCTION_STAGE,
        archive_existing_versions=archive_existing,
    )

    logger.info(
        "Model promoted to Production",
        name=model_name,
        version=version,
    )


def get_champion_model(
    model_name: str = MODEL_REGISTRY_NAME,
    stage: str = PRODUCTION_STAGE,
) -> Any:
    """
    Load and return the current champion model from the registry.

    Args:
        model_name: Registered model name.
        stage: Stage to load from (default: Production).

    Returns:
        Loaded sklearn model object.
    """
    model_uri = f"models:/{model_name}/{stage}"
    model = mlflow.sklearn.load_model(model_uri)
    logger.info("Champion model loaded", name=model_name, stage=stage)
    return model


def get_model_versions(
    model_name: str = MODEL_REGISTRY_NAME,
) -> list[dict[str, Any]]:
    """
    Return summary of all registered versions for a model.

    Returns:
        List of dicts with version, stage, creation_timestamp, run_id.
    """
    client = get_mlflow_client()
    versions = client.search_model_versions(f"name='{model_name}'")

    return [
        {
            "version": v.version,
            "stage": v.current_stage,
            "run_id": v.run_id,
            "description": v.description,
            "tags": dict(v.tags),
        }
        for v in versions
    ]


def archive_model_version(
    model_name: str = MODEL_REGISTRY_NAME,
    version: str = "1",
) -> None:
    """Archive a specific model version."""
    client = get_mlflow_client()
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=ARCHIVED_STAGE,
    )
    logger.info("Model version archived", name=model_name, version=version)


def compare_champion_challenger(
    X_test: Any,
    y_test: Any,
    model_name: str = MODEL_REGISTRY_NAME,
) -> dict[str, Any]:
    """
    Load champion (Production) and challenger (Staging), compare AUC-ROC.

    Returns:
        Dict with champion_auc, challenger_auc, and promote_recommendation.
    """
    from src.models.evaluate import evaluate_model

    results: dict[str, Any] = {}

    for stage in [PRODUCTION_STAGE, STAGING_STAGE]:
        try:
            model = get_champion_model(model_name=model_name, stage=stage)
            metrics = evaluate_model(model, X_test, y_test, model_name=f"{stage.lower()}_model")
            results[stage.lower()] = metrics
        except Exception as exc:
            logger.warning("Could not load model", stage=stage, error=str(exc))
            results[stage.lower()] = {}

    champion_auc = results.get("production", {}).get("auc_roc", 0)
    challenger_auc = results.get("staging", {}).get("auc_roc", 0)

    results["promote_recommendation"] = challenger_auc > champion_auc + 0.005

    logger.info(
        "Champion-Challenger comparison",
        champion_auc=round(champion_auc, 4),
        challenger_auc=round(challenger_auc, 4),
        promote=results["promote_recommendation"],
    )
    return results
