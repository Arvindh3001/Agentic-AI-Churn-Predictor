"""
Model Training Orchestration
==============================
Trains all 5 churn prediction models, logs experiments to MLflow,
and optionally triggers hyperparameter tuning.

Models:
    - Logistic Regression (baseline)
    - Random Forest
    - XGBoost
    - LightGBM
    - Stacking Ensemble (XGB + LGBM + RF → LR meta)

Usage:
    python -m src.models.train
    python -m src.models.train --tune --n-trials 50
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import structlog
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.models.ensemble import build_stacking_ensemble
from src.models.evaluate import evaluate_model
from src.models.hyperparameter_tuning import tune_xgboost, tune_lightgbm
from src.models.registry import log_and_register_model
from src.preprocessing.pipeline import run_pipeline

logger = structlog.get_logger(__name__)

MODELS: dict[str, Any] = {
    "logistic_regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "xgboost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=2.7,  # compensate for ~27% churn
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    ),
    "lightgbm": LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    ),
}


def train_all_models(
    data_path: str | Path = "data/synthetic/customers.csv",
    experiment_name: str = "churn-prediction",
    tune: bool = False,
    n_trials: int = 30,
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """
    Train all models, evaluate on the test set, and log to MLflow.

    Args:
        data_path: Path to the raw CSV.
        experiment_name: MLflow experiment name.
        tune: If True, run Optuna tuning for XGBoost and LightGBM.
        n_trials: Number of Optuna trials (only used if tune=True).
        seed: Random seed.

    Returns:
        Dict mapping model_name → {metrics, run_id, model}.
    """
    # ------------------------------------------------------------------ #
    # Data preparation
    # ------------------------------------------------------------------ #
    logger.info("Loading and preprocessing data", path=str(data_path))
    X_train, X_test, y_train, y_test = run_pipeline(
        data_path=data_path,
        seed=seed,
        save_pipeline=True,
    )
    logger.info(
        "Data ready",
        train_shape=X_train.shape,
        test_shape=X_test.shape,
    )

    # ------------------------------------------------------------------ #
    # MLflow setup
    # ------------------------------------------------------------------ #
    mlflow.set_experiment(experiment_name)

    results: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    # Optional: hyperparameter tuning
    # ------------------------------------------------------------------ #
    if tune:
        logger.info("Running Optuna hyperparameter tuning...")
        xgb_params = tune_xgboost(X_train, y_train, n_trials=n_trials, seed=seed)
        lgbm_params = tune_lightgbm(X_train, y_train, n_trials=n_trials, seed=seed)
        MODELS["xgboost"].set_params(**xgb_params)
        MODELS["lightgbm"].set_params(**lgbm_params)
        logger.info("Tuning complete", xgb_params=xgb_params, lgbm_params=lgbm_params)

    # ------------------------------------------------------------------ #
    # Train base models
    # ------------------------------------------------------------------ #
    for model_name, model in MODELS.items():
        logger.info("Training model", model=model_name)

        with mlflow.start_run(run_name=model_name) as run:
            model.fit(X_train, y_train)

            metrics = evaluate_model(model, X_test, y_test, model_name=model_name)
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path="model")

            run_id = run.info.run_id
            results[model_name] = {
                "metrics": metrics,
                "run_id": run_id,
                "model": model,
            }

            logger.info(
                "Model trained and logged",
                model=model_name,
                auc_roc=round(metrics.get("auc_roc", 0), 4),
                f1=round(metrics.get("f1_macro", 0), 4),
                run_id=run_id,
            )

    # ------------------------------------------------------------------ #
    # Train stacking ensemble
    # ------------------------------------------------------------------ #
    logger.info("Training stacking ensemble...")
    ensemble = build_stacking_ensemble(
        xgb_model=MODELS["xgboost"],
        lgbm_model=MODELS["lightgbm"],
        rf_model=MODELS["random_forest"],
    )

    with mlflow.start_run(run_name="stacking_ensemble") as run:
        ensemble.fit(X_train, y_train)
        ensemble_metrics = evaluate_model(ensemble, X_test, y_test, model_name="stacking_ensemble")
        mlflow.log_metrics(ensemble_metrics)
        mlflow.sklearn.log_model(ensemble, artifact_path="model")

        run_id = run.info.run_id
        results["stacking_ensemble"] = {
            "metrics": ensemble_metrics,
            "run_id": run_id,
            "model": ensemble,
        }

        logger.info(
            "Ensemble trained and logged",
            auc_roc=round(ensemble_metrics.get("auc_roc", 0), 4),
            f1=round(ensemble_metrics.get("f1_macro", 0), 4),
            run_id=run_id,
        )

    # ------------------------------------------------------------------ #
    # Register champion model
    # ------------------------------------------------------------------ #
    best_model_name = max(results, key=lambda k: results[k]["metrics"].get("auc_roc", 0))
    best_run_id = results[best_model_name]["run_id"]

    log_and_register_model(
        run_id=best_run_id,
        model_name="churn-ensemble",
        stage="Staging",
        tags={"champion": "true", "source_model": best_model_name},
    )

    logger.info(
        "Champion model registered",
        model=best_model_name,
        auc_roc=round(results[best_model_name]["metrics"].get("auc_roc", 0), 4),
    )

    _print_leaderboard(results)
    return results


def _print_leaderboard(results: dict[str, dict[str, Any]]) -> None:
    """Print a formatted model comparison table."""
    print("\n" + "=" * 70)
    print(f"{'Model':<25} {'AUC-ROC':>10} {'F1-Macro':>10} {'Brier':>10}")
    print("-" * 70)
    for name, data in sorted(results.items(), key=lambda x: -x[1]["metrics"].get("auc_roc", 0)):
        m = data["metrics"]
        print(
            f"{name:<25} {m.get('auc_roc', 0):>10.4f} "
            f"{m.get('f1_macro', 0):>10.4f} {m.get('brier_score', 0):>10.4f}"
        )
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train churn prediction models")
    parser.add_argument("--data-path", default="data/synthetic/customers.csv")
    parser.add_argument("--experiment", default="churn-prediction")
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter tuning")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_all_models(
        data_path=args.data_path,
        experiment_name=args.experiment,
        tune=args.tune,
        n_trials=args.n_trials,
        seed=args.seed,
    )
