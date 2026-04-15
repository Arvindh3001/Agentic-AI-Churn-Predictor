"""
Optuna Hyperparameter Tuning
==============================
Bayesian hyperparameter optimisation for XGBoost and LightGBM
using Optuna with stratified k-fold cross-validation.

Usage:
    from src.models.hyperparameter_tuning import tune_xgboost, tune_lightgbm

    best_xgb_params = tune_xgboost(X_train, y_train, n_trials=50)
    best_lgbm_params = tune_lightgbm(X_train, y_train, n_trials=50)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import optuna
import structlog
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

logger = structlog.get_logger(__name__)

# Suppress Optuna logs below WARNING
optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = 30,
    cv_folds: int = 5,
    seed: int = 42,
    timeout: int | None = 600,
) -> dict[str, Any]:
    """
    Run Optuna Bayesian optimisation for XGBoost hyperparameters.

    Args:
        X_train: Training feature matrix.
        y_train: Binary churn labels.
        n_trials: Number of Optuna trials.
        cv_folds: Stratified CV folds for objective.
        seed: Random seed.
        timeout: Max seconds to run (None = unlimited).

    Returns:
        Best hyperparameter dict for XGBClassifier.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    def objective(trial: optuna.Trial) -> float:
        params: dict[str, Any] = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 5.0),
            "eval_metric": "logloss",
            "random_state": seed,
            "verbosity": 0,
        }

        model = XGBClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
        study_name="xgboost_churn",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    best_params = study.best_params
    logger.info(
        "XGBoost tuning complete",
        best_auc=round(study.best_value, 4),
        best_params=best_params,
    )
    return best_params


def tune_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = 30,
    cv_folds: int = 5,
    seed: int = 42,
    timeout: int | None = 600,
) -> dict[str, Any]:
    """
    Run Optuna Bayesian optimisation for LightGBM hyperparameters.

    Args:
        X_train: Training feature matrix.
        y_train: Binary churn labels.
        n_trials: Number of Optuna trials.
        cv_folds: Stratified CV folds for objective.
        seed: Random seed.
        timeout: Max seconds to run (None = unlimited).

    Returns:
        Best hyperparameter dict for LGBMClassifier.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    def objective(trial: optuna.Trial) -> float:
        params: dict[str, Any] = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "class_weight": "balanced",
            "random_state": seed,
            "n_jobs": -1,
            "verbose": -1,
        }

        model = LGBMClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
        study_name="lightgbm_churn",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    best_params = study.best_params
    logger.info(
        "LightGBM tuning complete",
        best_auc=round(study.best_value, 4),
        best_params=best_params,
    )
    return best_params
