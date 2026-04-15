"""
Preprocessing Pipeline
======================
Scikit-learn Pipeline that takes raw customer CSV data and returns
a fully processed, model-ready feature matrix.

Steps:
    1. Schema validation (Pandera)
    2. Outlier capping (IQR)
    3. Missing value imputation (KNN / median)
    4. Temporal feature engineering
    5. Categorical encoding (Target / One-hot)
    6. Feature scaling (RobustScaler)
    7. Class imbalance handling (SMOTE — training only)

Usage:
    from src.preprocessing.pipeline import build_pipeline, run_pipeline

    X_train, X_test, y_train, y_test = run_pipeline("data/synthetic/customers.csv")
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.preprocessing.feature_engineering import FeatureEngineer
from src.preprocessing.validators import validate_raw_schema

logger = structlog.get_logger(__name__)

NUMERIC_FEATURES = [
    "age",
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_support_tickets_30d",
    "login_frequency_30d",
    "feature_adoption_rate",
    "nps_score",
    "usage_30d",
    "usage_60d",
    "usage_90d",
    "days_since_last_login",
    # Derived in feature engineering
    "usage_decline_rate",
    "support_escalation_flag",
    "charge_per_month_normalised",
    "tenure_bin",
]

CATEGORICAL_FEATURES = [
    "contract_type",
    "plan_tier",
    "payment_method",
]

TARGET_COL = "churn"


class IQROutlierCapper(BaseEstimator, TransformerMixin):
    """Cap numeric outliers at [Q1 - 1.5*IQR, Q3 + 1.5*IQR]."""

    def __init__(self, factor: float = 1.5) -> None:
        self.factor = factor
        self.lower_: dict[str, float] = {}
        self.upper_: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: Any = None) -> "IQROutlierCapper":
        for col in X.select_dtypes(include="number").columns:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            self.lower_[col] = q1 - self.factor * iqr
            self.upper_[col] = q3 + self.factor * iqr
        return self

    def transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        X = X.copy()
        for col, lower in self.lower_.items():
            if col in X.columns:
                X[col] = X[col].clip(lower=lower, upper=self.upper_[col])
        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """One-hot encode low-cardinality categorical columns."""

    def __init__(self, columns: list[str] | None = None) -> None:
        self.columns = columns or CATEGORICAL_FEATURES
        self._encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore",
            drop="first",
        )
        self.feature_names_out_: list[str] = []

    def fit(self, X: pd.DataFrame, y: Any = None) -> "CategoricalEncoder":
        present = [c for c in self.columns if c in X.columns]
        self._encoder.fit(X[present])
        self.feature_names_out_ = list(self._encoder.get_feature_names_out(present))
        self._present = present
        return self

    def transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        X = X.copy()
        encoded = self._encoder.transform(X[self._present])
        encoded_df = pd.DataFrame(encoded, columns=self.feature_names_out_, index=X.index)
        X = X.drop(columns=self._present)
        return pd.concat([X, encoded_df], axis=1)


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Keep only the expected feature columns, in fixed order."""

    def __init__(self, columns: list[str] | None = None) -> None:
        self.columns = columns or []
        self._fitted_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y: Any = None) -> "ColumnSelector":
        self._fitted_cols = [c for c in self.columns if c in X.columns]
        return self

    def transform(self, X: pd.DataFrame, y: Any = None) -> np.ndarray:
        return X[self._fitted_cols].values


def build_pipeline() -> Pipeline:
    """
    Construct and return the full scikit-learn preprocessing pipeline.
    Does NOT include model — this is feature processing only.
    """
    return Pipeline(
        steps=[
            ("feature_engineer", FeatureEngineer()),
            ("outlier_capper", IQROutlierCapper(factor=1.5)),
            ("categorical_encoder", CategoricalEncoder()),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load raw CSV, validate schema, return DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    df = pd.read_csv(path)
    logger.info("Raw data loaded", rows=len(df), columns=list(df.columns), path=str(path))

    validate_raw_schema(df)
    return df


def run_pipeline(
    data_path: str | Path = "data/synthetic/customers.csv",
    test_size: float = 0.2,
    seed: int = 42,
    save_pipeline: bool = True,
    pipeline_output_path: str | Path = "models/artifacts/preprocessing_pipeline.pkl",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Full end-to-end preprocessing run.

    Args:
        data_path: Path to raw CSV.
        test_size: Fraction of data for test split.
        seed: Random seed.
        save_pipeline: Whether to pickle the fitted pipeline.
        pipeline_output_path: Where to save the pickled pipeline.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test) as numpy arrays.
    """
    df = load_raw_data(data_path)

    X = df.drop(columns=[TARGET_COL, "customer_id"], errors="ignore")
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    pipeline = build_pipeline()
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)

    logger.info(
        "Pipeline fit complete",
        train_shape=X_train_processed.shape,
        test_shape=X_test_processed.shape,
        churn_rate_train=round(y_train.mean(), 4),
        churn_rate_test=round(y_test.mean(), 4),
    )

    if save_pipeline:
        out_path = Path(pipeline_output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(pipeline, f)
        logger.info("Pipeline saved", path=str(out_path))

    return X_train_processed, X_test_processed, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = run_pipeline()
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Churn rate — train: {y_train.mean():.2%} | test: {y_test.mean():.2%}")
