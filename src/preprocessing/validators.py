"""
Pandera Schema Validators
==========================
Validates the raw customer DataFrame schema before it enters the pipeline.
Raises pandera.errors.SchemaError on any violation.

Usage:
    from src.preprocessing.validators import validate_raw_schema
    validate_raw_schema(df)
"""

from __future__ import annotations

import pandera as pa
import structlog
from pandera import Column, DataFrameSchema, Check

logger = structlog.get_logger(__name__)

# ------------------------------------------------------------------ #
# Raw input schema
# ------------------------------------------------------------------ #
RAW_CUSTOMER_SCHEMA = DataFrameSchema(
    columns={
        "customer_id": Column(
            str,
            nullable=False,
            unique=True,
            description="UUID customer identifier",
        ),
        "age": Column(
            float,
            checks=[Check.greater_than_or_equal_to(18), Check.less_than_or_equal_to(100)],
            nullable=False,
            coerce=True,
        ),
        "tenure_months": Column(
            float,
            checks=[Check.greater_than_or_equal_to(0), Check.less_than_or_equal_to(240)],
            nullable=False,
            coerce=True,
        ),
        "contract_type": Column(
            str,
            checks=Check.isin(["Month-to-Month", "One Year", "Two Year"]),
            nullable=False,
        ),
        "plan_tier": Column(
            str,
            checks=Check.isin(["Basic", "Standard", "Premium"]),
            nullable=False,
        ),
        "payment_method": Column(
            str,
            checks=Check.isin([
                "Credit Card",
                "Bank Transfer",
                "Electronic Check",
                "Mailed Check",
            ]),
            nullable=False,
        ),
        "monthly_charges": Column(
            float,
            checks=[Check.greater_than_or_equal_to(0), Check.less_than_or_equal_to(500)],
            nullable=False,
            coerce=True,
        ),
        "total_charges": Column(
            float,
            checks=Check.greater_than_or_equal_to(0),
            nullable=True,  # may be NaN for brand-new customers
            coerce=True,
        ),
        "num_support_tickets_30d": Column(
            float,
            checks=[Check.greater_than_or_equal_to(0), Check.less_than_or_equal_to(100)],
            nullable=True,
            coerce=True,
        ),
        "login_frequency_30d": Column(
            float,
            checks=Check.greater_than_or_equal_to(0),
            nullable=True,
            coerce=True,
        ),
        "feature_adoption_rate": Column(
            float,
            checks=[Check.greater_than_or_equal_to(0.0), Check.less_than_or_equal_to(1.0)],
            nullable=True,
            coerce=True,
        ),
        "nps_score": Column(
            float,
            checks=[Check.greater_than_or_equal_to(-100), Check.less_than_or_equal_to(100)],
            nullable=True,
            coerce=True,
        ),
        "usage_30d": Column(
            float,
            checks=Check.greater_than_or_equal_to(0),
            nullable=True,
            coerce=True,
        ),
        "usage_60d": Column(
            float,
            checks=Check.greater_than_or_equal_to(0),
            nullable=True,
            coerce=True,
        ),
        "usage_90d": Column(
            float,
            checks=Check.greater_than_or_equal_to(0),
            nullable=True,
            coerce=True,
        ),
        "days_since_last_login": Column(
            float,
            checks=[Check.greater_than_or_equal_to(0), Check.less_than_or_equal_to(1825)],
            nullable=True,
            coerce=True,
        ),
        "churn": Column(
            int,
            checks=Check.isin([0, 1]),
            nullable=False,
            coerce=True,
            required=False,  # may be absent in inference payloads
        ),
    },
    strict=False,  # allow extra columns (e.g., engineered features)
    coerce=True,
)

# Inference-time schema (no churn label required)
INFERENCE_SCHEMA = DataFrameSchema(
    columns={
        col: col_def
        for col, col_def in RAW_CUSTOMER_SCHEMA.columns.items()
        if col != "churn"
    },
    strict=False,
    coerce=True,
)


def validate_raw_schema(df: "pa.typing.DataFrame") -> None:
    """
    Validate raw customer data against the expected schema.
    Logs a warning per violation and raises SchemaError on failure.
    """
    try:
        RAW_CUSTOMER_SCHEMA.validate(df, lazy=True)
        logger.info("Schema validation passed", rows=len(df))
    except pa.errors.SchemaErrors as exc:
        logger.warning(
            "Schema validation failed",
            n_errors=len(exc.failure_cases),
            errors=exc.failure_cases.to_dict(orient="records")[:5],
        )
        raise


def validate_inference_schema(df: "pa.typing.DataFrame") -> None:
    """Validate inference-time payload (no churn label required)."""
    try:
        INFERENCE_SCHEMA.validate(df, lazy=True)
        logger.info("Inference schema validation passed", rows=len(df))
    except pa.errors.SchemaErrors as exc:
        logger.warning(
            "Inference schema validation failed",
            n_errors=len(exc.failure_cases),
        )
        raise
