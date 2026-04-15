"""
Synthetic Customer Data Generator
==================================
Generates 10,000 realistic customer records with churn labels (~27% churn rate).
Churn is correlated with features in a realistic, non-trivial way.

Usage:
    python data/synthetic/generate_synthetic.py
    python data/synthetic/generate_synthetic.py --n-customers 5000 --seed 99
"""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = OUTPUT_DIR / "customers.csv"


def generate_customers(
    n_customers: int = 10_000,
    seed: int = 42,
    churn_rate: float = 0.27,
) -> pd.DataFrame:
    """
    Generate a synthetic churn dataset with realistic feature correlations.

    Args:
        n_customers: Number of customer rows to generate.
        seed: Random seed for reproducibility.
        churn_rate: Target churn rate (approximate).

    Returns:
        DataFrame with all churn features and labels.
    """
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # 1. Core demographic / account features
    # ------------------------------------------------------------------ #
    customer_ids: list[str] = [str(uuid.uuid4()) for _ in range(n_customers)]

    age = rng.integers(18, 76, size=n_customers).astype(float)

    tenure_months = rng.integers(1, 85, size=n_customers).astype(float)

    contract_type = rng.choice(
        ["Month-to-Month", "One Year", "Two Year"],
        size=n_customers,
        p=[0.55, 0.25, 0.20],
    )

    plan_tier = rng.choice(
        ["Basic", "Standard", "Premium"],
        size=n_customers,
        p=[0.40, 0.35, 0.25],
    )

    payment_method = rng.choice(
        ["Credit Card", "Bank Transfer", "Electronic Check", "Mailed Check"],
        size=n_customers,
        p=[0.30, 0.25, 0.30, 0.15],
    )

    # Monthly charges correlated with plan tier
    plan_base: dict[str, float] = {"Basic": 25.0, "Standard": 55.0, "Premium": 90.0}
    monthly_charges = np.array(
        [plan_base[t] + rng.normal(0, 8) for t in plan_tier]
    ).clip(18.0, 120.0)

    # Total charges = tenure × monthly + noise
    total_charges = (
        tenure_months * monthly_charges + rng.normal(0, 50, size=n_customers)
    ).clip(0)

    # ------------------------------------------------------------------ #
    # 2. Behavioural features
    # ------------------------------------------------------------------ #
    # Support tickets: higher for Month-to-Month + short tenure
    contract_ticket_bias = np.where(contract_type == "Month-to-Month", 1.5, 0.5)
    tenure_ticket_bias = np.clip(1.0 - tenure_months / 84, 0, 1)
    ticket_lambda = 1.0 + contract_ticket_bias * tenure_ticket_bias * 3
    num_support_tickets_30d = rng.poisson(ticket_lambda).clip(0, 10).astype(float)

    # Login frequency: higher for Premium + engaged users
    plan_login_bias: dict[str, float] = {"Basic": 10.0, "Standard": 20.0, "Premium": 35.0}
    login_base = np.array([plan_login_bias[t] for t in plan_tier])
    login_frequency_30d = rng.normal(login_base, 8).clip(0, 60).astype(float)

    # Feature adoption: correlated with plan tier and tenure
    plan_adoption: dict[str, float] = {"Basic": 0.25, "Standard": 0.55, "Premium": 0.80}
    adoption_base = np.array([plan_adoption[t] for t in plan_tier])
    tenure_factor = (tenure_months / 84) * 0.3
    feature_adoption_rate = (
        rng.beta(
            a=adoption_base * 5 + tenure_factor,
            b=(1 - adoption_base) * 5 + 0.5,
        )
    ).clip(0.0, 1.0)

    # NPS score: correlated with feature adoption and inverse support tickets
    nps_base = (feature_adoption_rate - 0.5) * 160 - num_support_tickets_30d * 10
    nps_score = (nps_base + rng.normal(0, 20, n_customers)).clip(-100, 100).astype(float)

    # ------------------------------------------------------------------ #
    # 3. Temporal usage (30/60/90 day trend) — declining = churn signal
    # ------------------------------------------------------------------ #
    usage_90d_base = rng.normal(50, 15, n_customers).clip(0, 100)

    # Some customers have declining usage — churn signal
    declining_mask = rng.random(n_customers) < 0.35
    decline_rate = rng.uniform(0.3, 0.8, n_customers)

    usage_90d = usage_90d_base.copy()
    usage_60d = np.where(
        declining_mask,
        usage_90d * (1 - decline_rate * 0.4),
        usage_90d * rng.uniform(0.9, 1.1, n_customers),
    ).clip(0, 100)
    usage_30d = np.where(
        declining_mask,
        usage_60d * (1 - decline_rate * 0.4),
        usage_60d * rng.uniform(0.9, 1.1, n_customers),
    ).clip(0, 100)

    # Days since last login: correlated with churn
    days_since_last_login = np.where(
        declining_mask,
        rng.integers(30, 365, size=n_customers),
        rng.integers(0, 60, size=n_customers),
    ).astype(float)

    # ------------------------------------------------------------------ #
    # 4. Churn label — logistic model over risk factors
    # ------------------------------------------------------------------ #
    # Normalise features to [0,1] for logit scoring
    def _norm(arr: np.ndarray) -> np.ndarray:
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-9)

    # Contract type risk
    contract_risk = np.where(
        contract_type == "Month-to-Month", 1.0,
        np.where(contract_type == "One Year", 0.4, 0.1),
    )

    # Logit score — higher = more likely to churn
    logit = (
        2.5 * contract_risk
        + 1.5 * _norm(num_support_tickets_30d)
        - 1.5 * _norm(feature_adoption_rate)
        - 1.2 * _norm(login_frequency_30d)
        + 1.8 * _norm(days_since_last_login)
        - 1.0 * _norm(nps_score + 100)
        + 1.0 * declining_mask.astype(float)
        - 0.8 * _norm(tenure_months)
        + rng.normal(0, 0.5, n_customers)  # noise
    )

    # Calibrate intercept to hit target churn rate
    target_log_odds = np.log(churn_rate / (1 - churn_rate))
    intercept = target_log_odds - logit.mean()
    churn_prob = 1 / (1 + np.exp(-(logit + intercept)))
    churn = (rng.random(n_customers) < churn_prob).astype(int)

    actual_rate = churn.mean()
    logger.info("Churn label generated", target_rate=churn_rate, actual_rate=round(actual_rate, 4))

    # ------------------------------------------------------------------ #
    # 5. Assemble DataFrame
    # ------------------------------------------------------------------ #
    df = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "age": age.round(0).astype(int),
            "tenure_months": tenure_months.round(0).astype(int),
            "contract_type": contract_type,
            "plan_tier": plan_tier,
            "payment_method": payment_method,
            "monthly_charges": monthly_charges.round(2),
            "total_charges": total_charges.round(2),
            "num_support_tickets_30d": num_support_tickets_30d.astype(int),
            "login_frequency_30d": login_frequency_30d.round(1),
            "feature_adoption_rate": feature_adoption_rate.round(4),
            "nps_score": nps_score.round(1),
            "usage_30d": usage_30d.round(2),
            "usage_60d": usage_60d.round(2),
            "usage_90d": usage_90d.round(2),
            "days_since_last_login": days_since_last_login.round(0).astype(int),
            "churn": churn,
        }
    )

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic churn dataset")
    parser.add_argument("--n-customers", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--churn-rate", type=float, default=0.27)
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    args = parser.parse_args()

    logger.info(
        "Starting synthetic data generation",
        n_customers=args.n_customers,
        seed=args.seed,
        target_churn_rate=args.churn_rate,
    )

    df = generate_customers(
        n_customers=args.n_customers,
        seed=args.seed,
        churn_rate=args.churn_rate,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(
        "Synthetic dataset saved",
        path=str(output_path),
        rows=len(df),
        churn_rate=round(df["churn"].mean(), 4),
        columns=list(df.columns),
    )
    print(df.describe())
    print(f"\nChurn rate: {df['churn'].mean():.2%}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
