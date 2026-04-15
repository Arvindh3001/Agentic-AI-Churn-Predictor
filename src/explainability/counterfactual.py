"""
Counterfactual Engine — DiCE Wrapper
======================================
Generates diverse counterfactual explanations (what needs to change to
prevent churn) using DiCE-ml, then filters and ranks them by business
feasibility: cost, effort, impact, and time-to-effect.

Output example:
    {
        "customer_id": "C-4821",
        "current_churn_prob": 0.87,
        "interventions": [
            {"action": "Reduce price by 15%", "new_churn_prob": 0.31, "cost_usd": 45},
            {"action": "Assign dedicated CSM",  "new_churn_prob": 0.28, "cost_usd": 120},
            {"action": "Offer loyalty upgrade", "new_churn_prob": 0.22, "cost_usd": 80}
        ]
    }

Usage:
    from src.explainability.counterfactual import CounterfactualEngine

    engine = CounterfactualEngine(model=fitted_model, X_train=X_train_df,
                                   feature_names=feature_names)
    result = engine.generate(customer_row_df, customer_id="C-4821")
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

# Business cost map: which feature changes map to which retention action
# cost_usd is approximate; effort is 1(low)–5(high); days_to_effect is typical timeline
INTERVENTION_CATALOGUE: list[dict[str, Any]] = [
    {
        "feature": "monthly_charges",
        "direction": "decrease",
        "action_template": "Reduce monthly price by {pct:.0f}%",
        "cost_usd_per_unit": 3.0,  # per USD discount per month
        "effort": 1,
        "days_to_effect": 7,
    },
    {
        "feature": "feature_adoption_rate",
        "direction": "increase",
        "action_template": "Assign dedicated onboarding / CSM support",
        "cost_usd_per_unit": 50.0,  # flat per intervention
        "effort": 3,
        "days_to_effect": 30,
    },
    {
        "feature": "contract_type",
        "direction": "upgrade",
        "action_template": "Offer annual contract upgrade with loyalty discount",
        "cost_usd_per_unit": 80.0,
        "effort": 2,
        "days_to_effect": 14,
    },
    {
        "feature": "num_support_tickets_30d",
        "direction": "decrease",
        "action_template": "Prioritise support queue — proactive resolution",
        "cost_usd_per_unit": 20.0,
        "effort": 2,
        "days_to_effect": 3,
    },
    {
        "feature": "plan_tier",
        "direction": "upgrade",
        "action_template": "Offer complimentary plan tier upgrade",
        "cost_usd_per_unit": 60.0,
        "effort": 1,
        "days_to_effect": 1,
    },
]

# Business constraints
MAX_DISCOUNT_PCT = 0.30       # max 30% price reduction
MAX_COST_USD = 300.0          # max retention spend per customer
MIN_NEW_CHURN_PROB = 0.05     # floor — can't promise zero churn


class CounterfactualEngine:
    """
    Generates and ranks actionable counterfactual interventions.

    Uses DiCE when available; falls back to a gradient-free perturbation
    search if DiCE is incompatible with the current Python version.

    Args:
        model: Fitted sklearn model with predict_proba.
        X_train: Training DataFrame (with feature names as columns).
        feature_names: List of feature column names.
        outcome_col: Name of the target column (excluded from CFs).
    """

    def __init__(
        self,
        model: Any,
        X_train: pd.DataFrame,
        feature_names: list[str],
        outcome_col: str = "churn",
        use_dice: bool = True,
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        self._X_train = X_train
        self._use_dice = use_dice
        self._dice_exp: Any = None

        if use_dice:
            self._init_dice(X_train, outcome_col)

    def _init_dice(self, X_train: pd.DataFrame, outcome_col: str) -> None:
        try:
            import dice_ml
            from dice_ml import Dice

            train_with_target = X_train.copy()
            if outcome_col not in train_with_target.columns:
                preds = self.model.predict(X_train.values)
                train_with_target[outcome_col] = preds

            d = dice_ml.Data(
                dataframe=train_with_target,
                continuous_features=[
                    f for f in self.feature_names
                    if train_with_target[f].dtype in [np.float64, np.float32, np.int64, np.int32]
                ],
                outcome_name=outcome_col,
            )
            m = dice_ml.Model(model=self.model, backend="sklearn")
            self._dice_exp = Dice(d, m, method="random")
            logger.info("DiCE counterfactual engine initialised")
        except Exception as exc:
            logger.warning("DiCE initialisation failed, using fallback perturbation", error=str(exc))
            self._use_dice = False

    def generate(
        self,
        customer_df: pd.DataFrame,
        customer_id: str = "unknown",
        n_counterfactuals: int = 10,
        desired_class: int = 0,  # flip from churn (1) to stay (0)
        top_k: int = 3,
    ) -> dict[str, Any]:
        """
        Generate and rank counterfactual interventions for one customer.

        Args:
            customer_df: Single-row DataFrame with customer features.
            customer_id: For output labelling.
            n_counterfactuals: How many diverse CFs to generate.
            desired_class: Target class (0 = stay).
            top_k: How many interventions to return in the final output.

        Returns:
            Structured intervention dict (see module docstring).
        """
        x = customer_df[self.feature_names].values[0]
        current_prob = float(self.model.predict_proba(x.reshape(1, -1))[0, 1])

        if self._use_dice and self._dice_exp is not None:
            cf_df = self._generate_dice_cfs(customer_df, n_counterfactuals, desired_class)
        else:
            cf_df = self._generate_perturbation_cfs(x, n_counterfactuals)

        interventions = self._rank_interventions(x, cf_df, customer_df)
        interventions = [i for i in interventions if i["cost_usd"] <= MAX_COST_USD]
        interventions = sorted(interventions, key=lambda i: (i["new_churn_prob"], i["cost_usd"]))

        result: dict[str, Any] = {
            "customer_id": customer_id,
            "current_churn_prob": round(current_prob, 4),
            "interventions": interventions[:top_k],
        }

        logger.info(
            "Counterfactuals generated",
            customer_id=customer_id,
            current_prob=round(current_prob, 4),
            n_interventions=len(interventions[:top_k]),
        )
        return result

    def _generate_dice_cfs(
        self,
        customer_df: pd.DataFrame,
        n: int,
        desired_class: int,
    ) -> pd.DataFrame:
        try:
            cf_result = self._dice_exp.generate_counterfactuals(
                query_instances=customer_df[self.feature_names],
                total_CFs=n,
                desired_class=desired_class,
                verbose=False,
            )
            cf_df: pd.DataFrame = cf_result.cf_examples_list[0].final_cfs_df
            return cf_df[self.feature_names]
        except Exception as exc:
            logger.warning("DiCE generation failed, falling back", error=str(exc))
            return self._generate_perturbation_cfs(
                customer_df[self.feature_names].values[0], n
            )

    def _generate_perturbation_cfs(
        self, x: np.ndarray, n: int
    ) -> pd.DataFrame:
        """
        Fallback: random feature perturbations that flip prediction to 'stay'.
        Searches the feature neighbourhood using training data quantiles.
        """
        rng = np.random.default_rng(42)
        candidates: list[np.ndarray] = []

        for _ in range(n * 50):  # over-sample, filter below
            perturbed = x.copy()
            n_changes = rng.integers(1, min(4, len(x)))
            change_idx = rng.choice(len(x), size=n_changes, replace=False)

            for idx in change_idx:
                col = self._X_train.iloc[:, idx]
                perturbed[idx] = float(col.sample(1, random_state=None).values[0])

            prob = float(self.model.predict_proba(perturbed.reshape(1, -1))[0, 1])
            if prob < 0.5:
                candidates.append(perturbed)

            if len(candidates) >= n:
                break

        if not candidates:
            logger.warning("No flipping counterfactuals found in perturbation search")
            return pd.DataFrame(columns=self.feature_names)

        return pd.DataFrame(candidates[:n], columns=self.feature_names)

    def _rank_interventions(
        self,
        x_orig: np.ndarray,
        cf_df: pd.DataFrame,
        customer_df: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """Convert raw CF rows to labelled, costed interventions."""
        if cf_df.empty:
            return []

        orig_series = pd.Series(x_orig, index=self.feature_names)
        interventions: list[dict[str, Any]] = []

        for _, cf_row in cf_df.iterrows():
            changed: list[str] = []
            total_cost = 0.0

            for feat in self.feature_names:
                orig_val = orig_series.get(feat, 0)
                cf_val = cf_row.get(feat, orig_val)
                if abs(float(cf_val) - float(orig_val)) > 1e-4:
                    changed.append(feat)

            if not changed:
                continue

            cf_proba = float(self.model.predict_proba(
                cf_row[self.feature_names].values.reshape(1, -1)
            )[0, 1])

            action_parts: list[str] = []
            for feat in changed:
                catalogue_entry = next(
                    (c for c in INTERVENTION_CATALOGUE if c["feature"] == feat), None
                )
                orig_val = float(orig_series.get(feat, 0))
                cf_val = float(cf_row.get(feat, orig_val))
                delta = cf_val - orig_val

                if catalogue_entry:
                    if feat == "monthly_charges" and delta < 0:
                        pct = abs(delta) / max(orig_val, 1) * 100
                        if pct > MAX_DISCOUNT_PCT * 100:
                            continue
                        action_parts.append(
                            catalogue_entry["action_template"].format(pct=pct)
                        )
                        total_cost += abs(delta) * catalogue_entry["cost_usd_per_unit"]
                    else:
                        action_parts.append(catalogue_entry["action_template"])
                        total_cost += catalogue_entry["cost_usd_per_unit"]
                else:
                    direction = "increase" if delta > 0 else "decrease"
                    action_parts.append(f"{direction.capitalize()} {feat} by {abs(delta):.2f}")
                    total_cost += 10.0  # default cost for unknown features

            if not action_parts:
                continue

            interventions.append(
                {
                    "action": "; ".join(action_parts),
                    "new_churn_prob": round(max(cf_proba, MIN_NEW_CHURN_PROB), 4),
                    "churn_prob_reduction": round(
                        max(0, float(self.model.predict_proba(
                            x_orig.reshape(1, -1))[0, 1]) - cf_proba),
                        4,
                    ),
                    "cost_usd": round(total_cost, 2),
                    "changed_features": changed,
                }
            )

        return interventions
