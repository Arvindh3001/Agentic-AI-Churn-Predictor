"""
Data Drift Detector
====================
Detects feature drift between a reference (training baseline) dataset
and a current (production/new) dataset using scipy statistical tests.

    - Numeric features  : Kolmogorov-Smirnov two-sample test
    - Categorical features: Chi-square test on frequency distributions

Generates an HTML report and a machine-readable drift summary dict.

Note: Replaces Evidently AI which is incompatible with Python 3.14+
      due to pydantic.v1 internals.

Usage:
    from src.preprocessing.drift_detector import DriftDetector

    detector = DriftDetector(reference_path="data/processed/train_baseline.csv")
    summary = detector.run(current_df=new_df)
    detector.save_report(summary, output_path="reports/explainability/drift_report.html")
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog
from scipy import stats

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
]

CATEGORICAL_FEATURES = [
    "contract_type",
    "plan_tier",
    "payment_method",
]

# Significance level for drift decision
ALPHA = 0.05


class DriftDetector:
    """
    Statistical drift detector using KS-test and chi-square test.

    Args:
        reference_path: Path to reference CSV (training baseline).
        reference_df: Alternatively, pass the reference DataFrame directly.
    """

    def __init__(
        self,
        reference_path: str | Path | None = None,
        reference_df: pd.DataFrame | None = None,
    ) -> None:
        if reference_df is not None:
            self._reference = reference_df
        elif reference_path is not None:
            self._reference = pd.read_csv(reference_path)
            logger.info(
                "Reference dataset loaded",
                rows=len(self._reference),
                path=str(reference_path),
            )
        else:
            raise ValueError("Provide either reference_path or reference_df.")

    def run(self, current_df: pd.DataFrame) -> dict[str, Any]:
        """
        Run drift detection against the reference dataset.

        Args:
            current_df: New data to compare against the baseline.

        Returns:
            Drift summary dict with per-feature results and overall status.
        """
        feature_results: dict[str, dict[str, Any]] = {}

        # Numeric features — KS test
        for col in NUMERIC_FEATURES:
            if col not in self._reference.columns or col not in current_df.columns:
                continue
            ref_vals = self._reference[col].dropna().values
            cur_vals = current_df[col].dropna().values
            if len(ref_vals) < 2 or len(cur_vals) < 2:
                continue

            ks_stat, p_value = stats.ks_2samp(ref_vals, cur_vals)
            drifted = bool(p_value < ALPHA)

            feature_results[col] = {
                "type": "numeric",
                "test": "kolmogorov_smirnov",
                "statistic": round(float(ks_stat), 6),
                "p_value": round(float(p_value), 6),
                "drifted": drifted,
                "ref_mean": round(float(ref_vals.mean()), 4),
                "cur_mean": round(float(cur_vals.mean()), 4),
                "ref_std": round(float(ref_vals.std()), 4),
                "cur_std": round(float(cur_vals.std()), 4),
            }

        # Categorical features — chi-square test
        for col in CATEGORICAL_FEATURES:
            if col not in self._reference.columns or col not in current_df.columns:
                continue

            all_cats = set(self._reference[col].unique()) | set(current_df[col].unique())
            ref_counts = self._reference[col].value_counts().reindex(all_cats, fill_value=0)
            cur_counts = current_df[col].value_counts().reindex(all_cats, fill_value=0)

            # Normalise to same total so chi-square is valid
            ref_freq = ref_counts / ref_counts.sum()
            cur_expected = ref_freq * cur_counts.sum()

            # Avoid zero expected cells
            mask = cur_expected > 0
            if mask.sum() < 2:
                continue

            chi2, p_value = stats.chisquare(
                f_obs=cur_counts[mask].values,
                f_exp=cur_expected[mask].values,
            )
            drifted = bool(p_value < ALPHA)

            feature_results[col] = {
                "type": "categorical",
                "test": "chi_square",
                "statistic": round(float(chi2), 6),
                "p_value": round(float(p_value), 6),
                "drifted": drifted,
                "ref_distribution": ref_freq.round(4).to_dict(),
                "cur_distribution": (
                    cur_counts / cur_counts.sum()
                ).round(4).to_dict(),
            }

        drifted_features = [f for f, r in feature_results.items() if r["drifted"]]
        n_tested = len(feature_results)
        n_drifted = len(drifted_features)
        share_drifted = round(n_drifted / n_tested, 4) if n_tested > 0 else 0.0
        dataset_drift = share_drifted >= 0.5  # majority of features drifted

        summary: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "reference_rows": len(self._reference),
            "current_rows": len(current_df),
            "features_tested": n_tested,
            "features_drifted": n_drifted,
            "share_drifted_features": share_drifted,
            "dataset_drift": dataset_drift,
            "alpha": ALPHA,
            "drifted_features": drifted_features,
            "feature_results": feature_results,
        }

        logger.info(
            "Drift analysis complete",
            features_tested=n_tested,
            features_drifted=n_drifted,
            dataset_drift=dataset_drift,
        )
        return summary

    def save_report(
        self,
        summary: dict[str, Any],
        output_path: str | Path = "reports/explainability/drift_report.html",
    ) -> Path:
        """Render drift summary as a self-contained HTML report."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        html = _render_html_report(summary)
        output_path.write_text(html, encoding="utf-8")
        logger.info("Drift report saved", path=str(output_path))
        return output_path

    def get_drift_summary(self, summary: dict[str, Any]) -> dict[str, Any]:
        """Return the compact machine-readable summary (MLflow-loggable)."""
        return {
            "share_drifted_features": summary["share_drifted_features"],
            "number_drifted_features": summary["features_drifted"],
            "dataset_drift": summary["dataset_drift"],
            "drifted_features": summary["drifted_features"],
        }


def _render_html_report(summary: dict[str, Any]) -> str:
    """Build a minimal self-contained HTML drift report."""
    rows = ""
    for feat, result in summary["feature_results"].items():
        status = "🔴 DRIFT" if result["drifted"] else "🟢 OK"
        stat = result["statistic"]
        pval = result["p_value"]
        test = result["test"].replace("_", " ").title()
        rows += (
            f"<tr>"
            f"<td>{feat}</td>"
            f"<td>{result['type']}</td>"
            f"<td>{test}</td>"
            f"<td>{stat:.4f}</td>"
            f"<td>{pval:.4f}</td>"
            f"<td>{status}</td>"
            f"</tr>\n"
        )

    drift_banner = (
        '<p style="color:red;font-weight:bold">⚠ DATASET DRIFT DETECTED</p>'
        if summary["dataset_drift"]
        else '<p style="color:green;font-weight:bold">✓ No significant dataset drift</p>'
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Churn Platform — Drift Report</title>
<style>
  body {{ font-family: sans-serif; max-width: 1000px; margin: 40px auto; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ccc; padding: 8px 12px; text-align: left; }}
  th {{ background: #f5f5f5; }}
  tr:nth-child(even) {{ background: #fafafa; }}
</style>
</head>
<body>
<h1>Data Drift Report</h1>
<p><strong>Generated:</strong> {summary['timestamp']}</p>
<p><strong>Reference rows:</strong> {summary['reference_rows']} |
   <strong>Current rows:</strong> {summary['current_rows']}</p>
<p><strong>Features drifted:</strong> {summary['features_drifted']} / {summary['features_tested']}
   ({summary['share_drifted_features']:.0%})</p>
{drift_banner}
<h2>Per-Feature Results (α = {summary['alpha']})</h2>
<table>
<thead><tr>
  <th>Feature</th><th>Type</th><th>Test</th>
  <th>Statistic</th><th>p-value</th><th>Status</th>
</tr></thead>
<tbody>
{rows}
</tbody>
</table>
<details><summary>Raw JSON</summary>
<pre>{json.dumps(summary, indent=2)}</pre>
</details>
</body>
</html>"""


def generate_baseline_report(
    synthetic_csv: str | Path = "data/synthetic/customers.csv",
    output_html: str | Path = "reports/explainability/baseline_drift_report.html",
    reference_fraction: float = 0.5,
    seed: int = 42,
) -> None:
    """
    Split synthetic dataset into reference / current halves,
    run the drift report, and save to disk.
    """
    df = pd.read_csv(synthetic_csv)
    reference = df.sample(frac=reference_fraction, random_state=seed)
    current = df.drop(reference.index)

    detector = DriftDetector(reference_df=reference)
    summary = detector.run(current)
    detector.save_report(summary, output_html)

    compact = detector.get_drift_summary(summary)
    logger.info("Baseline drift summary", **compact)
    print(f"\nDrift summary: {compact}")
    print(f"Report saved to: {output_html}")


if __name__ == "__main__":
    generate_baseline_report()
