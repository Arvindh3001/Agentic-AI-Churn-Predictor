"""
Fairness report generator.

Produces a structured JSON report and an HTML summary from BiasCheckResult
objects returned by bias_detector.check_bias().
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from src.fairness.bias_detector import BiasCheckResult, bias_summary


# ── JSON report ───────────────────────────────────────────────────────────────

def build_json_report(
    results: list[BiasCheckResult],
    model_version: str = "unknown",
    dataset_size: int = 0,
) -> dict[str, Any]:
    """
    Build a machine-readable fairness report dict.

    Suitable for serialisation to JSON or returning from a FastAPI endpoint.
    """
    summary = bias_summary(results)
    return {
        "report_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "dataset_size": dataset_size,
        "summary": summary,
        "checks": [r.to_dict() for r in results],
        "thresholds": {
            "demographic_parity_diff": 0.10,
            "equalized_odds_diff": 0.10,
            "disparate_impact_ratio": 0.80,
        },
    }


# ── HTML report ───────────────────────────────────────────────────────────────

_PASS_STYLE = "color:#24a148;font-weight:600"
_FAIL_STYLE = "color:#da1e28;font-weight:600"


def _status_badge(passed: bool) -> str:
    style = _PASS_STYLE if passed else _FAIL_STYLE
    label = "PASS" if passed else "FAIL"
    return f'<span style="{style}">{label}</span>'


def build_html_report(
    results: list[BiasCheckResult],
    model_version: str = "unknown",
    dataset_size: int = 0,
) -> str:
    """
    Build a styled HTML fairness report string.
    """
    summary = bias_summary(results)
    overall_style = _PASS_STYLE if summary["overall_pass"] else _FAIL_STYLE
    rows = ""
    for r in results:
        seg_html = ""
        for seg, stats in r.segment_stats.items():
            seg_html += f"<li><strong>{seg}</strong>: predicted churn rate {stats.get('predicted_churn_rate', 0):.1%}"
            if "actual_churn_rate" in stats:
                seg_html += f", actual {stats['actual_churn_rate']:.1%}"
            seg_html += f" (n={int(stats.get('count', 0))})</li>"

        failures_html = ""
        if r.failures:
            failures_html = "<ul>" + "".join(f"<li>{f}</li>" for f in r.failures) + "</ul>"

        dp = f"{r.demographic_parity_diff:.4f}" if r.demographic_parity_diff is not None else "N/A"
        eo = f"{r.equalized_odds_diff:.4f}" if r.equalized_odds_diff is not None else "N/A"
        di = f"{r.disparate_impact_ratio:.4f}" if r.disparate_impact_ratio is not None else "N/A"

        rows += f"""
        <tr>
          <td style="padding:12px 16px;font-weight:600">{r.attribute}</td>
          <td style="padding:12px 16px">{dp}</td>
          <td style="padding:12px 16px">{eo}</td>
          <td style="padding:12px 16px">{di}</td>
          <td style="padding:12px 16px">{_status_badge(r.passed)}</td>
          <td style="padding:12px 16px;font-size:0.8rem">
            {failures_html or '<span style="color:#525252">None</span>'}
          </td>
        </tr>
        <tr style="background:#f4f4f4">
          <td colspan="6" style="padding:8px 16px;font-size:0.8rem">
            <strong>Segment breakdown:</strong><ul style="margin:4px 0">{seg_html}</ul>
          </td>
        </tr>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>Fairness Report — {model_version}</title>
  <style>
    body {{font-family:"IBM Plex Sans",Helvetica Neue,Arial,sans-serif;margin:40px;color:#161616;background:#fff}}
    h1 {{font-weight:300;font-size:2rem;margin-bottom:4px}}
    .meta {{font-size:0.875rem;color:#525252;margin-bottom:32px}}
    .summary-grid {{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:#c6c6c6;margin-bottom:32px}}
    .tile {{background:#f4f4f4;padding:20px}}
    .tile-label {{font-size:0.75rem;text-transform:uppercase;letter-spacing:0.32px;color:#525252;margin-bottom:6px}}
    .tile-value {{font-size:1.75rem;font-weight:300}}
    table {{width:100%;border-collapse:collapse}}
    th {{background:#161616;color:#fff;padding:12px 16px;text-align:left;font-weight:400;font-size:0.875rem}}
    tr:hover td {{background:#edf5ff}}
  </style>
</head>
<body>
  <h1>Fairness &amp; Bias Report</h1>
  <p class="meta">Model: <strong>{model_version}</strong> · Dataset: <strong>{dataset_size:,}</strong> records ·
     Generated: <strong>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</strong></p>

  <div class="summary-grid">
    <div class="tile"><div class="tile-label">Overall</div>
      <div class="tile-value" style="{overall_style}">{'PASS' if summary['overall_pass'] else 'FAIL'}</div></div>
    <div class="tile"><div class="tile-label">Checks Run</div>
      <div class="tile-value">{summary['total_checks']}</div></div>
    <div class="tile"><div class="tile-label">Passed</div>
      <div class="tile-value" style="{_PASS_STYLE}">{summary['passed']}</div></div>
    <div class="tile"><div class="tile-label">Failed</div>
      <div class="tile-value" style="{_FAIL_STYLE if summary['failed'] > 0 else _PASS_STYLE}">{summary['failed']}</div></div>
  </div>

  <table>
    <thead>
      <tr>
        <th>Attribute</th>
        <th>Demographic Parity Δ</th>
        <th>Equalized Odds Δ</th>
        <th>Disparate Impact</th>
        <th>Status</th>
        <th>Failures</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>

  <p style="margin-top:32px;font-size:0.75rem;color:#8d8d8d">
    Thresholds: Demographic Parity &lt; 0.10 · Equalized Odds &lt; 0.10 · Disparate Impact &gt; 0.80 (EEOC 4/5ths rule)
  </p>
</body>
</html>"""
