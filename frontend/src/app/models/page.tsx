"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import {
  getToken,
  fetchFairnessReport,
  fetchRobustnessReport,
  fetchSurvivalAnalysis,
  fetchCohortAnalysis,
} from "@/lib/api";

// ── Type stubs ──────────────────────────────────────────────────────────────

interface FairnessCheck {
  attribute: string;
  demographic_parity_diff: number | null;
  equalized_odds_diff: number | null;
  disparate_impact_ratio: number | null;
  passed: boolean;
  failures: string[];
  segment_stats: Record<string, { predicted_churn_rate: number; count: number; actual_churn_rate?: number }>;
}

interface FairnessReport {
  summary: { total_checks: number; passed: number; failed: number; overall_pass: boolean };
  checks: FairnessCheck[];
  generated_at: string;
  model_version: string;
}

interface RobustnessReport {
  stability_score: number;
  perturbation_sensitivity: Record<string, number>;
  adversarial_max_shift: number;
  calibration_ece: number;
  calibration_bins: { bin_low: number; bin_high: number; avg_confidence: number; avg_accuracy: number; count: number }[];
  coverage_gaps: string[];
  overall_score: number;
  passed: boolean;
}

interface KMCurve {
  label: string;
  timeline: number[];
  survival: number[];
  ci_lower: number[];
  ci_upper: number[];
}

interface SurvivalData {
  kaplan_meier: { curves: KMCurve[]; median_survival: Record<string, number | null> };
  cox_ph: { concordance_index?: number; hazard_ratios?: Record<string, { hazard_ratio: number; p_value: number; significant: boolean }> };
  projection: { projection: { month: number; expected_churns: number }[]; total_expected_churns: number };
  dataset_size: number;
}

interface CohortData {
  cohort_matrix: { matrix: (number | null)[][]; cohorts: string[]; months: number[] };
  churn_rates: { avg_churn_by_month: number[]; months: number[] };
  dataset_size: number;
}

// ── Helpers ────────────────────────────────────────────────────────────────

function fmt(v: number | null | undefined, decimals = 3): string {
  if (v == null) return "N/A";
  return v.toFixed(decimals);
}

function PassBadge({ passed }: { passed: boolean }) {
  return (
    <span
      style={{
        padding: "2px 10px",
        borderRadius: "24px",
        fontSize: "0.75rem",
        fontWeight: 600,
        letterSpacing: "0.32px",
        backgroundColor: passed ? "#defbe6" : "#fff1f1",
        color: passed ? "#044317" : "#da1e28",
      }}
    >
      {passed ? "PASS" : "FAIL"}
    </span>
  );
}

type Tab = "fairness" | "robustness" | "survival" | "cohort";

// ── Page ────────────────────────────────────────────────────────────────────

export default function ModelsPage() {
  const router = useRouter();
  const [tab, setTab] = useState<Tab>("fairness");
  const [fairness, setFairness] = useState<FairnessReport | null>(null);
  const [robustness, setRobustness] = useState<RobustnessReport | null>(null);
  const [survival, setSurvival] = useState<SurvivalData | null>(null);
  const [cohort, setCohort] = useState<CohortData | null>(null);
  const [loading, setLoading] = useState<Record<Tab, boolean>>({
    fairness: false, robustness: false, survival: false, cohort: false,
  });
  const [errors, setErrors] = useState<Record<Tab, string | null>>({
    fairness: null, robustness: null, survival: null, cohort: null,
  });

  useEffect(() => {
    if (!getToken()) { router.replace("/login"); return; }
  }, [router]);

  function setTabLoading(t: Tab, v: boolean) {
    setLoading((prev) => ({ ...prev, [t]: v }));
  }
  function setTabError(t: Tab, msg: string | null) {
    setErrors((prev) => ({ ...prev, [t]: msg }));
  }

  async function loadFairness() {
    if (fairness) return;
    setTabLoading("fairness", true);
    try {
      const data = await fetchFairnessReport();
      setFairness(data as FairnessReport);
    } catch (e) {
      setTabError("fairness", e instanceof Error ? e.message : "Failed");
    } finally {
      setTabLoading("fairness", false);
    }
  }

  async function loadRobustness() {
    if (robustness) return;
    setTabLoading("robustness", true);
    try {
      const data = await fetchRobustnessReport();
      setRobustness(data as RobustnessReport);
    } catch (e) {
      setTabError("robustness", e instanceof Error ? e.message : "Failed");
    } finally {
      setTabLoading("robustness", false);
    }
  }

  async function loadSurvival() {
    if (survival) return;
    setTabLoading("survival", true);
    try {
      const data = await fetchSurvivalAnalysis();
      setSurvival(data as SurvivalData);
    } catch (e) {
      setTabError("survival", e instanceof Error ? e.message : "Failed");
    } finally {
      setTabLoading("survival", false);
    }
  }

  async function loadCohort() {
    if (cohort) return;
    setTabLoading("cohort", true);
    try {
      const data = await fetchCohortAnalysis();
      setCohort(data as CohortData);
    } catch (e) {
      setTabError("cohort", e instanceof Error ? e.message : "Failed");
    } finally {
      setTabLoading("cohort", false);
    }
  }

  function handleTab(t: Tab) {
    setTab(t);
    if (t === "fairness") loadFairness();
    if (t === "robustness") loadRobustness();
    if (t === "survival") loadSurvival();
    if (t === "cohort") loadCohort();
  }

  useEffect(() => { loadFairness(); }, []); // load default tab on mount

  const TABS: { key: Tab; label: string }[] = [
    { key: "fairness",   label: "Fairness" },
    { key: "robustness", label: "Robustness" },
    { key: "survival",   label: "Survival Analysis" },
    { key: "cohort",     label: "Cohort Retention" },
  ];

  return (
    <div className="space-y-0">
      {/* ── Header ──────────────────────────────────────── */}
      <div className="mb-8" style={{ paddingBottom: "16px", borderBottom: "1px solid #e0e0e0" }}>
        <h1 style={{ fontSize: "1.75rem", fontWeight: 400, color: "#161616", lineHeight: 1.29 }}>
          Model Intelligence
        </h1>
        <p style={{ fontSize: "0.875rem", color: "#525252", letterSpacing: "0.16px", marginTop: "4px" }}>
          Fairness · Robustness · Survival analysis · Cohort retention
        </p>
      </div>

      {/* ── Tab bar ─────────────────────────────────────── */}
      <div className="flex gap-px mb-px" style={{ backgroundColor: "#c6c6c6" }}>
        {TABS.map((t) => (
          <button
            key={t.key}
            onClick={() => handleTab(t.key)}
            style={{
              padding: "12px 24px",
              backgroundColor: tab === t.key ? "#ffffff" : "#f4f4f4",
              color: tab === t.key ? "#161616" : "#525252",
              fontSize: "0.875rem",
              fontWeight: tab === t.key ? 600 : 400,
              letterSpacing: "0.16px",
              border: "none",
              cursor: "pointer",
              borderBottom: tab === t.key ? "2px solid #0f62fe" : "2px solid transparent",
            }}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* ── Tab content ─────────────────────────────────── */}
      <div style={{ backgroundColor: "#ffffff" }}>
        {/* FAIRNESS TAB */}
        {tab === "fairness" && (
          loading.fairness ? <LoadingPane /> :
          errors.fairness  ? <ErrorPane msg={errors.fairness} /> :
          fairness ? <FairnessTab data={fairness} /> :
          <EmptyPane />
        )}

        {/* ROBUSTNESS TAB */}
        {tab === "robustness" && (
          loading.robustness ? <LoadingPane /> :
          errors.robustness  ? <ErrorPane msg={errors.robustness} /> :
          robustness ? <RobustnessTab data={robustness} /> :
          <EmptyPane />
        )}

        {/* SURVIVAL TAB */}
        {tab === "survival" && (
          loading.survival ? <LoadingPane /> :
          errors.survival  ? <ErrorPane msg={errors.survival} /> :
          survival ? <SurvivalTab data={survival} /> :
          <EmptyPane />
        )}

        {/* COHORT TAB */}
        {tab === "cohort" && (
          loading.cohort ? <LoadingPane /> :
          errors.cohort  ? <ErrorPane msg={errors.cohort} /> :
          cohort ? <CohortTab data={cohort} /> :
          <EmptyPane />
        )}
      </div>
    </div>
  );
}

// ── Sub-components ──────────────────────────────────────────────────────────

function LoadingPane() {
  return (
    <div className="flex items-center justify-center" style={{ height: "200px" }}>
      <p style={{ fontSize: "0.875rem", color: "#525252" }}>Loading…</p>
    </div>
  );
}
function ErrorPane({ msg }: { msg: string }) {
  return (
    <div style={{ padding: "24px", borderLeft: "4px solid #da1e28", backgroundColor: "#fff1f1", margin: "24px" }}>
      <p style={{ fontSize: "0.875rem", color: "#da1e28" }}>{msg}</p>
    </div>
  );
}
function EmptyPane() {
  return (
    <div className="flex items-center justify-center" style={{ height: "200px" }}>
      <p style={{ fontSize: "0.875rem", color: "#8d8d8d" }}>No data available</p>
    </div>
  );
}

// ── Fairness Tab ─────────────────────────────────────────────────────────────

function FairnessTab({ data }: { data: FairnessReport }) {
  const s = data.summary;
  return (
    <div>
      {/* Summary tiles */}
      <div className="grid grid-cols-4 gap-px" style={{ backgroundColor: "#c6c6c6" }}>
        {[
          { label: "OVERALL", value: s.overall_pass ? "PASS" : "FAIL", color: s.overall_pass ? "#24a148" : "#da1e28" },
          { label: "CHECKS RUN", value: s.total_checks, color: "#161616" },
          { label: "PASSED",  value: s.passed,  color: "#24a148" },
          { label: "FAILED",  value: s.failed,  color: s.failed > 0 ? "#da1e28" : "#24a148" },
        ].map(({ label, value, color }) => (
          <div key={label} style={{ backgroundColor: "#f4f4f4", padding: "20px" }}>
            <p style={{ fontSize: "0.75rem", color: "#525252", letterSpacing: "0.32px", textTransform: "uppercase", marginBottom: "6px" }}>{label}</p>
            <p style={{ fontSize: "1.5rem", fontWeight: 300, color }}>{value}</p>
          </div>
        ))}
      </div>

      {/* Checks */}
      {data.checks.map((check) => (
        <div key={check.attribute} style={{ borderBottom: "1px solid #e0e0e0" }}>
          <div className="flex items-center justify-between px-6 py-4" style={{ backgroundColor: "#f4f4f4" }}>
            <div>
              <p style={{ fontSize: "0.875rem", fontWeight: 600, color: "#161616" }}>
                {check.attribute}
              </p>
              {check.failures.length > 0 && (
                <p style={{ fontSize: "0.75rem", color: "#da1e28", marginTop: "2px" }}>
                  {check.failures[0]}
                </p>
              )}
            </div>
            <div className="flex items-center gap-6">
              <div style={{ textAlign: "right" }}>
                <p style={{ fontSize: "0.75rem", color: "#525252", letterSpacing: "0.32px" }}>DEMOG. PARITY Δ</p>
                <p style={{ fontSize: "1rem", fontWeight: 600, color: "#161616" }}>{fmt(check.demographic_parity_diff)}</p>
              </div>
              <div style={{ textAlign: "right" }}>
                <p style={{ fontSize: "0.75rem", color: "#525252", letterSpacing: "0.32px" }}>EQ. ODDS Δ</p>
                <p style={{ fontSize: "1rem", fontWeight: 600, color: "#161616" }}>{fmt(check.equalized_odds_diff)}</p>
              </div>
              <div style={{ textAlign: "right" }}>
                <p style={{ fontSize: "0.75rem", color: "#525252", letterSpacing: "0.32px" }}>DISPARATE IMPACT</p>
                <p style={{ fontSize: "1rem", fontWeight: 600, color: "#161616" }}>{fmt(check.disparate_impact_ratio)}</p>
              </div>
              <PassBadge passed={check.passed} />
            </div>
          </div>
          {/* Segment breakdown */}
          <div className="flex gap-px px-6 py-3" style={{ flexWrap: "wrap" }}>
            {Object.entries(check.segment_stats).map(([seg, stats]) => (
              <div key={seg} style={{ backgroundColor: "#f4f4f4", padding: "12px 16px", minWidth: "140px", marginRight: "1px", marginBottom: "1px" }}>
                <p style={{ fontSize: "0.75rem", fontWeight: 600, color: "#525252", letterSpacing: "0.32px", marginBottom: "4px" }}>{seg}</p>
                <p style={{ fontSize: "0.875rem", color: "#161616" }}>
                  {(stats.predicted_churn_rate * 100).toFixed(1)}% pred.
                </p>
                {stats.actual_churn_rate != null && (
                  <p style={{ fontSize: "0.75rem", color: "#8d8d8d" }}>
                    {(stats.actual_churn_rate * 100).toFixed(1)}% actual
                  </p>
                )}
                <p style={{ fontSize: "0.75rem", color: "#8d8d8d" }}>n={stats.count.toLocaleString()}</p>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Robustness Tab ────────────────────────────────────────────────────────────

function RobustnessTab({ data }: { data: RobustnessReport }) {
  const metrics = [
    { label: "OVERALL SCORE", value: (data.overall_score * 100).toFixed(1) + "%", color: data.overall_score >= 0.75 ? "#24a148" : "#da1e28" },
    { label: "STABILITY SCORE", value: (data.stability_score * 100).toFixed(1) + "%", color: "#161616" },
    { label: "CALIBRATION ECE", value: data.calibration_ece.toFixed(4), color: data.calibration_ece <= 0.10 ? "#24a148" : "#da1e28" },
    { label: "ADVERSARIAL SHIFT", value: data.adversarial_max_shift.toFixed(4), color: data.adversarial_max_shift <= 0.20 ? "#24a148" : "#da1e28" },
  ];

  const topFeatures = Object.entries(data.perturbation_sensitivity).slice(0, 8);
  const maxSens = Math.max(...topFeatures.map(([, v]) => v), 0.001);

  return (
    <div>
      {/* KPI tiles */}
      <div className="grid grid-cols-4 gap-px" style={{ backgroundColor: "#c6c6c6" }}>
        {metrics.map(({ label, value, color }) => (
          <div key={label} style={{ backgroundColor: "#f4f4f4", padding: "20px" }}>
            <p style={{ fontSize: "0.75rem", color: "#525252", letterSpacing: "0.32px", textTransform: "uppercase", marginBottom: "6px" }}>{label}</p>
            <p style={{ fontSize: "1.5rem", fontWeight: 300, color }}>{value}</p>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-px mt-px" style={{ backgroundColor: "#c6c6c6" }}>
        {/* Feature sensitivity */}
        <div style={{ backgroundColor: "#ffffff", padding: "24px" }}>
          <p style={{ fontSize: "0.875rem", fontWeight: 600, color: "#161616", marginBottom: "16px" }}>
            Feature Perturbation Sensitivity
          </p>
          {topFeatures.map(([feat, sens]) => (
            <div key={feat} className="mb-3">
              <div className="flex justify-between mb-1">
                <span style={{ fontSize: "0.75rem", color: "#525252" }}>{feat}</span>
                <span style={{ fontSize: "0.75rem", color: "#161616", fontWeight: 600 }}>{sens.toFixed(4)}</span>
              </div>
              <div style={{ height: "4px", backgroundColor: "#e0e0e0" }}>
                <div
                  style={{
                    height: "4px",
                    width: `${(sens / maxSens) * 100}%`,
                    backgroundColor: sens > maxSens * 0.6 ? "#da1e28" : "#0f62fe",
                  }}
                />
              </div>
            </div>
          ))}
        </div>

        {/* Calibration reliability diagram */}
        <div style={{ backgroundColor: "#ffffff", padding: "24px" }}>
          <p style={{ fontSize: "0.875rem", fontWeight: 600, color: "#161616", marginBottom: "16px" }}>
            Calibration Reliability (ECE = {data.calibration_ece.toFixed(4)})
          </p>
          {data.calibration_bins.length > 0 ? (
            <div>
              {/* Table header */}
              <div className="grid" style={{ gridTemplateColumns: "1fr 1fr 1fr 1fr", fontSize: "0.75rem", color: "#525252", letterSpacing: "0.32px", marginBottom: "8px" }}>
                <span>Bin</span><span>Confidence</span><span>Accuracy</span><span>Gap</span>
              </div>
              {data.calibration_bins.map((bin) => {
                const gap = Math.abs(bin.avg_confidence - bin.avg_accuracy);
                return (
                  <div
                    key={bin.bin_low}
                    className="grid py-1"
                    style={{
                      gridTemplateColumns: "1fr 1fr 1fr 1fr",
                      fontSize: "0.75rem",
                      borderBottom: "1px solid #f4f4f4",
                    }}
                  >
                    <span style={{ color: "#525252" }}>{bin.bin_low}–{bin.bin_high}</span>
                    <span style={{ color: "#161616" }}>{bin.avg_confidence.toFixed(3)}</span>
                    <span style={{ color: "#161616" }}>{bin.avg_accuracy.toFixed(3)}</span>
                    <span style={{ color: gap > 0.1 ? "#da1e28" : "#24a148", fontWeight: 600 }}>
                      {gap.toFixed(3)}
                    </span>
                  </div>
                );
              })}
            </div>
          ) : (
            <p style={{ fontSize: "0.875rem", color: "#8d8d8d" }}>Calibration data not available (requires ground-truth labels)</p>
          )}
        </div>
      </div>

      {/* Coverage gaps */}
      {data.coverage_gaps.length > 0 && (
        <div style={{ backgroundColor: "#fff2e8", borderLeft: "4px solid #ba4e00", padding: "16px", margin: "1px" }}>
          <p style={{ fontSize: "0.875rem", fontWeight: 600, color: "#ba4e00", marginBottom: "8px" }}>
            Under-represented Segments ({data.coverage_gaps.length})
          </p>
          <div className="flex flex-wrap gap-2">
            {data.coverage_gaps.map((g) => (
              <span key={g} style={{ fontSize: "0.75rem", color: "#ba4e00", backgroundColor: "#fff2e8", border: "1px solid #ba4e00", padding: "2px 8px", borderRadius: "24px" }}>
                {g}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Survival Tab ──────────────────────────────────────────────────────────────

function SurvivalTab({ data }: { data: SurvivalData }) {
  const km = data.kaplan_meier;
  const cph = data.cox_ph;
  const proj = data.projection;

  // Sample every 6 months for display
  const displayMonths = [0, 6, 12, 18, 24, 36, 48, 60];

  return (
    <div>
      {/* Median survival */}
      <div className="grid gap-px" style={{ backgroundColor: "#c6c6c6", gridTemplateColumns: `repeat(${Object.keys(km.median_survival).length + 1}, 1fr)` }}>
        <div style={{ backgroundColor: "#f4f4f4", padding: "20px" }}>
          <p style={{ fontSize: "0.75rem", color: "#525252", letterSpacing: "0.32px", textTransform: "uppercase", marginBottom: "6px" }}>DATASET SIZE</p>
          <p style={{ fontSize: "1.5rem", fontWeight: 300, color: "#161616" }}>{data.dataset_size.toLocaleString()}</p>
        </div>
        {Object.entries(km.median_survival).map(([label, val]) => (
          <div key={label} style={{ backgroundColor: "#f4f4f4", padding: "20px" }}>
            <p style={{ fontSize: "0.75rem", color: "#525252", letterSpacing: "0.32px", textTransform: "uppercase", marginBottom: "6px" }}>
              MEDIAN {label.toUpperCase()} SURVIVAL
            </p>
            <p style={{ fontSize: "1.5rem", fontWeight: 300, color: "#0f62fe" }}>
              {val != null ? `${val}mo` : "Not reached"}
            </p>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-px mt-px" style={{ backgroundColor: "#c6c6c6" }}>
        {/* KM curves table */}
        <div style={{ backgroundColor: "#ffffff", padding: "24px" }}>
          <p style={{ fontSize: "0.875rem", fontWeight: 600, color: "#161616", marginBottom: "16px" }}>
            Kaplan-Meier Survival at Key Months
          </p>
          <div>
            {/* Header */}
            <div className="flex" style={{ borderBottom: "1px solid #e0e0e0", paddingBottom: "8px", marginBottom: "4px" }}>
              <span style={{ flex: 1, fontSize: "0.75rem", color: "#525252", letterSpacing: "0.32px" }}>CURVE</span>
              {displayMonths.map((m) => (
                <span key={m} style={{ width: "48px", textAlign: "right", fontSize: "0.75rem", color: "#525252", letterSpacing: "0.32px" }}>t={m}</span>
              ))}
            </div>
            {km.curves.map((curve) => (
              <div key={curve.label} className="flex py-2" style={{ borderBottom: "1px solid #f4f4f4" }}>
                <span style={{ flex: 1, fontSize: "0.875rem", color: "#161616" }}>{curve.label}</span>
                {displayMonths.map((m) => {
                  const idx = curve.timeline.indexOf(m);
                  const val = idx >= 0 ? curve.survival[idx] : null;
                  return (
                    <span key={m} style={{ width: "48px", textAlign: "right", fontSize: "0.875rem", color: val != null && val < 0.5 ? "#da1e28" : "#161616" }}>
                      {val != null ? (val * 100).toFixed(0) + "%" : "—"}
                    </span>
                  );
                })}
              </div>
            ))}
          </div>
        </div>

        {/* Cox PH hazard ratios */}
        <div style={{ backgroundColor: "#ffffff", padding: "24px" }}>
          <p style={{ fontSize: "0.875rem", fontWeight: 600, color: "#161616", marginBottom: "4px" }}>
            Cox PH Hazard Ratios
          </p>
          {cph.concordance_index != null && (
            <p style={{ fontSize: "0.75rem", color: "#525252", marginBottom: "16px" }}>
              Concordance index: <strong>{cph.concordance_index.toFixed(4)}</strong>
            </p>
          )}
          {cph.hazard_ratios && Object.keys(cph.hazard_ratios).length > 0 ? (
            Object.entries(cph.hazard_ratios)
              .sort(([, a], [, b]) => Math.abs(Math.log(b.hazard_ratio)) - Math.abs(Math.log(a.hazard_ratio)))
              .slice(0, 8)
              .map(([feat, hr]) => (
                <div key={feat} className="flex items-center justify-between py-2" style={{ borderBottom: "1px solid #f4f4f4" }}>
                  <span style={{ fontSize: "0.875rem", color: "#161616" }}>{feat}</span>
                  <div className="flex items-center gap-3">
                    <span style={{ fontSize: "0.875rem", fontWeight: 600, color: hr.hazard_ratio > 1 ? "#da1e28" : "#24a148" }}>
                      {hr.hazard_ratio.toFixed(3)}
                    </span>
                    {hr.significant && (
                      <span style={{ fontSize: "0.65rem", padding: "1px 6px", borderRadius: "24px", backgroundColor: "#edf5ff", color: "#0043ce" }}>
                        p&lt;0.05
                      </span>
                    )}
                  </div>
                </div>
              ))
          ) : (
            <p style={{ fontSize: "0.875rem", color: "#8d8d8d" }}>{(cph as Record<string, unknown>).error as string ?? "No Cox PH data"}</p>
          )}
        </div>
      </div>

      {/* Churn projection */}
      {proj.projection && (
        <div style={{ backgroundColor: "#f4f4f4", padding: "24px", marginTop: "1px" }}>
          <p style={{ fontSize: "0.875rem", fontWeight: 600, color: "#161616", marginBottom: "4px" }}>
            12-Month Churn Projection
          </p>
          <p style={{ fontSize: "0.75rem", color: "#525252", marginBottom: "16px" }}>
            Total expected churns: <strong>{proj.total_expected_churns.toFixed(0)}</strong>
          </p>
          <div className="flex gap-px" style={{ backgroundColor: "#c6c6c6" }}>
            {proj.projection.slice(0, 12).map((p) => {
              const maxChurns = Math.max(...proj.projection.map((x) => x.expected_churns));
              const pct = maxChurns > 0 ? (p.expected_churns / maxChurns) * 100 : 0;
              return (
                <div key={p.month} style={{ flex: 1, backgroundColor: "#ffffff", padding: "12px 4px", textAlign: "center" }}>
                  <div style={{ height: "60px", display: "flex", alignItems: "flex-end", justifyContent: "center", marginBottom: "4px" }}>
                    <div style={{ width: "100%", height: `${pct}%`, backgroundColor: "#0f62fe", minHeight: "4px" }} />
                  </div>
                  <p style={{ fontSize: "0.65rem", color: "#525252" }}>M{p.month}</p>
                  <p style={{ fontSize: "0.75rem", color: "#161616", fontWeight: 600 }}>{p.expected_churns.toFixed(0)}</p>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Cohort Tab ────────────────────────────────────────────────────────────────

function CohortTab({ data }: { data: CohortData }) {
  const { cohort_matrix, churn_rates } = data;
  const { matrix, cohorts, months } = cohort_matrix;

  function retentionColor(v: number | null): string {
    if (v == null) return "#f4f4f4";
    if (v >= 0.9) return "#defbe6";
    if (v >= 0.7) return "#edf5ff";
    if (v >= 0.5) return "#fcf4d6";
    if (v >= 0.3) return "#fff2e8";
    return "#fff1f1";
  }

  const displayMonths = months.filter((m) => m % 3 === 0).slice(0, 12);

  return (
    <div>
      {/* Heatmap */}
      <div style={{ padding: "24px", overflowX: "auto" }}>
        <p style={{ fontSize: "0.875rem", fontWeight: 600, color: "#161616", marginBottom: "16px" }}>
          Cohort Retention Heatmap (% still active)
        </p>
        <table style={{ borderCollapse: "collapse", fontSize: "0.75rem", minWidth: "100%" }}>
          <thead>
            <tr>
              <th style={{ padding: "8px 12px", backgroundColor: "#161616", color: "#ffffff", fontWeight: 400, textAlign: "left", whiteSpace: "nowrap" }}>
                Cohort
              </th>
              {displayMonths.map((m) => (
                <th key={m} style={{ padding: "8px 10px", backgroundColor: "#161616", color: "#ffffff", fontWeight: 400, textAlign: "center" }}>
                  M{m}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.slice(0, 12).map((row, ci) => (
              <tr key={cohorts[ci]}>
                <td style={{ padding: "8px 12px", backgroundColor: "#f4f4f4", color: "#525252", fontWeight: 600, whiteSpace: "nowrap", borderBottom: "1px solid #e0e0e0" }}>
                  {cohorts[ci]}
                </td>
                {displayMonths.map((m) => {
                  const v = row[m] ?? null;
                  return (
                    <td
                      key={m}
                      style={{
                        padding: "8px 10px",
                        textAlign: "center",
                        backgroundColor: retentionColor(v),
                        color: "#161616",
                        borderBottom: "1px solid #e0e0e0",
                        fontWeight: v != null && v < 0.5 ? 600 : 400,
                      }}
                    >
                      {v != null ? `${(v * 100).toFixed(0)}%` : "—"}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Period churn rates */}
      {churn_rates.avg_churn_by_month.length > 0 && (
        <div style={{ backgroundColor: "#f4f4f4", padding: "24px", margin: "1px" }}>
          <p style={{ fontSize: "0.875rem", fontWeight: 600, color: "#161616", marginBottom: "16px" }}>
            Average Period Churn Rate by Month of Life
          </p>
          <div className="flex gap-px" style={{ backgroundColor: "#c6c6c6" }}>
            {churn_rates.avg_churn_by_month.slice(1, 13).map((rate, i) => {
              const maxRate = Math.max(...churn_rates.avg_churn_by_month, 0.001);
              const pct = (rate / maxRate) * 100;
              return (
                <div key={i} style={{ flex: 1, backgroundColor: "#ffffff", padding: "8px 4px", textAlign: "center" }}>
                  <div style={{ height: "48px", display: "flex", alignItems: "flex-end", justifyContent: "center", marginBottom: "4px" }}>
                    <div style={{ width: "100%", height: `${pct}%`, backgroundColor: rate > maxRate * 0.7 ? "#da1e28" : "#0f62fe", minHeight: "3px" }} />
                  </div>
                  <p style={{ fontSize: "0.65rem", color: "#525252" }}>M{churn_rates.months[i + 1] ?? i + 2}</p>
                  <p style={{ fontSize: "0.7rem", color: "#161616", fontWeight: 600 }}>{(rate * 100).toFixed(1)}%</p>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
