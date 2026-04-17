"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { getToken, optimizeRetention } from "@/lib/api";

interface OptimizeResult {
  budget: number;
  customers_considered: number;
  total_cost: number;
  total_expected_revenue_saved: number;
  roi: number;
  plan: {
    customer_id: string;
    action: string;
    cost: number;
    prob_reduction: number;
    expected_revenue_saved: number;
  }[];
  solver_status?: string;
  error?: string;
}

export default function OptimizationPage() {
  const router = useRouter();
  const [budget, setBudget] = useState(50000);
  const [maxActions, setMaxActions] = useState(2);
  const [riskTiers, setRiskTiers] = useState<string[]>(["CRITICAL", "HIGH"]);
  const [result, setResult] = useState<OptimizeResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [ran, setRan] = useState(false);

  useEffect(() => {
    if (!getToken()) { router.replace("/login"); }
  }, [router]);

  function toggleTier(tier: string) {
    setRiskTiers((prev) =>
      prev.includes(tier) ? prev.filter((t) => t !== tier) : [...prev, tier]
    );
  }

  async function runOptimizer() {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await optimizeRetention({
        total_budget: budget,
        max_actions_per_customer: maxActions,
        risk_tier_filter: riskTiers,
      });
      setResult(res as unknown as OptimizeResult);
      setRan(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Optimizer failed");
    } finally {
      setLoading(false);
    }
  }

  const TIER_COLORS: Record<string, { bg: string; color: string }> = {
    CRITICAL: { bg: "#fff1f1", color: "#da1e28" },
    HIGH: { bg: "#fff2e8", color: "#ba4e00" },
    MEDIUM: { bg: "#fcf4d6", color: "#684e00" },
    LOW: { bg: "#defbe6", color: "#044317" },
  };

  return (
    <div className="space-y-0">
      {/* ── Page header ─────────────────────────────────── */}
      <div className="mb-8" style={{ paddingBottom: "16px", borderBottom: "1px solid #e0e0e0" }}>
        <h1 style={{ fontSize: "1.75rem", fontWeight: 400, color: "#161616", lineHeight: 1.29 }}>
          Retention Budget Optimizer
        </h1>
        <p style={{ fontSize: "0.875rem", color: "#525252", letterSpacing: "0.16px", marginTop: "4px" }}>
          Knapsack integer programming — maximise expected retained revenue within budget constraints
        </p>
      </div>

      {/* ── Config panel ─────────────────────────────────── */}
      <div style={{ backgroundColor: "#f4f4f4", padding: "24px", marginBottom: "1px" }}>
        <p style={{ fontSize: "0.875rem", fontWeight: 600, color: "#161616", marginBottom: "20px", letterSpacing: "0.16px" }}>
          Optimizer Settings
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Budget */}
          <div>
            <label className="cds-label">
              Total Budget —{" "}
              <span style={{ color: "#0f62fe", fontWeight: 600 }}>
                ${budget.toLocaleString()}
              </span>
            </label>
            <input
              type="range"
              min={5000}
              max={500000}
              step={5000}
              value={budget}
              onChange={(e) => setBudget(parseInt(e.target.value))}
              className="w-full h-1 mt-2"
              style={{ accentColor: "#0f62fe" }}
            />
            <div className="flex justify-between mt-1" style={{ fontSize: "0.75rem", color: "#8d8d8d" }}>
              <span>$5k</span><span>$500k</span>
            </div>
          </div>

          {/* Max actions */}
          <div>
            <label className="cds-label">Max Actions per Customer</label>
            <div className="flex gap-px mt-2" style={{ backgroundColor: "#c6c6c6" }}>
              {[1, 2, 3].map((n) => (
                <button
                  key={n}
                  onClick={() => setMaxActions(n)}
                  style={{
                    flex: 1,
                    height: "40px",
                    backgroundColor: maxActions === n ? "#0f62fe" : "#ffffff",
                    color: maxActions === n ? "#ffffff" : "#161616",
                    fontSize: "0.875rem",
                    border: "none",
                    cursor: "pointer",
                  }}
                >
                  {n}
                </button>
              ))}
            </div>
          </div>

          {/* Risk tiers */}
          <div>
            <label className="cds-label">Risk Tiers to Include</label>
            <div className="flex flex-wrap gap-2 mt-2">
              {["CRITICAL", "HIGH", "MEDIUM", "LOW"].map((tier) => {
                const selected = riskTiers.includes(tier);
                const s = TIER_COLORS[tier];
                return (
                  <button
                    key={tier}
                    onClick={() => toggleTier(tier)}
                    style={{
                      padding: "4px 12px",
                      borderRadius: "24px",
                      fontSize: "0.75rem",
                      letterSpacing: "0.32px",
                      border: `2px solid ${selected ? s.color : "#c6c6c6"}`,
                      backgroundColor: selected ? s.bg : "#ffffff",
                      color: selected ? s.color : "#525252",
                      cursor: "pointer",
                    }}
                  >
                    {tier}
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        <div className="mt-6">
          <button
            onClick={runOptimizer}
            disabled={loading || riskTiers.length === 0}
            className="cds-btn cds-btn--primary"
            style={{ minWidth: "180px", justifyContent: "center", padding: "0 24px" }}
          >
            {loading ? "Optimizing…" : "Run Optimizer"}
          </button>
        </div>
      </div>

      {/* ── Error ────────────────────────────────────────── */}
      {error && (
        <div style={{ backgroundColor: "#fff1f1", borderLeft: "4px solid #da1e28", padding: "16px", marginTop: "1px" }}>
          <p style={{ fontSize: "0.875rem", color: "#da1e28" }}>{error}</p>
        </div>
      )}

      {/* ── Results ──────────────────────────────────────── */}
      {result && !result.error && (
        <>
          {/* KPI tiles */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-px mt-1" style={{ backgroundColor: "#c6c6c6" }}>
            {[
              { label: "BUDGET ALLOCATED", value: `$${result.total_cost?.toLocaleString() ?? 0}`, color: "#0f62fe" },
              { label: "CUSTOMERS COVERED", value: result.plan?.length ?? 0, color: "#161616" },
              { label: "EXPECTED REVENUE SAVED", value: `$${result.total_expected_revenue_saved?.toLocaleString() ?? 0}`, color: "#24a148" },
              { label: "ROI", value: result.roi != null ? `${result.roi.toFixed(1)}×` : "—", color: "#24a148" },
            ].map(({ label, value, color }) => (
              <div key={label} style={{ backgroundColor: "#f4f4f4", padding: "24px" }}>
                <p style={{ fontSize: "0.75rem", color: "#525252", letterSpacing: "0.32px", textTransform: "uppercase", marginBottom: "8px" }}>
                  {label}
                </p>
                <p style={{ fontSize: "2rem", fontWeight: 300, color, lineHeight: 1.25 }}>{value}</p>
              </div>
            ))}
          </div>

          {/* Plan table */}
          <div style={{ backgroundColor: "#ffffff", marginTop: "8px" }}>
            <div className="flex items-center justify-between px-6 py-4" style={{ borderBottom: "1px solid #e0e0e0" }}>
              <p style={{ fontSize: "1rem", fontWeight: 600, color: "#161616" }}>Action Plan</p>
              <span style={{ fontSize: "0.75rem", color: "#525252", backgroundColor: "#f4f4f4", padding: "4px 10px", letterSpacing: "0.32px" }}>
                {result.plan?.length ?? 0} assignments · solver: {result.solver_status ?? "optimal"}
              </span>
            </div>

            {result.plan && result.plan.length > 0 ? (
              <div>
                {/* Header row */}
                <div
                  className="grid px-6 py-2"
                  style={{
                    gridTemplateColumns: "2fr 2fr 1fr 1fr 1fr",
                    backgroundColor: "#e0e0e0",
                    fontSize: "0.75rem",
                    color: "#525252",
                    letterSpacing: "0.32px",
                    textTransform: "uppercase",
                  }}
                >
                  <span>Customer</span>
                  <span>Action</span>
                  <span>Cost</span>
                  <span>Churn Δ</span>
                  <span>Rev. Saved</span>
                </div>
                {result.plan.map((row, i) => (
                  <div
                    key={`${row.customer_id}-${i}`}
                    className="grid px-6 py-3"
                    style={{
                      gridTemplateColumns: "2fr 2fr 1fr 1fr 1fr",
                      backgroundColor: i % 2 === 0 ? "#ffffff" : "#f4f4f4",
                      borderBottom: "1px solid #e0e0e0",
                      fontSize: "0.875rem",
                      alignItems: "center",
                    }}
                  >
                    <span className="font-mono" style={{ color: "#0f62fe", fontSize: "0.75rem" }}>
                      {row.customer_id}
                    </span>
                    <span style={{ color: "#161616" }}>
                      {row.action?.replace(/_/g, " ")}
                    </span>
                    <span style={{ color: "#525252" }}>${row.cost}</span>
                    <span style={{ color: "#da1e28" }}>−{(row.prob_reduction * 100).toFixed(0)}%</span>
                    <span style={{ color: "#24a148", fontWeight: 600 }}>
                      ${row.expected_revenue_saved?.toLocaleString() ?? "—"}
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="px-6 py-12 text-center">
                <p style={{ fontSize: "0.875rem", color: "#8d8d8d" }}>No actions assigned within budget</p>
              </div>
            )}
          </div>
        </>
      )}

      {result?.error && (
        <div style={{ backgroundColor: "#fff2e8", borderLeft: "4px solid #ba4e00", padding: "16px", marginTop: "1px" }}>
          <p style={{ fontSize: "0.875rem", color: "#ba4e00" }}>{result.error}</p>
        </div>
      )}

      {!ran && !loading && (
        <div style={{ backgroundColor: "#f4f4f4", padding: "48px", textAlign: "center", marginTop: "1px" }}>
          <p style={{ fontSize: "0.875rem", color: "#8d8d8d", letterSpacing: "0.16px" }}>
            Configure settings above and click <strong>Run Optimizer</strong> to generate the retention action plan.
          </p>
        </div>
      )}
    </div>
  );
}
