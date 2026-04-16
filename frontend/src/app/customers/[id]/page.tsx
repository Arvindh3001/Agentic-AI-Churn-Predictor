"use client";

import { useEffect, useState } from "react";
import { useRouter, useParams } from "next/navigation";
import {
  fetchCustomer,
  analyseCustomer,
  recordHITLDecision,
  submitFeedback,
  getToken,
} from "@/lib/api";
import type { Customer, AnalysisResult, RiskFactor, RetentionAction } from "@/types";
import RiskBadge from "@/components/RiskBadge";
import type { RiskTier } from "@/types";

const STEPS = [
  { key: "data_intelligence",    label: "Data Intelligence" },
  { key: "prediction",           label: "Churn Prediction" },
  { key: "explanation",          label: "SHAP Explanation" },
  { key: "counterfactual",       label: "Counterfactual Analysis" },
  { key: "retention_strategist", label: "Retention Strategy" },
  { key: "hitl",                 label: "HITL Review" },
];

function useAnimatedSteps(running: boolean) {
  const [idx, setIdx] = useState(-1);
  useEffect(() => {
    if (!running) { setIdx(-1); return; }
    setIdx(0);
    const interval = setInterval(() =>
      setIdx((prev) => (prev < STEPS.length - 1 ? prev + 1 : prev)), 5000
    );
    return () => clearInterval(interval);
  }, [running]);
  return idx;
}

// Carbon Progress Step
function ProgressStep({
  state,
  label,
  isLast,
  statusText,
}: {
  state: "complete" | "active" | "pending";
  label: string;
  isLast: boolean;
  statusText?: string;
}) {
  return (
    <div className="flex gap-4">
      <div className="flex flex-col items-center" style={{ width: "24px" }}>
        {/* Circle */}
        <span
          className={state === "active" ? "animate-pulse" : ""}
          style={{
            width: "24px",
            height: "24px",
            borderRadius: "50%",
            flexShrink: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: "12px",
            fontWeight: 600,
            backgroundColor:
              state === "complete" ? "#24a148" :
              state === "active"   ? "#0f62fe" : "#ffffff",
            border:
              state === "pending" ? "2px solid #c6c6c6" : "none",
            color:
              state === "pending" ? "#8d8d8d" : "#ffffff",
            position: "relative",
            zIndex: 1,
          }}
        >
          {state === "complete" ? "✓" : state === "active" ? "◉" : ""}
        </span>
        {/* Connector */}
        {!isLast && (
          <div
            style={{
              width: "2px",
              flex: 1,
              minHeight: "16px",
              backgroundColor: state === "complete" ? "#24a148" : "#e0e0e0",
              margin: "4px 0",
            }}
          />
        )}
      </div>
      <div className={`flex items-center justify-between flex-1 ${isLast ? "" : "pb-4"}`}>
        <span
          style={{
            fontSize: "0.875rem",
            letterSpacing: "0.16px",
            color:
              state === "complete" ? "#161616" :
              state === "active"   ? "#0f62fe" : "#8d8d8d",
            fontWeight: state === "pending" ? 400 : 600,
          }}
        >
          {label}
        </span>
        {statusText && (
          <span
            className={state === "active" ? "animate-pulse" : ""}
            style={{
              fontSize: "0.75rem",
              letterSpacing: "0.32px",
              color: state === "complete" ? "#24a148" : "#0f62fe",
            }}
          >
            {statusText}
          </span>
        )}
      </div>
    </div>
  );
}

export default function CustomerPage() {
  const router = useRouter();
  const { id } = useParams<{ id: string }>();

  const [customer, setCustomer] = useState<Customer | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [running, setRunning] = useState(false);
  const [feedbackOutcome, setFeedbackOutcome] = useState<"retained" | "churned" | "unknown" | "">("");
  const [feedbackNotes, setFeedbackNotes] = useState("");
  const [feedbackSent, setFeedbackSent] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const animatedStep = useAnimatedSteps(running);

  useEffect(() => {
    if (!getToken()) { router.replace("/login"); return; }
    fetchCustomer(id)
      .then(({ customer }) => setCustomer(customer))
      .catch((e) => setError(e.message));
  }, [id, router]);

  async function startAnalysis() {
    setError(null);
    setResult(null);
    setFeedbackSent(false);
    setRunning(true);
    try {
      const res = await analyseCustomer(id, false);
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Pipeline failed");
    } finally {
      setRunning(false);
    }
  }

  async function handleHITLDecision(decision: "approved" | "rejected") {
    if (!result?.run_id) return;
    try {
      await recordHITLDecision(result.run_id, decision, "Decision via dashboard");
    } catch (e) {
      setError(e instanceof Error ? e.message : "HITL action failed");
    }
  }

  async function handleFeedback(e: React.FormEvent) {
    e.preventDefault();
    if (!result?.run_id || !feedbackOutcome) return;
    try {
      await submitFeedback({
        run_id: result.run_id,
        customer_id: id,
        outcome: feedbackOutcome,
        notes: feedbackNotes,
        submitted_by: "csm_dashboard",
      });
      setFeedbackSent(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Feedback failed");
    }
  }

  if (!customer) {
    return (
      <div className="flex items-center justify-center h-64">
        <p style={{ fontSize: "0.875rem", color: "#525252", letterSpacing: "0.16px" }}>
          {error ?? "Loading customer…"}
        </p>
      </div>
    );
  }

  const completedSteps = new Set(result?.completed_steps ?? []);
  const prediction = result?.churn_probability != null ? result : null;
  const pendingHITL = result?.retention_plan?.pending_hitl && !result?.hitl_decision;
  const prob = prediction?.churn_probability ?? 0;

  return (
    <div style={{ maxWidth: "768px" }} className="space-y-0">
      {/* ── Breadcrumb ──────────────────────────────────────────── */}
      <div
        className="flex items-center gap-2 mb-6"
        style={{ fontSize: "0.75rem", color: "#525252", letterSpacing: "0.32px" }}
      >
        <button
          onClick={() => router.push("/")}
          style={{ color: "#0f62fe", letterSpacing: "0.32px" }}
          className="hover:underline"
        >
          Dashboard
        </button>
        <span style={{ color: "#c6c6c6" }}>›</span>
        <span className="font-mono" style={{ color: "#161616" }}>{id}</span>
      </div>

      {/* ── Customer card ───────────────────────────────────────── */}
      <div style={{ backgroundColor: "#f4f4f4", padding: "24px", marginBottom: "1px" }}>
        <div className="flex items-start justify-between mb-4">
          <div>
            <h1
              className="font-mono"
              style={{ fontSize: "1.25rem", fontWeight: 600, color: "#161616", lineHeight: 1.4 }}
            >
              {id}
            </h1>
            <p style={{ fontSize: "0.75rem", color: "#525252", letterSpacing: "0.32px", marginTop: "4px" }}>
              {customer.contract_type} · {customer.tenure_months} months tenure
            </p>
          </div>
          <RiskBadge tier={customer.churn === 1 ? "HIGH" : "LOW"} size="sm" />
        </div>

        {/* Stats grid — white tiles on gray-10 */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-px" style={{ backgroundColor: "#c6c6c6" }}>
          {[
            { label: "Monthly Charges",  value: `$${customer.monthly_charges.toFixed(0)}` },
            { label: "Support Tickets",  value: String(customer.num_support_tickets_30d ?? "—") },
            { label: "Feature Adoption", value: customer.feature_adoption_rate != null ? `${(customer.feature_adoption_rate * 100).toFixed(0)}%` : "—" },
            { label: "NPS Score",        value: String(customer.nps_score ?? "—") },
          ].map(({ label, value }) => (
            <div key={label} style={{ backgroundColor: "#ffffff", padding: "16px" }} className="text-center">
              <p style={{ fontSize: "0.75rem", color: "#525252", letterSpacing: "0.32px", marginBottom: "4px" }}>
                {label}
              </p>
              <p style={{ fontSize: "1rem", fontWeight: 600, color: "#161616" }}>{value}</p>
            </div>
          ))}
        </div>
      </div>

      {/* ── Error banner ────────────────────────────────────────── */}
      {error && (
        <div
          style={{
            backgroundColor: "#fff1f1",
            borderLeft: "4px solid #da1e28",
            padding: "12px 16px",
            marginBottom: "1px",
          }}
        >
          <p style={{ fontSize: "0.875rem", color: "#da1e28", letterSpacing: "0.16px" }}>{error}</p>
        </div>
      )}

      {/* ── Run button ──────────────────────────────────────────── */}
      {!running && !result && (
        <button
          onClick={startAnalysis}
          className="cds-btn cds-btn--primary w-full"
          style={{ justifyContent: "center", padding: "0 16px", marginBottom: "1px", marginTop: "1px" }}
        >
          Run Full AI Analysis →
        </button>
      )}

      {/* ── Pipeline progress ───────────────────────────────────── */}
      <div style={{ backgroundColor: "#f4f4f4", padding: "24px", marginTop: "1px" }}>
        <div className="flex items-center justify-between mb-5">
          <p style={{ fontSize: "0.875rem", fontWeight: 600, color: "#161616", letterSpacing: "0.16px" }}>
            Pipeline Progress
          </p>
          {running && (
            <span
              className="flex items-center gap-1.5 animate-pulse"
              style={{ fontSize: "0.75rem", color: "#0f62fe", letterSpacing: "0.32px" }}
            >
              <span className="w-2 h-2 rounded-full" style={{ backgroundColor: "#0f62fe" }} />
              Running…
            </span>
          )}
          {result && (
            <span style={{ fontSize: "0.75rem", color: "#24a148", letterSpacing: "0.32px" }}>
              Complete
            </span>
          )}
        </div>

        <div>
          {STEPS.map((step, i) => {
            const done = completedSteps.has(step.key);
            const active = running && i === animatedStep;
            const state = done ? "complete" : active ? "active" : "pending";
            const statusText = done ? "done" : active ? "processing…" : undefined;
            return (
              <ProgressStep
                key={step.key}
                state={state}
                label={step.label}
                isLast={i === STEPS.length - 1}
                statusText={statusText}
              />
            );
          })}
        </div>

        {running && (
          <p
            style={{
              fontSize: "0.75rem",
              color: "#8d8d8d",
              letterSpacing: "0.32px",
              marginTop: "16px",
              textAlign: "center",
            }}
          >
            Running 6-agent AI pipeline — this takes approximately 30 seconds…
          </p>
        )}
      </div>

      {/* ── Prediction results ──────────────────────────────────── */}
      {prediction && (
        <>
          {/* Churn probability */}
          <div style={{ backgroundColor: "#f4f4f4", padding: "24px", marginTop: "1px" }}>
            <p style={{ fontSize: "0.875rem", fontWeight: 600, color: "#161616", letterSpacing: "0.16px", marginBottom: "16px" }}>
              Churn Prediction
            </p>
            <div className="flex items-center gap-5 mb-4">
              <p style={{ fontSize: "2.625rem", fontWeight: 300, color: "#161616", lineHeight: 1.19 }}>
                {(prob * 100).toFixed(1)}%
              </p>
              <div>
                <RiskBadge tier={(prediction.risk_tier ?? "MEDIUM") as RiskTier} />
                {prediction.confidence_interval && (
                  <p style={{ fontSize: "0.75rem", color: "#525252", letterSpacing: "0.32px", marginTop: "4px" }}>
                    95% CI: [{(prediction.confidence_interval[0] * 100).toFixed(1)}% –{" "}
                    {(prediction.confidence_interval[1] * 100).toFixed(1)}%]
                  </p>
                )}
              </div>
            </div>
            {/* Rectangular progress bar — no rounded corners per Carbon spec */}
            <div style={{ width: "100%", height: "8px", backgroundColor: "#e0e0e0" }}>
              <div
                style={{
                  width: `${prob * 100}%`,
                  height: "100%",
                  backgroundColor:
                    prob >= 0.85 ? "#da1e28" :
                    prob >= 0.70 ? "#ff832b" :
                    prob >= 0.40 ? "#f1c21b" : "#24a148",
                  transition: "width 0.7s ease",
                }}
              />
            </div>
            <div
              className="flex justify-between mt-1"
              style={{ fontSize: "0.75rem", color: "#8d8d8d", letterSpacing: "0.32px" }}
            >
              <span>0%</span>
              <span>50%</span>
              <span>100%</span>
            </div>
          </div>

          {/* SHAP risk factors */}
          {result?.top_risk_factors && result.top_risk_factors.length > 0 && (
            <div style={{ backgroundColor: "#f4f4f4", padding: "24px", marginTop: "1px" }}>
              <p style={{ fontSize: "0.875rem", fontWeight: 600, color: "#161616", letterSpacing: "0.16px", marginBottom: "16px" }}>
                Top Risk Factors (SHAP)
              </p>
              <div className="space-y-3">
                {(result.top_risk_factors as RiskFactor[]).slice(0, 6).map((f, i) => {
                  const pct = Math.min(Math.abs(f.shap_value) * 200, 100);
                  return (
                    <div key={i} className="flex items-center gap-3">
                      <span style={{ fontSize: "0.75rem", color: "#8d8d8d", width: "16px", flexShrink: 0 }}>
                        {i + 1}.
                      </span>
                      <span
                        style={{ fontSize: "0.875rem", color: "#161616", letterSpacing: "0.16px", flex: 1 }}
                        className="truncate"
                      >
                        {f.label ?? f.feature}
                      </span>
                      {/* Rectangular bar */}
                      <div style={{ width: "96px", height: "6px", backgroundColor: "#e0e0e0", flexShrink: 0 }}>
                        <div
                          style={{
                            width: `${pct}%`,
                            height: "100%",
                            backgroundColor: f.shap_value > 0 ? "#da1e28" : "#24a148",
                          }}
                        />
                      </div>
                      <span
                        className="font-mono font-semibold"
                        style={{
                          fontSize: "0.875rem",
                          letterSpacing: "0.16px",
                          color: f.shap_value > 0 ? "#da1e28" : "#24a148",
                          width: "56px",
                          textAlign: "right",
                          flexShrink: 0,
                        }}
                      >
                        {f.shap_value > 0 ? "+" : ""}{f.shap_value.toFixed(3)}
                      </span>
                      <span
                        style={{
                          backgroundColor: f.shap_value > 0 ? "#fff1f1" : "#defbe6",
                          color: f.shap_value > 0 ? "#da1e28" : "#044317",
                          borderRadius: "24px",
                          padding: "2px 8px",
                          fontSize: "0.75rem",
                          letterSpacing: "0.32px",
                          flexShrink: 0,
                        }}
                      >
                        {f.shap_value > 0 ? "↑ risk" : "↓ risk"}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Narrative */}
          {result?.narrative && (
            <div style={{ backgroundColor: "#f4f4f4", padding: "24px", marginTop: "1px" }}>
              <p style={{ fontSize: "0.875rem", fontWeight: 600, color: "#161616", letterSpacing: "0.16px", marginBottom: "12px" }}>
                Analysis Narrative
              </p>
              <p
                style={{
                  fontSize: "0.875rem",
                  color: "#525252",
                  lineHeight: 1.6,
                  letterSpacing: "0.16px",
                }}
                className="whitespace-pre-wrap"
              >
                {result.narrative}
              </p>
            </div>
          )}

          {/* Retention Plan */}
          {result?.retention_plan && result.retention_plan.selected_actions?.length > 0 && (
            <div style={{ marginTop: "1px" }}>
              {/* Header */}
              <div
                style={{
                  backgroundColor: "#f4f4f4",
                  padding: "16px 24px",
                  borderBottom: "1px solid #e0e0e0",
                }}
              >
                <div className="flex items-center justify-between">
                  <p style={{ fontSize: "0.875rem", fontWeight: 600, color: "#161616", letterSpacing: "0.16px" }}>
                    Retention Plan
                  </p>
                  <div className="flex items-center gap-3">
                    <span style={{ fontSize: "0.75rem", color: "#525252", letterSpacing: "0.32px" }}>
                      A/B group:{" "}
                      <span style={{ color: "#161616", fontWeight: 600 }}>
                        {result.retention_plan.ab_group}
                      </span>
                    </span>
                    {result.retention_plan.pending_hitl && (
                      <span
                        style={{
                          backgroundColor: "#fff2e8",
                          color: "#ba4e00",
                          borderRadius: "24px",
                          padding: "2px 8px",
                          fontSize: "0.75rem",
                          letterSpacing: "0.32px",
                        }}
                      >
                        Pending HITL
                      </span>
                    )}
                  </div>
                </div>
              </div>
              {/* Stats row */}
              <div className="grid grid-cols-3 gap-px" style={{ backgroundColor: "#c6c6c6" }}>
                {[
                  { label: "Total Cost",    value: `$${result.retention_plan.total_cost_usd.toFixed(0)}`, color: "#161616" },
                  { label: "Revenue Saved", value: `$${result.retention_plan.estimated_revenue_saved_usd.toFixed(0)}`, color: "#24a148" },
                  {
                    label: "ROI",
                    value: `${result.retention_plan.estimated_roi >= 0 ? "+" : ""}${(result.retention_plan.estimated_roi * 100).toFixed(1)}%`,
                    color: result.retention_plan.estimated_roi >= 0 ? "#24a148" : "#da1e28",
                  },
                ].map(({ label, value, color }) => (
                  <div key={label} style={{ backgroundColor: "#f4f4f4", padding: "16px 24px" }} className="text-center">
                    <p style={{ fontSize: "0.75rem", color: "#525252", letterSpacing: "0.32px", marginBottom: "4px" }}>
                      {label}
                    </p>
                    <p style={{ fontSize: "1.25rem", fontWeight: 300, color }}>{value}</p>
                  </div>
                ))}
              </div>
              {/* Actions */}
              {(result.retention_plan.selected_actions as RetentionAction[]).map((action, i) => (
                <div
                  key={i}
                  className="flex items-start justify-between gap-4 px-6 py-4"
                  style={{
                    backgroundColor: i % 2 === 0 ? "#ffffff" : "#f4f4f4",
                    borderBottom: "1px solid #e0e0e0",
                  }}
                >
                  <div className="flex-1">
                    <p style={{ fontSize: "0.875rem", color: "#161616", letterSpacing: "0.16px" }}>
                      {action.label ?? action.action ?? "Retention action"}
                    </p>
                    {action.days_to_effect && (
                      <p style={{ fontSize: "0.75rem", color: "#525252", letterSpacing: "0.32px", marginTop: "2px" }}>
                        Effect in ~{action.days_to_effect} days
                      </p>
                    )}
                  </div>
                  <div className="text-right shrink-0">
                    <p style={{ fontSize: "0.875rem", fontWeight: 600, color: "#0f62fe", letterSpacing: "0.16px" }}>
                      ${action.cost_usd.toFixed(0)}
                    </p>
                    {action.prob_reduction && (
                      <p style={{ fontSize: "0.75rem", color: "#24a148", letterSpacing: "0.32px" }}>
                        -{(action.prob_reduction * 100).toFixed(1)}% churn
                      </p>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* HITL Decision */}
          {result?.hitl_decision && (
            <div style={{ backgroundColor: "#f4f4f4", padding: "24px", marginTop: "1px" }}>
              <p style={{ fontSize: "0.875rem", fontWeight: 600, color: "#161616", letterSpacing: "0.16px", marginBottom: "12px" }}>
                HITL Decision
              </p>
              <div className="flex items-center gap-3">
                <span
                  style={{
                    backgroundColor:
                      result.hitl_decision.status === "rejected" ? "#fff1f1" : "#defbe6",
                    color: result.hitl_decision.status === "rejected" ? "#da1e28" : "#044317",
                    borderRadius: "24px",
                    padding: "4px 12px",
                    fontSize: "0.875rem",
                    fontWeight: 600,
                    letterSpacing: "0.16px",
                  }}
                >
                  {result.hitl_decision.status.replace(/_/g, " ").toUpperCase()}
                </span>
                <span style={{ fontSize: "0.875rem", color: "#525252", letterSpacing: "0.16px" }}>
                  by {result.hitl_decision.decided_by} · {result.hitl_decision.decided_at}
                </span>
              </div>
              {result.hitl_decision.notes && (
                <p style={{ fontSize: "0.875rem", color: "#525252", letterSpacing: "0.16px", marginTop: "8px" }}>
                  {result.hitl_decision.notes}
                </p>
              )}
            </div>
          )}

          {/* Pipeline warnings */}
          {result?.errors && result.errors.length > 0 && (
            <div
              style={{
                backgroundColor: "#fff2e8",
                borderLeft: "4px solid #ff832b",
                padding: "16px 24px",
                marginTop: "1px",
              }}
            >
              <p style={{ fontSize: "0.75rem", fontWeight: 600, color: "#ba4e00", letterSpacing: "0.32px", marginBottom: "8px" }}>
                PIPELINE WARNINGS
              </p>
              {result.errors.map((e, i) => (
                <p key={i} style={{ fontSize: "0.875rem", color: "#ba4e00", letterSpacing: "0.16px" }}>{e}</p>
              ))}
            </div>
          )}
        </>
      )}

      {/* ── HITL approval ───────────────────────────────────────── */}
      {pendingHITL && (
        <div
          style={{
            backgroundColor: "#fff2e8",
            borderLeft: "4px solid #ff832b",
            padding: "24px",
            marginTop: "1px",
          }}
        >
          <p style={{ fontSize: "0.875rem", fontWeight: 600, color: "#ba4e00", letterSpacing: "0.16px", marginBottom: "4px" }}>
            HITL Approval Required
          </p>
          <p style={{ fontSize: "0.875rem", color: "#ba4e00", letterSpacing: "0.16px", marginBottom: "16px" }}>
            This CRITICAL-tier customer requires human review before CRM dispatch.
          </p>
          <div className="flex gap-px" style={{ backgroundColor: "#c6c6c6" }}>
            <button
              onClick={() => handleHITLDecision("approved")}
              className="cds-btn flex-1"
              style={{ backgroundColor: "#24a148", color: "#ffffff", justifyContent: "center", padding: "0 16px" }}
            >
              Approve
            </button>
            <button
              onClick={() => handleHITLDecision("rejected")}
              className="cds-btn flex-1"
              style={{ backgroundColor: "#da1e28", color: "#ffffff", justifyContent: "center", padding: "0 16px" }}
            >
              Reject
            </button>
          </div>
        </div>
      )}

      {/* ── Feedback form ───────────────────────────────────────── */}
      {result && !feedbackSent && (
        <div style={{ backgroundColor: "#f4f4f4", padding: "24px", marginTop: "1px" }}>
          <p style={{ fontSize: "0.875rem", fontWeight: 600, color: "#161616", letterSpacing: "0.16px", marginBottom: "16px" }}>
            Record Outcome Feedback
          </p>
          <form onSubmit={handleFeedback}>
            {/* Outcome selector */}
            <div className="flex gap-px mb-4" style={{ backgroundColor: "#c6c6c6" }}>
              {(["retained", "churned", "unknown"] as const).map((o) => (
                <button
                  type="button"
                  key={o}
                  onClick={() => setFeedbackOutcome(o)}
                  className="flex-1 cds-btn"
                  style={{
                    height: "48px",
                    justifyContent: "center",
                    padding: "0 16px",
                    fontSize: "0.875rem",
                    letterSpacing: "0.16px",
                    textTransform: "capitalize",
                    backgroundColor:
                      feedbackOutcome === o
                        ? o === "retained" ? "#24a148"
                          : o === "churned" ? "#da1e28"
                          : "#393939"
                        : "#ffffff",
                    color: feedbackOutcome === o ? "#ffffff" : "#161616",
                    fontWeight: feedbackOutcome === o ? 600 : 400,
                  }}
                >
                  {o}
                </button>
              ))}
            </div>

            <textarea
              value={feedbackNotes}
              onChange={(e) => setFeedbackNotes(e.target.value)}
              placeholder="Optional notes…"
              rows={3}
              className="cds-input mb-4"
              style={{ height: "auto", padding: "12px 16px", borderBottomColor: "#161616" }}
            />

            <button
              type="submit"
              disabled={!feedbackOutcome}
              className="cds-btn cds-btn--secondary w-full"
              style={{ justifyContent: "center", padding: "0 16px" }}
            >
              Submit Feedback
            </button>
          </form>
        </div>
      )}

      {/* ── Feedback success ────────────────────────────────────── */}
      {feedbackSent && (
        <div
          style={{
            backgroundColor: "#defbe6",
            borderLeft: "4px solid #24a148",
            padding: "16px 24px",
            marginTop: "1px",
          }}
        >
          <p style={{ fontSize: "0.875rem", color: "#044317", letterSpacing: "0.16px", fontWeight: 600 }}>
            Feedback recorded — thank you!
          </p>
        </div>
      )}

      {/* ── Re-run button ───────────────────────────────────────── */}
      {result && !running && (
        <button
          onClick={startAnalysis}
          className="cds-btn cds-btn--ghost w-full"
          style={{ justifyContent: "center", padding: "0 16px", marginTop: "1px" }}
        >
          ↺ Re-run analysis
        </button>
      )}
    </div>
  );
}
