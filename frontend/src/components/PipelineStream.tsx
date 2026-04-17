"use client";

import { useEffect, useRef, useState } from "react";
import { WS_URL } from "@/lib/api";
import type { PipelineEvent, RiskTier } from "@/types";
import RiskBadge from "./RiskBadge";
import RetentionPlanCard from "./RetentionPlanCard";

const STEPS = [
  "data_intelligence",
  "prediction",
  "explanation",
  "counterfactual",
  "retention_strategist",
  "hitl",
];

const STEP_LABELS: Record<string, string> = {
  data_intelligence: "Data Intelligence",
  prediction: "Churn Prediction",
  explanation: "SHAP Explanation",
  counterfactual: "Counterfactual Analysis",
  retention_strategist: "Retention Strategy",
  hitl: "HITL Review",
};

interface Props {
  runId: string;
  onComplete?: (event: PipelineEvent) => void;
}

// Carbon Progress Indicator step circle
function StepCircle({ state }: { state: "complete" | "active" | "pending" }) {
  const base = "w-6 h-6 flex items-center justify-center text-xs font-semibold shrink-0 relative z-10";

  if (state === "complete") {
    return (
      <span
        className={base}
        style={{
          backgroundColor: "#24a148",
          color: "#ffffff",
          borderRadius: "50%",
          fontSize: "14px",
        }}
      >
        ✓
      </span>
    );
  }

  if (state === "active") {
    return (
      <span
        className={`${base} animate-pulse`}
        style={{
          backgroundColor: "#0f62fe",
          color: "#ffffff",
          borderRadius: "50%",
        }}
      >
        ◉
      </span>
    );
  }

  return (
    <span
      className={base}
      style={{
        backgroundColor: "#ffffff",
        border: "2px solid #c6c6c6",
        color: "#8d8d8d",
        borderRadius: "50%",
      }}
    />
  );
}

export default function PipelineStream({ runId, onComplete }: Props) {
  const [events, setEvents] = useState<PipelineEvent[]>([]);
  const [completedSteps, setCompletedSteps] = useState<Set<string>>(new Set());
  const [activeStep, setActiveStep] = useState<string | null>(null);
  const [finalResult, setFinalResult] = useState<PipelineEvent | null>(null);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const ws = new WebSocket(`${WS_URL}/ws/agent/${runId}`);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);

    ws.onmessage = (msg) => {
      const event: PipelineEvent = JSON.parse(msg.data);
      setEvents((prev) => [...prev, event]);

      if (event.status === "running") setActiveStep(event.step);

      if (event.status === "completed") {
        setCompletedSteps((prev) => new Set([...Array.from(prev), event.step]));
        setActiveStep(null);
      }

      if (event.status === "final") {
        setFinalResult(event);
        if (event.completed_steps) {
          setCompletedSteps(new Set(event.completed_steps));
        }
        onComplete?.(event);
      }
    };

    ws.onclose = () => setConnected(false);
    ws.onerror = () => setConnected(false);

    return () => ws.close();
  }, [runId, onComplete]);

  const prediction = finalResult?.prediction;
  const explanation = finalResult?.explanation;
  const retention = finalResult?.retention_plan;
  const hitl = finalResult?.hitl_decision;
  const prob = prediction?.churn_probability ?? 0;

  return (
    <div className="space-y-6">
      {/* Pipeline Progress — Carbon Progress Indicator */}
      <div style={{ backgroundColor: "#f4f4f4" }} className="p-4">
        <div className="flex items-center justify-between mb-4">
          <p
            className="font-semibold"
            style={{ fontSize: "0.875rem", color: "#161616", letterSpacing: "0.16px" }}
          >
            Pipeline Progress
          </p>
          {connected && !finalResult && (
            <span
              className="flex items-center gap-1.5"
              style={{ fontSize: "0.75rem", color: "#0f62fe", letterSpacing: "0.32px" }}
            >
              <span className="w-2 h-2 rounded-full animate-pulse" style={{ backgroundColor: "#0f62fe" }} />
              Live
            </span>
          )}
          {finalResult && (
            <span style={{ fontSize: "0.75rem", color: "#24a148", letterSpacing: "0.32px" }}>
              Complete
            </span>
          )}
        </div>

        <div className="space-y-0">
          {STEPS.map((step, i) => {
            const done = completedSteps.has(step);
            const running = activeStep === step;
            const state = done ? "complete" : running ? "active" : "pending";
            const isLast = i === STEPS.length - 1;

            return (
              <div key={step} className="flex gap-4 relative">
                {/* Left column: circle + connector */}
                <div className="flex flex-col items-center" style={{ width: "24px" }}>
                  <StepCircle state={state} />
                  {!isLast && (
                    <div
                      className="flex-1 w-px my-1"
                      style={{ backgroundColor: done ? "#24a148" : "#c6c6c6", minHeight: "16px" }}
                    />
                  )}
                </div>

                {/* Right column: label + optional status */}
                <div className={`flex items-center justify-between flex-1 ${isLast ? "pb-0" : "pb-4"}`}>
                  <span
                    style={{
                      fontSize: "0.875rem",
                      letterSpacing: "0.16px",
                      color: done ? "#161616" : running ? "#0f62fe" : "#8d8d8d",
                      fontWeight: done ? 600 : running ? 600 : 400,
                    }}
                  >
                    {STEP_LABELS[step] ?? step}
                  </span>
                  {running && (
                    <span
                      className="animate-pulse"
                      style={{ fontSize: "0.75rem", color: "#0f62fe", letterSpacing: "0.32px" }}
                    >
                      processing…
                    </span>
                  )}
                  {done && (
                    <span style={{ fontSize: "0.75rem", color: "#24a148", letterSpacing: "0.32px" }}>
                      done
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Results */}
      {finalResult && prediction && (
        <>
          {/* Prediction */}
          <div style={{ backgroundColor: "#f4f4f4" }} className="p-4">
            <p
              className="font-semibold mb-4"
              style={{ fontSize: "0.875rem", color: "#161616", letterSpacing: "0.16px" }}
            >
              Prediction Result
            </p>
            <div className="flex items-center gap-5 mb-4">
              <p style={{ fontSize: "2.625rem", fontWeight: 300, color: "#161616", lineHeight: 1.19 }}>
                {(prob * 100).toFixed(1)}%
              </p>
              <div>
                <RiskBadge tier={prediction.risk_tier as RiskTier} />
                <p style={{ fontSize: "0.75rem", color: "#525252", letterSpacing: "0.32px" }} className="mt-1">
                  CI: [{((prediction.confidence_interval?.[0] ?? 0) * 100).toFixed(1)}% –{" "}
                  {((prediction.confidence_interval?.[1] ?? 0) * 100).toFixed(1)}%]
                </p>
              </div>
            </div>
            {/* Rectangular progress bar — no rounded corners */}
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
          </div>

          {/* Narrative */}
          {explanation?.narrative_text && (
            <div style={{ backgroundColor: "#f4f4f4" }} className="p-4">
              <p
                className="font-semibold mb-3"
                style={{ fontSize: "0.875rem", color: "#161616", letterSpacing: "0.16px" }}
              >
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
                {explanation.narrative_text}
              </p>
            </div>
          )}

          {/* SHAP Factors */}
          {explanation?.top_risk_factors && explanation.top_risk_factors.length > 0 && (
            <div style={{ backgroundColor: "#f4f4f4" }} className="p-4">
              <p
                className="font-semibold mb-4"
                style={{ fontSize: "0.875rem", color: "#161616", letterSpacing: "0.16px" }}
              >
                Top Risk Factors (SHAP)
              </p>
              <div className="space-y-3">
                {explanation.top_risk_factors.slice(0, 5).map((f, i) => (
                  <div key={i} className="flex items-center gap-3">
                    <span style={{ fontSize: "0.75rem", color: "#8d8d8d", width: "16px", flexShrink: 0 }}>
                      {i + 1}
                    </span>
                    <span style={{ fontSize: "0.875rem", color: "#161616", letterSpacing: "0.16px" }} className="flex-1">
                      {f.label ?? f.feature}
                    </span>
                    <span
                      className="font-mono font-semibold"
                      style={{
                        fontSize: "0.875rem",
                        color: f.shap_value > 0 ? "#da1e28" : "#24a148",
                        letterSpacing: "0.16px",
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
                      }}
                    >
                      {f.shap_value > 0 ? "↑ risk" : "↓ risk"}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Retention Plan */}
          {retention && <RetentionPlanCard plan={retention} />}

          {/* HITL Decision */}
          {hitl && (
            <div style={{ backgroundColor: "#f4f4f4" }} className="p-4">
              <p
                className="font-semibold mb-3"
                style={{ fontSize: "0.875rem", color: "#161616", letterSpacing: "0.16px" }}
              >
                HITL Decision
              </p>
              <div className="flex items-center gap-3">
                <span
                  style={{
                    backgroundColor: hitl.status === "rejected" ? "#fff1f1" : "#defbe6",
                    color: hitl.status === "rejected" ? "#da1e28" : "#044317",
                    borderRadius: "24px",
                    padding: "4px 8px",
                    fontSize: "0.875rem",
                    fontWeight: 600,
                    letterSpacing: "0.16px",
                  }}
                >
                  {hitl.status.replace("_", " ").toUpperCase()}
                </span>
                <span style={{ fontSize: "0.875rem", color: "#525252", letterSpacing: "0.16px" }}>
                  by {hitl.decided_by} · {hitl.decided_at}
                </span>
              </div>
              {hitl.notes && (
                <p style={{ fontSize: "0.875rem", color: "#525252", letterSpacing: "0.16px" }} className="mt-2">
                  {hitl.notes}
                </p>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}
