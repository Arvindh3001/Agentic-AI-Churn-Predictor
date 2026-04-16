"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { fetchAuditLog, fetchFeedbackStats, getToken } from "@/lib/api";

// Carbon tag colors for event types
const EVENT_STYLES: Record<string, { bg: string; color: string }> = {
  hitl_decision:      { bg: "#edf5ff", color: "#0f62fe" },
  crm_action:         { bg: "#defbe6", color: "#044317" },
  feedback_recorded:  { bg: "#edf5ff", color: "#0043ce" },
  retrain_trigger:    { bg: "#fff2e8", color: "#ba4e00" },
};

export default function HITLQueuePage() {
  const router = useRouter();
  const [entries, setEntries] = useState<Record<string, unknown>[]>([]);
  const [stats, setStats] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!getToken()) { router.replace("/login"); return; }
    Promise.all([fetchAuditLog(50), fetchFeedbackStats()])
      .then(([audit, st]) => { setEntries(audit.entries); setStats(st); })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [router]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <p style={{ fontSize: "0.875rem", color: "#525252", letterSpacing: "0.16px" }}>
          Loading audit log…
        </p>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ backgroundColor: "#fff1f1", borderLeft: "4px solid #da1e28", padding: "16px" }}>
        <p style={{ fontSize: "0.875rem", color: "#da1e28", letterSpacing: "0.16px" }}>{error}</p>
      </div>
    );
  }

  const total    = (stats?.total_feedback as number) ?? 0;
  const retained = (stats?.retained as number) ?? 0;
  const churned  = (stats?.churned as number) ?? 0;
  const unknown  = (stats?.unknown as number) ?? 0;

  return (
    <div className="space-y-0">
      {/* ── Page header ─────────────────────────────────────────── */}
      <div
        className="mb-8"
        style={{ paddingBottom: "16px", borderBottom: "1px solid #e0e0e0" }}
      >
        <h1
          style={{ fontSize: "1.75rem", fontWeight: 400, color: "#161616", lineHeight: 1.29 }}
        >
          HITL Queue
        </h1>
        <p style={{ fontSize: "0.875rem", color: "#525252", letterSpacing: "0.16px", marginTop: "4px" }}>
          Human-in-the-loop audit log and feedback statistics
        </p>
      </div>

      {/* ── Feedback stats — flat gray-10 tiles ─────────────────── */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-px mb-px" style={{ backgroundColor: "#c6c6c6" }}>
        {[
          { label: "TOTAL FEEDBACK", value: total,    color: "#161616" },
          { label: "RETAINED",       value: retained, color: "#24a148" },
          { label: "CHURNED",        value: churned,  color: "#da1e28" },
          { label: "UNKNOWN",        value: unknown,  color: "#8d8d8d" },
        ].map(({ label, value, color }) => (
          <div key={label} style={{ backgroundColor: "#f4f4f4", padding: "24px" }}>
            <p
              style={{
                fontSize: "0.75rem",
                color: "#525252",
                letterSpacing: "0.32px",
                textTransform: "uppercase",
                marginBottom: "8px",
              }}
            >
              {label}
            </p>
            <p style={{ fontSize: "2rem", fontWeight: 300, color, lineHeight: 1.25 }}>
              {value}
            </p>
          </div>
        ))}
      </div>

      {/* ── A/B breakdown ───────────────────────────────────────── */}
      {stats?.ab_breakdown && (
        <div style={{ backgroundColor: "#f4f4f4", padding: "24px", marginTop: "1px", marginBottom: "1px" }}>
          <p
            className="font-semibold mb-4"
            style={{ fontSize: "0.875rem", color: "#161616", letterSpacing: "0.16px" }}
          >
            A/B Group Breakdown
          </p>
          <div className="flex gap-px" style={{ backgroundColor: "#c6c6c6" }}>
            {Object.entries(stats.ab_breakdown as Record<string, unknown>).map(([group, counts]) => (
              <div key={group} style={{ backgroundColor: "#ffffff", padding: "16px", flex: 1 }}>
                <p
                  style={{
                    fontSize: "0.75rem",
                    fontWeight: 600,
                    color: "#525252",
                    letterSpacing: "0.32px",
                    textTransform: "uppercase",
                    marginBottom: "8px",
                  }}
                >
                  {group}
                </p>
                {Object.entries(counts as Record<string, number>).map(([k, v]) => (
                  <p key={k} style={{ fontSize: "0.875rem", color: "#525252", letterSpacing: "0.16px" }}>
                    {k}:{" "}
                    <span style={{ color: "#161616", fontWeight: 600 }}>{v}</span>
                  </p>
                ))}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Audit log table ─────────────────────────────────────── */}
      <div style={{ backgroundColor: "#ffffff", marginTop: "8px" }}>
        <div
          className="flex items-center justify-between px-6 py-4"
          style={{ borderBottom: "1px solid #e0e0e0" }}
        >
          <p style={{ fontSize: "1rem", fontWeight: 600, color: "#161616", lineHeight: 1.5 }}>
            Audit Log
          </p>
          <span
            style={{
              fontSize: "0.75rem",
              color: "#525252",
              backgroundColor: "#f4f4f4",
              padding: "4px 10px",
              letterSpacing: "0.32px",
            }}
          >
            {entries.length} events
          </span>
        </div>

        {entries.length === 0 ? (
          <div className="px-6 py-12 text-center">
            <p style={{ fontSize: "0.875rem", color: "#8d8d8d", letterSpacing: "0.16px" }}>
              No audit events recorded yet
            </p>
          </div>
        ) : (
          <div>
            {entries.map((entry, i) => {
              const eventType = entry.event as string;
              const tagStyle = EVENT_STYLES[eventType] ?? { bg: "#f4f4f4", color: "#525252" };
              const isEven = i % 2 === 0;

              return (
                <div
                  key={i}
                  className="flex items-start gap-4 px-6 py-3"
                  style={{
                    backgroundColor: isEven ? "#ffffff" : "#f4f4f4",
                    borderBottom: "1px solid #e0e0e0",
                  }}
                >
                  {/* Event type tag */}
                  <span
                    style={{
                      ...tagStyle,
                      borderRadius: "24px",
                      padding: "2px 8px",
                      fontSize: "0.75rem",
                      fontWeight: 400,
                      letterSpacing: "0.32px",
                      whiteSpace: "nowrap",
                      flexShrink: 0,
                    }}
                  >
                    {eventType?.replace(/_/g, " ")}
                  </span>

                  {/* Details */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-3 flex-wrap">
                      {entry.run_id && (
                        <span
                          className="font-mono"
                          style={{ fontSize: "0.75rem", color: "#525252", letterSpacing: "0.16px" }}
                        >
                          {(entry.run_id as string).slice(0, 12)}…
                        </span>
                      )}
                      {entry.customer_id && (
                        <span
                          className="font-mono"
                          style={{ fontSize: "0.75rem", color: "#0f62fe", letterSpacing: "0.16px" }}
                        >
                          {entry.customer_id as string}
                        </span>
                      )}
                      {entry.decision && (
                        <span
                          style={{
                            fontSize: "0.875rem",
                            fontWeight: 600,
                            letterSpacing: "0.16px",
                            color: entry.decision === "approved" ? "#24a148" : "#da1e28",
                          }}
                        >
                          {entry.decision as string}
                        </span>
                      )}
                      {entry.outcome && (
                        <span style={{ fontSize: "0.875rem", color: "#0f62fe", letterSpacing: "0.16px" }}>
                          {entry.outcome as string}
                        </span>
                      )}
                    </div>
                    {entry.notes && (
                      <p
                        style={{
                          fontSize: "0.75rem",
                          color: "#8d8d8d",
                          letterSpacing: "0.32px",
                          marginTop: "2px",
                        }}
                        className="truncate"
                      >
                        {entry.notes as string}
                      </p>
                    )}
                  </div>

                  {/* Timestamp */}
                  <span
                    style={{
                      fontSize: "0.75rem",
                      color: "#8d8d8d",
                      letterSpacing: "0.32px",
                      whiteSpace: "nowrap",
                      flexShrink: 0,
                    }}
                  >
                    {entry.timestamp
                      ? new Date(entry.timestamp as string).toLocaleString()
                      : ""}
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
