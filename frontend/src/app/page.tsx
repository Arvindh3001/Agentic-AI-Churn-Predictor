"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import {
  fetchHighRisk,
  fetchCustomers,
  fetchFeedbackStats,
  updateCustomer,
  getToken,
} from "@/lib/api";
import type { Customer } from "@/types";
import RiskBadge from "@/components/RiskBadge";
import EditCustomerModal from "@/components/EditCustomerModal";
import { useCustomerSocket } from "@/hooks/useCustomerSocket";

// ------------------------------------------------------------------ //
// Types
// ------------------------------------------------------------------ //

type RiskTierLabel = "CRITICAL" | "HIGH" | "MEDIUM" | "LOW";
type ToastType = "success" | "error" | "info";

interface Toast {
  id: string;
  message: string;
  type: ToastType;
}

// ------------------------------------------------------------------ //
// Helpers
// ------------------------------------------------------------------ //

function getRiskTier(c: Customer): RiskTierLabel {
  if (c.monthly_charges >= 90) return "CRITICAL";
  if (c.monthly_charges >= 70) return "HIGH";
  if (c.monthly_charges >= 40) return "MEDIUM";
  return "LOW";
}

function getRiskScore(c: Customer): number {
  let score = 0;
  score += Math.min(c.monthly_charges / 120, 1) * 30;
  score += Math.max(0, 1 - c.tenure_months / 60) * 20;
  if (c.num_support_tickets_30d) score += Math.min(c.num_support_tickets_30d / 10, 1) * 25;
  if (c.feature_adoption_rate != null) score += (1 - c.feature_adoption_rate) * 15;
  if (c.nps_score != null) score += Math.max(0, (5 - c.nps_score) / 10) * 10;
  return Math.min(Math.round(score), 99);
}

function scoreStyle(score: number): { color: string; bg: string } {
  if (score >= 75) return { color: "#da1e28", bg: "#fff1f1" };
  if (score >= 55) return { color: "#ba4e00", bg: "#fff2e8" };
  if (score >= 35) return { color: "#684e00", bg: "#fcf4d6" };
  return { color: "#044317", bg: "#defbe6" };
}

// ------------------------------------------------------------------ //
// Component
// ------------------------------------------------------------------ //

export default function DashboardPage() {
  const router = useRouter();

  const [watchlist, setWatchlist] = useState<Customer[]>([]);
  const [allStats, setAllStats] = useState<{ total: number; churned: number } | null>(null);
  const [feedbackStats, setFeedbackStats] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [editTarget, setEditTarget] = useState<Customer | null>(null);
  const [flashIds, setFlashIds] = useState<Set<string>>(new Set());
  const [toasts, setToasts] = useState<Toast[]>([]);

  const setToastsRef = useRef(setToasts);
  setToastsRef.current = setToasts;

  function showToast(message: string, type: ToastType = "success") {
    const id = Date.now().toString(36) + Math.random().toString(36).slice(2);
    setToastsRef.current((prev) => [...prev, { id, message, type }]);
    setTimeout(() => setToastsRef.current((prev) => prev.filter((t) => t.id !== id)), 3500);
  }

  function flashRow(customerId: string) {
    setFlashIds((prev) => new Set([...Array.from(prev), customerId]));
    setTimeout(() => setFlashIds((prev) => { const n = new Set(prev); n.delete(customerId); return n; }), 2000);
  }

  const handleSocketUpdate = useCallback((updated: Customer) => {
    setWatchlist((prev) => prev.map((c) => c.customer_id === updated.customer_id ? updated : c));
    setFlashIds((prev) => new Set([...Array.from(prev), updated.customer_id]));
    setTimeout(() => setFlashIds((prev) => { const n = new Set(prev); n.delete(updated.customer_id); return n; }), 2000);
    const id = "ws-" + updated.customer_id + Date.now();
    setToastsRef.current((prev) => [...prev, { id, message: `${updated.customer_id} synced in real-time`, type: "info" as ToastType }]);
    setTimeout(() => setToastsRef.current((prev) => prev.filter((t) => t.id !== id)), 3500);
  }, []);

  useCustomerSocket(handleSocketUpdate);

  useEffect(() => {
    if (!getToken()) { router.replace("/login"); return; }
    Promise.all([
      fetchHighRisk(25),
      fetchCustomers({ limit: 1 }),
      fetchCustomers({ status_filter: "churned", limit: 1 }),
      fetchFeedbackStats(),
    ])
      .then(([hr, all, churned, fb]) => {
        setWatchlist(hr.customers);
        setAllStats({ total: all.total, churned: churned.total });
        setFeedbackStats(fb);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [router]);

  const allSelected = watchlist.length > 0 && selected.size === watchlist.length;
  const someSelected = selected.size > 0 && !allSelected;

  function toggleSelectAll() {
    setSelected(allSelected ? new Set() : new Set(watchlist.map((c) => c.customer_id)));
  }

  function toggleSelect(id: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  }

  async function handleSaveEdit(updates: Parameters<typeof updateCustomer>[1]) {
    if (!editTarget) return;
    const res = await updateCustomer(editTarget.customer_id, updates);
    setWatchlist((prev) => prev.map((c) => c.customer_id === editTarget.customer_id ? res.customer : c));
    flashRow(editTarget.customer_id);
    setEditTarget(null);
    showToast(`${editTarget.customer_id} updated`);
  }

  // ---- Loading / Error ------------------------------------------- //

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-64 gap-4">
        <div
          className="w-8 h-8 border-4 border-t-transparent rounded-full animate-spin"
          style={{ borderColor: "#0f62fe", borderTopColor: "transparent" }}
        />
        <p style={{ fontSize: "0.875rem", color: "#525252", letterSpacing: "0.16px" }}>
          Loading intelligence platform…
        </p>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ backgroundColor: "#fff1f1", borderLeft: "4px solid #da1e28", padding: "16px" }}>
        <p style={{ fontSize: "0.875rem", color: "#da1e28", letterSpacing: "0.16px" }}>
          <strong>Error:</strong> {error}
        </p>
      </div>
    );
  }

  // ---- Derived values -------------------------------------------- //

  const totalCustomers = allStats?.total ?? 0;
  const churnedCount = allStats?.churned ?? 0;
  const churnRate = totalCustomers > 0 ? (churnedCount / totalCustomers) * 100 : 0;
  const revenueAtRisk = watchlist.slice(0, 10).reduce((s, c) => s + c.monthly_charges, 0);

  const critical = watchlist.filter((c) => getRiskTier(c) === "CRITICAL");
  const high = watchlist.filter((c) => getRiskTier(c) === "HIGH");

  const totalFeedback = (feedbackStats?.total_feedback as number) ?? 0;
  const retained = (feedbackStats?.retained as number) ?? 0;
  const retentionRate = totalFeedback > 0 ? (retained / totalFeedback) * 100 : null;

  // ---------------------------------------------------------------- //
  // Render
  // ---------------------------------------------------------------- //

  return (
    <div className="space-y-0">
      {/* ── Page header ──────────────────────────────────────────── */}
      <div
        className="flex items-center justify-between mb-8"
        style={{ paddingBottom: "16px", borderBottom: "1px solid #e0e0e0" }}
      >
        <div>
          <h1
            style={{ fontSize: "1.75rem", fontWeight: 400, color: "#161616", lineHeight: 1.29 }}
          >
            Churn Intelligence Dashboard
          </h1>
          <p style={{ fontSize: "0.875rem", color: "#525252", letterSpacing: "0.16px", marginTop: "4px" }}>
            Real-time customer risk monitoring · {totalCustomers.toLocaleString()} customers tracked
          </p>
        </div>
        <span
          className="flex items-center gap-1.5"
          style={{
            fontSize: "0.75rem",
            color: "#24a148",
            backgroundColor: "#defbe6",
            borderRadius: "24px",
            padding: "4px 12px",
            letterSpacing: "0.32px",
            fontWeight: 400,
          }}
        >
          <span className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ backgroundColor: "#24a148" }} />
          Live
        </span>
      </div>

      {/* ── KPI strip ────────────────────────────────────────────── */}
      {/* Carbon: flat gray-10 tiles, 1px hairline gap between them, no border/shadow */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-px mb-px" style={{ backgroundColor: "#c6c6c6" }}>
        {[
          {
            label: "TOTAL CUSTOMERS",
            value: totalCustomers.toLocaleString(),
            sub: "across all segments",
            valueColor: "#161616",
          },
          {
            label: "CHURN RATE",
            value: `${churnRate.toFixed(1)}%`,
            sub: `${churnedCount.toLocaleString()} at-risk`,
            valueColor: "#da1e28",
          },
          {
            label: "REVENUE AT RISK",
            value: `$${revenueAtRisk.toLocaleString()}`,
            sub: "top 10 watchlist · monthly",
            valueColor: "#ba4e00",
          },
          {
            label: "RETENTION RATE",
            value: retentionRate !== null ? `${retentionRate.toFixed(1)}%` : "—",
            sub: retentionRate !== null ? `from ${totalFeedback} interventions` : "no feedback yet",
            valueColor: retentionRate !== null ? "#24a148" : "#8d8d8d",
          },
        ].map(({ label, value, sub, valueColor }) => (
          <div key={label} style={{ backgroundColor: "#f4f4f4", padding: "24px" }}>
            <p
              style={{
                fontSize: "0.75rem",
                color: "#525252",
                letterSpacing: "0.32px",
                marginBottom: "8px",
                textTransform: "uppercase",
              }}
            >
              {label}
            </p>
            <p style={{ fontSize: "2rem", fontWeight: 300, color: valueColor, lineHeight: 1.25 }}>
              {value}
            </p>
            <p style={{ fontSize: "0.75rem", color: "#8d8d8d", letterSpacing: "0.32px", marginTop: "4px" }}>
              {sub}
            </p>
          </div>
        ))}
      </div>

      {/* ── Risk tier summary ────────────────────────────────────── */}
      <div className="grid grid-cols-2 gap-px mb-8" style={{ backgroundColor: "#c6c6c6" }}>
        <div
          style={{
            backgroundColor: "#fff1f1",
            padding: "16px 24px",
            borderLeft: "4px solid #da1e28",
          }}
          className="flex items-center gap-4"
        >
          <p style={{ fontSize: "2rem", fontWeight: 300, color: "#da1e28", lineHeight: 1 }}>
            {critical.length}
          </p>
          <div>
            <p style={{ fontSize: "0.875rem", fontWeight: 600, color: "#da1e28", letterSpacing: "0.16px" }}>
              CRITICAL Risk
            </p>
            <p style={{ fontSize: "0.75rem", color: "#da1e28", letterSpacing: "0.32px", marginTop: "2px" }}>
              Monthly ≥ $90 · Immediate attention required
            </p>
          </div>
        </div>
        <div
          style={{
            backgroundColor: "#fff2e8",
            padding: "16px 24px",
            borderLeft: "4px solid #ff832b",
          }}
          className="flex items-center gap-4"
        >
          <p style={{ fontSize: "2rem", fontWeight: 300, color: "#ba4e00", lineHeight: 1 }}>
            {high.length}
          </p>
          <div>
            <p style={{ fontSize: "0.875rem", fontWeight: 600, color: "#ba4e00", letterSpacing: "0.16px" }}>
              HIGH Risk
            </p>
            <p style={{ fontSize: "0.75rem", color: "#ba4e00", letterSpacing: "0.32px", marginTop: "2px" }}>
              Monthly $70–$89 · Proactive outreach advised
            </p>
          </div>
        </div>
      </div>

      {/* ── Customer watchlist ───────────────────────────────────── */}
      <div style={{ backgroundColor: "#ffffff" }}>
        {/* Table header bar */}
        <div
          className="flex items-center justify-between px-6 py-4"
          style={{ borderBottom: "1px solid #e0e0e0" }}
        >
          <div>
            <p style={{ fontSize: "1rem", fontWeight: 600, color: "#161616", lineHeight: 1.5 }}>
              High-Risk Customer Watchlist
            </p>
            <p style={{ fontSize: "0.75rem", color: "#525252", letterSpacing: "0.32px", marginTop: "2px" }}>
              Sorted by revenue exposure · Hover a row to Edit or Analyse
            </p>
          </div>
          <div className="flex items-center gap-3">
            {selected.size > 0 && (
              <span
                style={{
                  fontSize: "0.75rem",
                  fontWeight: 600,
                  color: "#0f62fe",
                  backgroundColor: "#edf5ff",
                  borderRadius: "24px",
                  padding: "4px 10px",
                  letterSpacing: "0.32px",
                }}
              >
                {selected.size} selected
              </span>
            )}
            <span
              style={{
                fontSize: "0.75rem",
                color: "#525252",
                backgroundColor: "#f4f4f4",
                padding: "4px 10px",
                letterSpacing: "0.32px",
              }}
            >
              {watchlist.length} customers
            </span>
          </div>
        </div>

        {/* Table */}
        <div className="overflow-x-auto">
          <table className="w-full" style={{ borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ backgroundColor: "#f4f4f4", borderBottom: "1px solid #c6c6c6" }}>
                {/* Select-all */}
                <th className="pl-6 pr-3 py-3">
                  <input
                    type="checkbox"
                    checked={allSelected}
                    ref={(el) => { if (el) el.indeterminate = someSelected; }}
                    onChange={toggleSelectAll}
                    style={{ accentColor: "#0f62fe", cursor: "pointer" }}
                  />
                </th>
                {["CUSTOMER", "RISK", "SCORE", "MONTHLY $", "TENURE", "CONTRACT", "TICKETS", "ADOPTION", "NPS", ""].map(
                  (h) => (
                    <th
                      key={h}
                      className="text-left px-4 py-3"
                      style={{
                        fontSize: "0.75rem",
                        fontWeight: 600,
                        color: "#525252",
                        letterSpacing: "0.32px",
                        textTransform: "uppercase",
                        whiteSpace: "nowrap",
                      }}
                    >
                      {h}
                    </th>
                  )
                )}
              </tr>
            </thead>
            <tbody>
              {watchlist.map((c, idx) => {
                const tier = getRiskTier(c);
                const score = getRiskScore(c);
                const { color: sc, bg: sb } = scoreStyle(score);
                const isSelected = selected.has(c.customer_id);
                const isFlashing = flashIds.has(c.customer_id);
                const isEven = idx % 2 === 0;

                const rowBg = isFlashing
                  ? "#edf5ff"
                  : isSelected
                    ? "#edf5ff"
                    : isEven
                      ? "#ffffff"
                      : "#f4f4f4";

                return (
                  <tr
                    key={c.customer_id}
                    className="group"
                    style={{
                      backgroundColor: rowBg,
                      borderBottom: "1px solid #e0e0e0",
                      transition: "background-color 0.5s ease",
                    }}
                    onMouseEnter={(e) => {
                      if (!isFlashing && !isSelected)
                        (e.currentTarget as HTMLElement).style.backgroundColor = "#e8e8e8";
                    }}
                    onMouseLeave={(e) => {
                      if (!isFlashing && !isSelected)
                        (e.currentTarget as HTMLElement).style.backgroundColor = rowBg;
                    }}
                  >
                    {/* Checkbox */}
                    <td className="pl-6 pr-3 py-3">
                      <input
                        type="checkbox"
                        checked={isSelected}
                        onChange={() => toggleSelect(c.customer_id)}
                        style={{ accentColor: "#0f62fe", cursor: "pointer" }}
                      />
                    </td>

                    {/* Customer ID */}
                    <td className="px-4 py-3">
                      <span
                        className="font-mono"
                        style={{ fontSize: "0.875rem", color: "#161616", letterSpacing: "0.16px" }}
                      >
                        {c.customer_id}
                      </span>
                    </td>

                    {/* Risk */}
                    <td className="px-4 py-3">
                      <RiskBadge tier={tier} size="sm" />
                    </td>

                    {/* Score */}
                    <td className="px-4 py-3">
                      <span
                        style={{
                          fontSize: "0.75rem",
                          fontWeight: 600,
                          color: sc,
                          backgroundColor: sb,
                          borderRadius: "24px",
                          padding: "2px 8px",
                          letterSpacing: "0.32px",
                        }}
                      >
                        {score}
                      </span>
                    </td>

                    {/* Monthly */}
                    <td className="px-4 py-3">
                      <span style={{ fontSize: "0.875rem", fontWeight: 600, color: "#161616", letterSpacing: "0.16px" }}>
                        ${c.monthly_charges.toFixed(0)}
                      </span>
                    </td>

                    {/* Tenure */}
                    <td className="px-4 py-3">
                      <span style={{ fontSize: "0.875rem", color: "#525252", letterSpacing: "0.16px" }}>
                        {c.tenure_months}mo
                      </span>
                    </td>

                    {/* Contract */}
                    <td className="px-4 py-3">
                      <span style={{ fontSize: "0.75rem", color: "#525252", letterSpacing: "0.32px" }}>
                        {c.contract_type}
                      </span>
                    </td>

                    {/* Tickets */}
                    <td className="px-4 py-3">
                      {c.num_support_tickets_30d != null ? (
                        <span
                          style={{
                            fontSize: "0.875rem",
                            letterSpacing: "0.16px",
                            color: c.num_support_tickets_30d >= 5 ? "#da1e28" : "#525252",
                            fontWeight: c.num_support_tickets_30d >= 5 ? 600 : 400,
                          }}
                        >
                          {c.num_support_tickets_30d >= 5 ? "⚠ " : ""}
                          {c.num_support_tickets_30d}
                        </span>
                      ) : (
                        <span style={{ color: "#c6c6c6" }}>—</span>
                      )}
                    </td>

                    {/* Adoption */}
                    <td className="px-4 py-3">
                      {c.feature_adoption_rate != null ? (
                        <div className="flex items-center gap-2">
                          <div style={{ width: "64px", height: "4px", backgroundColor: "#e0e0e0", flexShrink: 0 }}>
                            <div
                              style={{
                                width: `${c.feature_adoption_rate * 100}%`,
                                height: "100%",
                                backgroundColor:
                                  c.feature_adoption_rate >= 0.6 ? "#24a148" :
                                    c.feature_adoption_rate >= 0.3 ? "#f1c21b" : "#da1e28",
                                transition: "width 0.5s ease",
                              }}
                            />
                          </div>
                          <span style={{ fontSize: "0.75rem", color: "#525252", letterSpacing: "0.32px" }}>
                            {(c.feature_adoption_rate * 100).toFixed(0)}%
                          </span>
                        </div>
                      ) : (
                        <span style={{ color: "#c6c6c6" }}>—</span>
                      )}
                    </td>

                    {/* NPS */}
                    <td className="px-4 py-3">
                      {c.nps_score != null ? (
                        <span
                          style={{
                            fontSize: "0.875rem",
                            fontWeight: 600,
                            letterSpacing: "0.16px",
                            color:
                              c.nps_score >= 7 ? "#24a148" :
                                c.nps_score >= 5 ? "#684e00" : "#da1e28",
                          }}
                        >
                          {c.nps_score}
                        </span>
                      ) : (
                        <span style={{ color: "#c6c6c6" }}>—</span>
                      )}
                    </td>

                    {/* Row actions */}
                    <td className="px-4 py-3 text-right">
                      <div className="flex items-center justify-end gap-2 opacity-0 group-hover:opacity-100 transition-opacity duration-100">
                        <button
                          onClick={() => setEditTarget(c)}
                          className="cds-btn cds-btn--ghost cds-btn--sm"
                          style={{ padding: "0 12px", height: "32px", fontSize: "0.75rem" }}
                        >
                          <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                              d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                          </svg>
                          Edit
                        </button>
                        <Link
                          href={`/customers/${c.customer_id}`}
                          className="cds-btn cds-btn--primary cds-btn--sm"
                          style={{ padding: "0 12px", height: "32px", fontSize: "0.75rem", display: "inline-flex", alignItems: "center" }}
                        >
                          Analyse →
                        </Link>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* ── Legend ───────────────────────────────────────────────── */}
      <div style={{ backgroundColor: "#f4f4f4", padding: "16px 24px", marginTop: "1px" }}>
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
          Risk Score Legend
        </p>
        <div className="flex gap-8 flex-wrap">
          {[
            { range: "75–99", label: "Critical", color: "#da1e28" },
            { range: "55–74", label: "High", color: "#ba4e00" },
            { range: "35–54", label: "Medium", color: "#684e00" },
            { range: "0–34", label: "Low", color: "#044317" },
          ].map(({ range, label, color }) => (
            <span
              key={label}
              style={{ fontSize: "0.875rem", color: "#525252", letterSpacing: "0.16px" }}
            >
              <span style={{ fontWeight: 600, color }}>{range}</span>
              {" · "}{label}
            </span>
          ))}
        </div>
        <p style={{ fontSize: "0.75rem", color: "#8d8d8d", letterSpacing: "0.32px", marginTop: "8px" }}>
          Heuristic score from: monthly charges, tenure, support tickets, feature adoption, NPS.
          Full ML probability available after running the AI pipeline.
        </p>
      </div>

      {/* ── Edit modal ───────────────────────────────────────────── */}
      {editTarget && (
        <EditCustomerModal
          customer={editTarget}
          onSave={handleSaveEdit}
          onClose={() => setEditTarget(null)}
        />
      )}

      {/* ── Batch selection bar ──────────────────────────────────── */}
      {selected.size > 0 && (
        <div
          className="fixed bottom-6 left-1/2 -translate-x-1/2 z-40 flex items-center gap-4 px-6 py-3"
          style={{
            backgroundColor: "#161616",
            color: "#ffffff",
            boxShadow: "0 2px 6px rgba(0,0,0,0.50)",
          }}
        >
          <span style={{ fontSize: "0.875rem", fontWeight: 600, letterSpacing: "0.16px" }}>
            {selected.size} customer{selected.size !== 1 ? "s" : ""} selected
          </span>
          <div style={{ width: "1px", height: "16px", backgroundColor: "#393939" }} />
          <button
            disabled
            title="Batch analysis — coming soon"
            style={{
              fontSize: "0.875rem",
              color: "#525252",
              cursor: "not-allowed",
              letterSpacing: "0.16px",
              display: "flex",
              alignItems: "center",
              gap: "6px",
            }}
          >
            Batch Analyse
            <span
              style={{
                fontSize: "0.75rem",
                color: "#525252",
                backgroundColor: "#262626",
                padding: "2px 8px",
                borderRadius: "24px",
                letterSpacing: "0.32px",
              }}
            >
              soon
            </span>
          </button>
          <div style={{ width: "1px", height: "16px", backgroundColor: "#393939" }} />
          <button
            onClick={() => setSelected(new Set())}
            style={{ fontSize: "0.875rem", color: "#c6c6c6", letterSpacing: "0.16px" }}
            className="hover:text-white transition-colors"
          >
            Clear
          </button>
        </div>
      )}

      {/* ── Toast notifications ──────────────────────────────────── */}
      <div className="fixed bottom-4 right-4 flex flex-col gap-2 z-50 pointer-events-none">
        {toasts.map((t) => (
          <div
            key={t.id}
            className="flex items-center gap-2 px-4 py-3 text-white pointer-events-auto toast-animate"
            style={{
              backgroundColor:
                t.type === "success" ? "#24a148" :
                  t.type === "error" ? "#da1e28" : "#0f62fe",
              fontSize: "0.875rem",
              letterSpacing: "0.16px",
              boxShadow: "0 2px 6px rgba(0,0,0,0.30)",
            }}
          >
            <span>{t.type === "success" ? "✓" : t.type === "error" ? "✕" : "↻"}</span>
            {t.message}
          </div>
        ))}
      </div>
    </div>
  );
}
