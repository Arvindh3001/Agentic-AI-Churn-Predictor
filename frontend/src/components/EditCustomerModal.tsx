"use client";

import { useState } from "react";
import type { Customer } from "@/types";

const CONTRACT_TYPES = ["Month-to-Month", "One Year", "Two Year"];

interface Props {
  customer: Customer;
  onSave: (
    updates: Partial<
      Pick<
        Customer,
        | "contract_type"
        | "monthly_charges"
        | "num_support_tickets_30d"
        | "feature_adoption_rate"
        | "nps_score"
      >
    >
  ) => Promise<void>;
  onClose: () => void;
}

export default function EditCustomerModal({ customer, onSave, onClose }: Props) {
  const [form, setForm] = useState({
    contract_type: customer.contract_type,
    monthly_charges: customer.monthly_charges,
    num_support_tickets_30d: customer.num_support_tickets_30d ?? 0,
    feature_adoption_pct: Math.round((customer.feature_adoption_rate ?? 0.5) * 100),
    nps_score: customer.nps_score ?? 5,
  });
  const [saving, setSaving] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);

  function set<K extends keyof typeof form>(key: K, value: (typeof form)[K]) {
    setForm((prev) => ({ ...prev, [key]: value }));
  }

  async function handleSave() {
    setSaving(true);
    setFormError(null);
    try {
      await onSave({
        contract_type: form.contract_type,
        monthly_charges: form.monthly_charges,
        num_support_tickets_30d: form.num_support_tickets_30d,
        feature_adoption_rate: form.feature_adoption_pct / 100,
        nps_score: form.nps_score,
      });
    } catch (e) {
      setFormError(e instanceof Error ? e.message : "Save failed");
      setSaving(false);
    }
  }

  const npsColor =
    form.nps_score >= 7 ? "#24a148" :
    form.nps_score >= 5 ? "#ba4e00" :
    "#da1e28";

  const adoptionColor =
    form.feature_adoption_pct >= 60 ? "#24a148" :
    form.feature_adoption_pct >= 30 ? "#ba4e00" :
    "#da1e28";

  return (
    /* Dark scrim overlay */
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      style={{ backgroundColor: "rgba(0,0,0,0.5)" }}
      onClick={onClose}
    >
      {/* Modal — Carbon rectangular (0px radius), white background */}
      <div
        style={{
          backgroundColor: "#ffffff",
          width: "100%",
          maxWidth: "480px",
          margin: "0 16px",
          boxShadow: "0 2px 6px rgba(0,0,0,0.30)",
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal header */}
        <div
          className="flex items-center justify-between px-4 py-4"
          style={{ borderBottom: "1px solid #e0e0e0" }}
        >
          <div>
            <p
              className="font-semibold"
              style={{ fontSize: "1rem", color: "#161616", lineHeight: 1.5 }}
            >
              Edit Customer
            </p>
            <p
              className="font-mono"
              style={{ fontSize: "0.75rem", color: "#525252", letterSpacing: "0.32px", marginTop: "2px" }}
            >
              {customer.customer_id}
            </p>
          </div>
          <button
            onClick={onClose}
            style={{ color: "#525252", padding: "8px", lineHeight: 1 }}
            className="hover:bg-[#f4f4f4] transition-colors"
          >
            <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Form body */}
        <div className="px-4 py-5 space-y-5">
          {/* Contract Type */}
          <div>
            <label className="cds-label">Contract Type</label>
            <select
              value={form.contract_type}
              onChange={(e) => set("contract_type", e.target.value)}
              className="cds-input"
              style={{ borderBottomColor: "#161616" }}
            >
              {CONTRACT_TYPES.map((t) => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
          </div>

          {/* Monthly Charges */}
          <div>
            <label className="cds-label">Monthly Charges ($)</label>
            <input
              type="number"
              min={0}
              max={250}
              step={0.01}
              value={form.monthly_charges}
              onChange={(e) => set("monthly_charges", parseFloat(e.target.value) || 0)}
              className="cds-input"
              style={{ borderBottomColor: "#161616" }}
            />
          </div>

          {/* Support Tickets */}
          <div>
            <label className="cds-label">Support Tickets (last 30 days)</label>
            <input
              type="number"
              min={0}
              max={50}
              value={form.num_support_tickets_30d}
              onChange={(e) => set("num_support_tickets_30d", parseInt(e.target.value) || 0)}
              className="cds-input"
              style={{ borderBottomColor: "#161616" }}
            />
          </div>

          {/* Feature Adoption Rate */}
          <div>
            <label className="cds-label">
              Feature Adoption Rate —{" "}
              <span style={{ color: adoptionColor, fontWeight: 600 }}>{form.feature_adoption_pct}%</span>
            </label>
            <input
              type="range"
              min={0}
              max={100}
              value={form.feature_adoption_pct}
              onChange={(e) => set("feature_adoption_pct", parseInt(e.target.value))}
              className="w-full h-1 mt-2"
              style={{ accentColor: "#0f62fe" }}
            />
            <div
              className="flex justify-between mt-1"
              style={{ fontSize: "0.75rem", color: "#8d8d8d", letterSpacing: "0.32px" }}
            >
              <span>0%</span>
              <span>100%</span>
            </div>
          </div>

          {/* NPS Score */}
          <div>
            <label className="cds-label">
              NPS Score —{" "}
              <span style={{ color: npsColor, fontWeight: 600 }}>{form.nps_score}/10</span>
            </label>
            <input
              type="range"
              min={1}
              max={10}
              value={form.nps_score}
              onChange={(e) => set("nps_score", parseInt(e.target.value))}
              className="w-full h-1 mt-2"
              style={{ accentColor: "#0f62fe" }}
            />
            <div
              className="flex justify-between mt-1"
              style={{ fontSize: "0.75rem", color: "#8d8d8d", letterSpacing: "0.32px" }}
            >
              <span>1 — Detractor</span>
              <span>10 — Promoter</span>
            </div>
          </div>

          {formError && (
            <div
              style={{
                backgroundColor: "#fff1f1",
                borderLeft: "4px solid #da1e28",
                padding: "12px 16px",
                fontSize: "0.875rem",
                color: "#da1e28",
                letterSpacing: "0.16px",
              }}
            >
              {formError}
            </div>
          )}
        </div>

        {/* Footer — Carbon-style button row */}
        <div
          className="flex items-center"
          style={{ borderTop: "1px solid #e0e0e0" }}
        >
          <button
            onClick={handleSave}
            disabled={saving}
            className="cds-btn cds-btn--primary flex-1"
            style={{ justifyContent: "center", padding: "0 16px" }}
          >
            {saving ? "Saving…" : "Save changes"}
          </button>
          <button
            onClick={onClose}
            disabled={saving}
            className="cds-btn cds-btn--secondary flex-1"
            style={{ justifyContent: "center", padding: "0 16px" }}
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}
