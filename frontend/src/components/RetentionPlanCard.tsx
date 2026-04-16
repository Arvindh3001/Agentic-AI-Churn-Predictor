import type { RetentionPlan } from "@/types";

interface Props {
  plan: RetentionPlan;
}

export default function RetentionPlanCard({ plan }: Props) {
  const roi = plan.estimated_roi;
  const roiPositive = roi >= 0;

  return (
    <div style={{ backgroundColor: "#f4f4f4" }} className="p-4">
      {/* Header */}
      <div className="flex items-start justify-between mb-1">
        <p
          className="font-semibold"
          style={{ fontSize: "0.875rem", color: "#161616", letterSpacing: "0.16px" }}
        >
          Retention Plan
        </p>
        {plan.pending_hitl && (
          <span
            className="inline-flex items-center font-medium"
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
      <p style={{ fontSize: "0.75rem", color: "#525252", letterSpacing: "0.32px" }} className="mb-4">
        A/B group:{" "}
        <span style={{ color: "#161616", fontWeight: 600 }}>{plan.ab_group}</span>
        {plan.crm_action_id && (
          <>
            {" · "}
            <span className="font-mono">{plan.crm_action_id}</span>
          </>
        )}
      </p>

      {/* Summary stats — white tiles on gray-10 card */}
      <div className="grid grid-cols-3 gap-px mb-4" style={{ backgroundColor: "#c6c6c6" }}>
        {[
          {
            label: "Total Cost",
            value: `$${plan.total_cost_usd.toFixed(0)}`,
            color: "#161616",
          },
          {
            label: "Revenue Saved",
            value: `$${plan.estimated_revenue_saved_usd.toFixed(0)}`,
            color: "#24a148",
          },
          {
            label: "ROI",
            value: `${roiPositive ? "+" : ""}${(roi * 100).toFixed(1)}%`,
            color: roiPositive ? "#24a148" : "#da1e28",
          },
        ].map(({ label, value, color }) => (
          <div key={label} style={{ backgroundColor: "#ffffff" }} className="p-3 text-center">
            <p style={{ fontSize: "0.75rem", color: "#525252", letterSpacing: "0.32px" }} className="mb-1">
              {label}
            </p>
            <p
              className="font-semibold"
              style={{ fontSize: "1.125rem", color, lineHeight: "1.4" }}
            >
              {value}
            </p>
          </div>
        ))}
      </div>

      {/* Action rows */}
      {plan.selected_actions.length > 0 ? (
        <div style={{ borderTop: "1px solid #c6c6c6" }}>
          {plan.selected_actions.map((action, i) => {
            const label = action.label ?? action.action ?? "Retention action";
            return (
              <div
                key={i}
                className="flex items-start justify-between gap-4 py-3"
                style={{ borderBottom: "1px solid #e0e0e0" }}
              >
                <div className="flex-1">
                  <p
                    style={{
                      fontSize: "0.875rem",
                      color: "#161616",
                      letterSpacing: "0.16px",
                      fontWeight: 400,
                    }}
                  >
                    {label}
                  </p>
                  {action.days_to_effect && (
                    <p style={{ fontSize: "0.75rem", color: "#525252", letterSpacing: "0.32px" }} className="mt-0.5">
                      Effect in ~{action.days_to_effect} days
                    </p>
                  )}
                </div>
                <div className="text-right shrink-0">
                  <p
                    className="font-semibold"
                    style={{ fontSize: "0.875rem", color: "#0f62fe", letterSpacing: "0.16px" }}
                  >
                    ${action.cost_usd.toFixed(0)}
                  </p>
                  {action.prob_reduction && (
                    <p style={{ fontSize: "0.75rem", color: "#24a148", letterSpacing: "0.32px" }}>
                      -{(action.prob_reduction * 100).toFixed(1)}% churn
                    </p>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <p style={{ fontSize: "0.875rem", color: "#6f6f6f", letterSpacing: "0.16px" }}>
          No actions selected
        </p>
      )}
    </div>
  );
}
