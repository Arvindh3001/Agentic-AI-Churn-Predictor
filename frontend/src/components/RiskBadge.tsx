import type { RiskTier } from "@/types";

// IBM Carbon tag colors — pill shape (24px radius, the one exception to 0px rule)
const STYLES: Record<RiskTier, { bg: string; color: string }> = {
  CRITICAL: { bg: "#fff1f1", color: "#da1e28" }, // Red 10 / Red 60
  HIGH:     { bg: "#fff2e8", color: "#ba4e00" }, // Orange 10 / Orange 60
  MEDIUM:   { bg: "#fcf4d6", color: "#684e00" }, // Yellow 10 / Yellow 70
  LOW:      { bg: "#defbe6", color: "#044317" }, // Green 10 / Green 70
};

interface Props {
  tier: RiskTier | string;
  size?: "sm" | "md";
}

export default function RiskBadge({ tier, size = "md" }: Props) {
  const style = STYLES[tier as RiskTier] ?? { bg: "#f4f4f4", color: "#525252" };
  const padding = size === "sm" ? "2px 8px" : "4px 8px";
  const fontSize = size === "sm" ? "0.75rem" : "0.875rem";

  return (
    <span
      className="inline-flex items-center font-semibold whitespace-nowrap"
      style={{
        backgroundColor: style.bg,
        color: style.color,
        borderRadius: "24px", // Carbon pill — only exception to 0px rule
        padding,
        fontSize,
        lineHeight: "1.33",
        letterSpacing: "0.32px",
      }}
    >
      {tier}
    </span>
  );
}
