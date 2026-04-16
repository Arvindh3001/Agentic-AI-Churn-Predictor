// Shared TypeScript types for the Churn Intelligence Platform frontend

export type RiskTier = "CRITICAL" | "HIGH" | "MEDIUM" | "LOW";

export interface Customer {
  customer_id: string;
  monthly_charges: number;
  tenure_months: number;
  contract_type: string;
  num_support_tickets_30d?: number;
  feature_adoption_rate?: number;
  nps_score?: number;
  login_frequency_30d?: number;
  churn: number; // 1 = churned, 0 = active
}

export interface RiskFactor {
  feature: string;
  label: string;
  shap_value: number;
  direction?: "increases_risk" | "reduces_risk";
}

export interface RetentionAction {
  action?: string;
  label?: string;
  cost_usd: number;
  prob_reduction?: number;
  days_to_effect?: number;
  effort?: number;
  feasibility_score?: number;
}

export interface RetentionPlan {
  selected_actions: RetentionAction[];
  total_cost_usd: number;
  estimated_revenue_saved_usd: number;
  estimated_roi: number;
  ab_group: string;
  crm_action_id: string;
  confidence: number;
  pending_hitl: boolean;
}

export interface HITLDecision {
  status: "approved" | "rejected" | "auto_approved";
  decided_by: string;
  decided_at: string;
  notes: string;
}

export interface PredictionResult {
  churn_probability: number;
  risk_tier: RiskTier;
  confidence_interval: [number, number];
  model_version: string;
  is_uncertain: boolean;
}

export interface AnalysisResult {
  run_id: string;
  status: string;
  customer_id: string;
  mode?: string;
  churn_probability?: number;
  risk_tier?: RiskTier;
  confidence_interval?: [number, number];
  narrative?: string;
  top_risk_factors?: RiskFactor[];
  retention_plan?: RetentionPlan;
  hitl_decision?: HITLDecision;
  completed_steps: string[];
  errors: string[];
}

export interface PipelineEvent {
  step: string;
  status: string;
  message?: string;
  // Final payload fields
  prediction?: PredictionResult;
  explanation?: {
    narrative_text: string;
    top_risk_factors: RiskFactor[];
  };
  retention_plan?: RetentionPlan;
  hitl_decision?: HITLDecision;
  completed_steps?: string[];
  errors?: string[];
}

export interface AuthToken {
  access_token: string;
  token_type: string;
  expires_in: number;
  username: string;
  role: string;
}
