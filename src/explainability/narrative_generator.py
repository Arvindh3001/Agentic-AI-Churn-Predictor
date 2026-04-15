"""
Narrative Generator
====================
Converts structured SHAP/LIME explanations and counterfactual interventions
into human-readable plain-English narratives.

Two modes:
    1. Template-based (default) — deterministic, no LLM required.
       Works offline, fast, audit-friendly.

    2. LLM-based (stub, wired fully in Phase 3 Explanation Agent) —
       uses OpenAI / Anthropic to generate richer, contextual narratives.
       Activated by passing llm_provider != None.

Usage:
    from src.explainability.narrative_generator import NarrativeGenerator

    gen = NarrativeGenerator()  # template mode
    text = gen.generate(shap_result, lime_result, cf_result)

    gen_llm = NarrativeGenerator(llm_provider="openai", model="gpt-4o")
    text = gen_llm.generate(shap_result, lime_result, cf_result)
"""

from __future__ import annotations

from typing import Any, Literal

import structlog

logger = structlog.get_logger(__name__)

LLMProvider = Literal["openai", "anthropic", "google"]

# Risk tier thresholds
HIGH_RISK_THRESHOLD = 0.70
MEDIUM_RISK_THRESHOLD = 0.40


class NarrativeGenerator:
    """
    Generates plain-English explanation narratives for churn predictions.

    Args:
        llm_provider: If set, uses an LLM for generation. One of
                      'openai', 'anthropic', 'google'. None = template mode.
        model: LLM model ID (e.g. 'gpt-4o'). Ignored in template mode.
        temperature: LLM sampling temperature.
    """

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        model: str = "gpt-4o",
        temperature: float = 0.3,
    ) -> None:
        self.llm_provider = llm_provider
        self.model = model
        self.temperature = temperature
        self._llm_client: Any = None

        if llm_provider is not None:
            self._init_llm_client()

    def _init_llm_client(self) -> None:
        """Lazy-init LLM client. Fully wired in Phase 3."""
        try:
            if self.llm_provider == "openai":
                from openai import OpenAI
                self._llm_client = OpenAI()
            elif self.llm_provider == "anthropic":
                import anthropic
                self._llm_client = anthropic.Anthropic()
            elif self.llm_provider == "google":
                import google.generativeai as genai
                self._llm_client = genai
            logger.info("LLM client initialised", provider=self.llm_provider, model=self.model)
        except ImportError as exc:
            logger.warning(
                "LLM client init failed — falling back to template mode",
                provider=self.llm_provider,
                error=str(exc),
            )
            self._llm_client = None

    def generate(
        self,
        shap_result: dict[str, Any],
        lime_result: dict[str, Any] | None = None,
        cf_result: dict[str, Any] | None = None,
        customer_context: dict[str, Any] | None = None,
    ) -> str:
        """
        Generate a full explanation narrative.

        Args:
            shap_result: Output from ShapExplainer.explain_instance().
            lime_result: Output from LimeExplainer.explain_instance() (optional).
            cf_result: Output from CounterfactualEngine.generate() (optional).
            customer_context: Optional raw customer feature dict for richer context.

        Returns:
            Plain-English narrative string.
        """
        if self.llm_provider is not None and self._llm_client is not None:
            return self._generate_llm(shap_result, lime_result, cf_result, customer_context)

        return self._generate_template(shap_result, lime_result, cf_result, customer_context)

    # ------------------------------------------------------------------ #
    # Template-based generation
    # ------------------------------------------------------------------ #

    def _generate_template(
        self,
        shap_result: dict[str, Any],
        lime_result: dict[str, Any] | None,
        cf_result: dict[str, Any] | None,
        customer_context: dict[str, Any] | None,
    ) -> str:
        prob = shap_result["churn_probability"]
        risk_label = _risk_label(prob)
        customer_id = cf_result.get("customer_id", "This customer") if cf_result else "This customer"

        # Risk summary sentence
        intro = (
            f"{customer_id} has a **{risk_label} churn risk** "
            f"with a predicted probability of **{prob:.0%}**."
        )

        # SHAP drivers section
        positive_drivers = shap_result.get("top_positive_drivers", [])
        negative_drivers = shap_result.get("top_negative_drivers", [])
        drivers_text = _format_drivers(positive_drivers, negative_drivers)

        # LIME corroboration (if available)
        corroboration = ""
        if lime_result:
            lime_top = lime_result.get("top_positive_drivers", [])
            if lime_top:
                top_lime_feat = _clean_condition(lime_top[0]["condition"])
                corroboration = (
                    f"\n\nA model-agnostic analysis (LIME) corroborates this, "
                    f"highlighting **{top_lime_feat}** as the strongest local driver."
                )

        # Counterfactual interventions (if available)
        interventions_text = ""
        if cf_result and cf_result.get("interventions"):
            interventions_text = "\n\n**Recommended Retention Actions:**\n"
            for i, intervention in enumerate(cf_result["interventions"], 1):
                new_prob = intervention["new_churn_prob"]
                reduction = prob - new_prob
                cost = intervention["cost_usd"]
                interventions_text += (
                    f"  {i}. {intervention['action']} → "
                    f"reduces churn risk to {new_prob:.0%} "
                    f"(↓{reduction:.0%} reduction, estimated cost: ${cost:.0f})\n"
                )

        narrative = intro + "\n\n" + drivers_text + corroboration + interventions_text
        logger.debug("Template narrative generated", customer_id=customer_id, risk=risk_label)
        return narrative.strip()

    # ------------------------------------------------------------------ #
    # LLM-based generation (stub — fully implemented in Phase 3 agent)
    # ------------------------------------------------------------------ #

    def _generate_llm(
        self,
        shap_result: dict[str, Any],
        lime_result: dict[str, Any] | None,
        cf_result: dict[str, Any] | None,
        customer_context: dict[str, Any] | None,
    ) -> str:
        """
        Generate narrative via LLM. Called by the Phase 3 Explanation Agent.
        Falls back to template if LLM call fails.
        """
        prompt = _build_llm_prompt(shap_result, lime_result, cf_result, customer_context)

        try:
            if self.llm_provider == "openai":
                response = self._llm_client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a customer success analyst. Write concise, "
                                "actionable churn risk summaries for account managers. "
                                "Use plain English. No jargon. Max 3 short paragraphs."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                )
                return response.choices[0].message.content.strip()

            elif self.llm_provider == "anthropic":
                response = self._llm_client.messages.create(
                    model=self.model,
                    max_tokens=512,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text.strip()

        except Exception as exc:
            logger.warning(
                "LLM narrative generation failed — falling back to template",
                error=str(exc),
            )

        return self._generate_template(shap_result, lime_result, cf_result, customer_context)


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _risk_label(prob: float) -> str:
    if prob >= HIGH_RISK_THRESHOLD:
        return "HIGH"
    elif prob >= MEDIUM_RISK_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def _format_drivers(
    positive: list[dict[str, Any]],
    negative: list[dict[str, Any]],
) -> str:
    lines = ["**Key churn drivers (SHAP analysis):**"]

    if positive:
        lines.append("Factors *increasing* churn risk:")
        for d in positive[:3]:
            feat = d["feature"].replace("_", " ").title()
            val = d["shap_value"]
            lines.append(f"  • {feat} (contribution: +{val:.3f})")

    if negative:
        lines.append("Factors *reducing* churn risk:")
        for d in negative[:3]:
            feat = d["feature"].replace("_", " ").title()
            val = abs(d["shap_value"])
            lines.append(f"  • {feat} (contribution: −{val:.3f})")

    return "\n".join(lines)


def _clean_condition(condition: str) -> str:
    """Extract a readable feature name from a LIME condition string."""
    # e.g. "0.50 < feature_adoption_rate <= 0.80" → "Feature Adoption Rate"
    for part in condition.split():
        if "_" in part and not any(c.isdigit() for c in part):
            return part.replace("_", " ").title()
    return condition


def _build_llm_prompt(
    shap_result: dict[str, Any],
    lime_result: dict[str, Any] | None,
    cf_result: dict[str, Any] | None,
    customer_context: dict[str, Any] | None,
) -> str:
    """Build a structured prompt for the LLM narrative generator."""
    prob = shap_result["churn_probability"]
    pos_drivers = [
        f"{d['feature']} (+{d['shap_value']:.3f})"
        for d in shap_result.get("top_positive_drivers", [])[:3]
    ]
    neg_drivers = [
        f"{d['feature']} ({d['shap_value']:.3f})"
        for d in shap_result.get("top_negative_drivers", [])[:3]
    ]

    interventions_str = ""
    if cf_result and cf_result.get("interventions"):
        interventions_str = "\n".join(
            f"  - {i['action']} (reduces churn to {i['new_churn_prob']:.0%}, cost ${i['cost_usd']:.0f})"
            for i in cf_result["interventions"]
        )

    customer_id = cf_result.get("customer_id", "unknown") if cf_result else "unknown"

    return f"""Customer ID: {customer_id}
Churn probability: {prob:.0%}

Top risk factors (SHAP):
  Increasing churn: {', '.join(pos_drivers) or 'none'}
  Reducing churn: {', '.join(neg_drivers) or 'none'}

Recommended interventions:
{interventions_str or '  None available'}

Customer context: {customer_context or 'not provided'}

Write a 2-3 sentence plain-English summary for an account manager explaining:
1. Why this customer is at risk
2. The single most impactful action they should take
"""
