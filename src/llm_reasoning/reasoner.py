"""
Layer 3 — Summarizer + Reasoner
MedGemma: clinical summarization
Gemma 4 E4B: clinical reasoning + explanation
"""

from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

# ── Prompt Templates ──────────────────────────────────────────────────────────

SUMMARIZATION_PROMPT = """\
You are a critical care physician assistant. Summarize this ICU patient's current clinical state.
Base your summary ONLY on the data provided. Do not add information not present below.

{patient_context}

Write a concise clinical summary covering:
1. Primary diagnosis and acuity level
2. Key abnormal findings (vitals and labs)
3. Current medications and their indications
4. Clinical trajectory (improving / stable / deteriorating)

Format: 3-4 sentences, clinical tone, no bullet points."""


REASONING_PROMPT = """\
You are a clinical decision support AI for an ICU.
Your reasoning must be grounded in the provided context only.
Cite specific data points from the graph in your explanation.
Do not introduce clinical facts not present in the context.

<patient_graph_context>
{patient_context}
</patient_graph_context>

<medical_knowledge_context>
{knowledge_context}
</medical_knowledge_context>

<severity_classification>
Severity: {severity}
Confidence: {confidence}
Rule triggers: {rule_triggers}
</severity_classification>

Question: {query}

Respond with a structured explanation:
- Severity assessment and rationale (2-3 sentences citing specific values)
- Key clinical concerns (bullet points)
- Drug interactions or guideline alerts (if any)
- Recommended monitoring focus"""


ENTITY_EXTRACTION_PROMPT = """\
You are a clinical entity extractor.
Extract all medical entities and relationships from the clinical note below.
Return ONLY valid JSON. No explanation, no markdown, just the JSON object.

Required schema:
{{
  "conditions": [{{"name": "string", "status": "active|resolved|chronic"}}],
  "medications": [{{"name": "string", "dose": "string", "route": "string", "indication": "string"}}],
  "lab_results": [{{"test": "string", "value": 0.0, "unit": "string", "flag": "HIGH|LOW|NORMAL"}}],
  "vitals": [{{"type": "string", "value": 0.0, "unit": "string"}}],
  "relations": [{{"from": "string", "relation": "string", "to": "string"}}]
}}

Clinical note:
{clinical_text}"""


# ── Summarizer ────────────────────────────────────────────────────────────────

def generate_summary(
    patient_context: str,
    model_manager,
    max_tokens: int = 400,
) -> str:
    """
    Generate a clinical summary of the patient using MedGemma.

    Args:
        patient_context: Subgraph text from Layer 2
        model_manager:   ModelManager instance
        max_tokens:      Max output tokens

    Returns:
        Clinical summary string
    """
    prompt = SUMMARIZATION_PROMPT.format(patient_context=patient_context)
    summary = model_manager.medgemma_generate(prompt, max_tokens=max_tokens, temperature=0.3)
    return summary.strip()


# ── Reasoner ─────────────────────────────────────────────────────────────────

def generate_explanation(
    patient_context: str,
    knowledge_context: str,
    severity: str,
    confidence: float,
    rule_triggers: list[str],
    query: str,
    model_manager,
    max_tokens: int = 600,
) -> str:
    """
    Generate a clinical reasoning explanation using Gemma 4 E4B.

    Args:
        patient_context:   Subgraph text
        knowledge_context: PrimeKG + guideline context
        severity:          "RED" | "AMBER" | "GREEN"
        confidence:        Float 0–1
        rule_triggers:     List of triggered clinical rules
        query:             User's question or "Explain this patient's severity"
        model_manager:     ModelManager instance

    Returns:
        Explanation string
    """
    prompt = REASONING_PROMPT.format(
        patient_context=patient_context,
        knowledge_context=knowledge_context,
        severity=severity,
        confidence=f"{confidence:.0%}",
        rule_triggers=", ".join(rule_triggers) if rule_triggers else "none",
        query=query,
    )
    explanation = model_manager.gemma4_generate(prompt, max_tokens=max_tokens, temperature=0.4)
    return explanation.strip()


# ── Entity Extraction ─────────────────────────────────────────────────────────

def extract_entities_from_note(
    clinical_text: str,
    model_manager,
    max_retries: int = 2,
) -> dict:
    """
    Use MedGemma to extract structured entities from an unstructured clinical note.
    Retries up to max_retries times if JSON parsing fails.

    Returns:
        Dict with keys: conditions, medications, lab_results, vitals, relations
    """
    import json, re

    prompt = ENTITY_EXTRACTION_PROMPT.format(clinical_text=clinical_text[:3000])
    empty_result = {
        "conditions": [], "medications": [],
        "lab_results": [], "vitals": [], "relations": [],
    }

    for attempt in range(max_retries + 1):
        raw = model_manager.medgemma_generate(prompt, max_tokens=800, temperature=0.1)
        try:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                return json.loads(match.group())
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            if attempt < max_retries:
                logger.warning(f"Entity extraction JSON parse failed (attempt {attempt+1}), retrying…")
            else:
                logger.error("Entity extraction failed after all retries. Returning empty.")
    return empty_result
