"""
Layer 3 — Severity Classifier
Stage 1: Rule-based (fast, deterministic)
Stage 2: LLM augmentation via Gemma 4 E4B (adds nuance and explanation)
"""

from __future__ import annotations
import logging
import json
import re
from typing import Optional

logger = logging.getLogger(__name__)


# ── Stage 1 — Rule Engine ─────────────────────────────────────────────────────

RED_RULES: list[tuple[str, str, float]] = [
    # (vital/lab, type, threshold)
    ("MAP",          "vital",  65.0),
    ("SpO2",         "vital",  90.0),
    ("HR",           "vital", 130.0),
    ("Lactate",      "lab",     4.0),
    ("Creatinine",   "lab",     3.0),
    ("Troponin I",   "lab",    10.0),
    ("pH",           "lab",     7.25),
]

AMBER_RULES: list[tuple[str, str, float]] = [
    ("MAP",          "vital",  70.0),
    ("SpO2",         "vital",  94.0),
    ("HR",           "vital", 110.0),
    ("Lactate",      "lab",     2.0),
    ("Creatinine",   "lab",     1.5),
    ("WBC",          "lab",    12.0),
    ("Procalcitonin","lab",     2.0),
    ("BNP",          "lab",   500.0),
]


def _below_threshold(value: float, vtype: str, threshold: float) -> bool:
    """Return True if value crosses the threshold in the dangerous direction."""
    LOW_METRICS = {"MAP", "SpO2", "pH", "PaO2", "PaO2/FiO2", "Hemoglobin", "Platelets"}
    return value < threshold if vtype in LOW_METRICS else value > threshold


def rule_based_severity(patient: dict) -> tuple[str, list[str]]:
    """
    Stage 1: fast, deterministic severity from thresholds.

    Returns:
        (severity: "RED" | "AMBER" | "GREEN", triggered_rules: list[str])
    """
    vitals = patient.get("vitals_latest", {})
    labs   = {l["test"]: l["value"] for l in patient.get("lab_results", [])}

    red_triggers: list[str] = []

    # Check RED rules
    for metric, mtype, threshold in RED_RULES:
        value = vitals.get(metric) if mtype == "vital" else labs.get(metric)
        if value is not None and _below_threshold(value, metric, threshold):
            direction = "below" if metric in {"MAP", "SpO2", "pH"} else "above"
            red_triggers.append(f"{metric} {value} ({direction} {threshold})")

    if red_triggers:
        return "RED", red_triggers

    amber_triggers: list[str] = []

    # Check AMBER rules
    for metric, mtype, threshold in AMBER_RULES:
        value = vitals.get(metric) if mtype == "vital" else labs.get(metric)
        if value is not None and _below_threshold(value, metric, threshold):
            direction = "below" if metric in {"MAP", "SpO2"} else "above"
            amber_triggers.append(f"{metric} {value} ({direction} {threshold})")

    # Also flag if ≥3 active conditions
    n_conditions = len([c for c in patient.get("conditions", [])
                        if c.get("status") == "active"])
    if n_conditions >= 3:
        amber_triggers.append(f"{n_conditions} active conditions")

    if amber_triggers:
        return "AMBER", amber_triggers

    return "GREEN", []


# ── Stage 2 — LLM Augmentation ──────────────────────────────────────────────

SEVERITY_PROMPT_TEMPLATE = """\
You are a critical care physician reviewing an ICU patient.

<patient_graph_context>
{patient_context}
</patient_graph_context>

<medical_knowledge>
{knowledge_context}
</medical_knowledge>

<rule_engine_result>
Stage 1 classifier: {stage1_severity}
Triggered rules: {triggered_rules}
</rule_engine_result>

Based on ALL of the above, provide your clinical assessment.
Respond ONLY with valid JSON in this exact format:
{{
  "severity": "RED" or "AMBER" or "GREEN",
  "confidence": 0.0 to 1.0,
  "explanation": "3 sentences citing specific data points",
  "evidence": ["specific finding 1", "specific finding 2", "..."],
  "drug_alerts": ["alert 1 if any drug interaction found"],
  "guideline_alerts": ["relevant guideline note if applicable"]
}}
"""


def llm_augment_severity(
    patient_context: str,
    knowledge_context: str,
    stage1_severity: str,
    triggered_rules: list[str],
    llm_fn,  # callable: prompt_str → response_str
) -> dict:
    """
    Stage 2: Ask the LLM to confirm/refine severity and generate explanation.

    Args:
        patient_context:   Text from subgraph_retriever.get_patient_subgraph_text()
        knowledge_context: Text from retriever.get_full_context()
        stage1_severity:   "RED" | "AMBER" | "GREEN"
        triggered_rules:   List of rule strings from Stage 1
        llm_fn:            Callable that takes a prompt string and returns response string

    Returns:
        Parsed dict with severity, confidence, explanation, evidence, alerts
    """
    prompt = SEVERITY_PROMPT_TEMPLATE.format(
        patient_context=patient_context,
        knowledge_context=knowledge_context,
        stage1_severity=stage1_severity,
        triggered_rules=", ".join(triggered_rules) if triggered_rules else "none",
    )

    response_text = llm_fn(prompt)

    # Parse JSON from response
    try:
        # Find JSON block (model may wrap in markdown)
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            result = json.loads(match.group())
        else:
            result = json.loads(response_text)
        return result
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"LLM severity response not valid JSON: {e}")
        logger.debug(f"Raw response: {response_text[:300]}")
        # Fallback: return stage 1 result with explanation from raw text
        return {
            "severity": stage1_severity,
            "confidence": 0.7,
            "explanation": response_text[:500] if response_text else "See rule triggers.",
            "evidence": triggered_rules,
            "drug_alerts": [],
            "guideline_alerts": [],
        }


def classify_severity(
    patient: dict,
    patient_context: str,
    knowledge_context: str,
    llm_fn=None,
) -> dict:
    """
    Full two-stage severity classification.

    Args:
        patient:          Patient dict
        patient_context:  Subgraph text from Layer 2
        knowledge_context: Global KB context from Pipeline A
        llm_fn:           Optional LLM callable for Stage 2 (None = Stage 1 only)

    Returns:
        SeverityResult-compatible dict
    """
    stage1_severity, triggered_rules = rule_based_severity(patient)

    if llm_fn is None:
        # Stage 1 only (used before LLM is loaded)
        return {
            "severity": stage1_severity,
            "confidence": 0.85 if stage1_severity == "GREEN" else 0.90,
            "triggered_rules": triggered_rules,
            "explanation": (
                f"Rule-based classification: {stage1_severity}. "
                + (f"Triggers: {', '.join(triggered_rules)}." if triggered_rules
                   else "All parameters within normal range.")
            ),
            "evidence": triggered_rules,
            "drug_alerts": [],
            "guideline_alerts": [],
        }

    result = llm_augment_severity(
        patient_context=patient_context,
        knowledge_context=knowledge_context,
        stage1_severity=stage1_severity,
        triggered_rules=triggered_rules,
        llm_fn=llm_fn,
    )
    result.setdefault("triggered_rules", triggered_rules)
    return result
