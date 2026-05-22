"""
ICU Copilot — Shared Data Schemas (Pydantic Models)
All modules import from here to ensure consistency.
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum


class SeverityColor(str, Enum):
    RED = "RED"
    AMBER = "AMBER"
    GREEN = "GREEN"


class LabFlag(str, Enum):
    HIGH = "HIGH"
    LOW = "LOW"
    NORMAL = "NORMAL"
    CRITICAL = "CRITICAL"


class ConditionStatus(str, Enum):
    ACTIVE = "active"
    RESOLVED = "resolved"
    CHRONIC = "chronic"


# ── Per-patient entity models ────────────────────────────────────────────────

class Demographics(BaseModel):
    name: str
    age: int
    sex: str                        # "M" | "F"
    weight_kg: Optional[float] = None
    bed: str                        # e.g. "3A"
    admission_date: Optional[str] = None


class Condition(BaseModel):
    name: str
    icd_code: Optional[str] = None
    status: ConditionStatus = ConditionStatus.ACTIVE
    onset_date: Optional[str] = None
    primekg_node_id: Optional[str] = None   # set during graph linking


class Medication(BaseModel):
    name: str
    dose: Optional[str] = None
    route: Optional[str] = None          # "IV", "PO", "SC", …
    frequency: Optional[str] = None
    indication: Optional[str] = None
    rxnorm_code: Optional[str] = None
    primekg_node_id: Optional[str] = None


class LabResult(BaseModel):
    test: str
    value: float
    unit: str
    flag: LabFlag = LabFlag.NORMAL
    timestamp: Optional[str] = None
    loinc_code: Optional[str] = None


class Vital(BaseModel):
    type: str          # "MAP" | "HR" | "SpO2" | "RR" | "Temp" | "SBP" | "DBP"
    value: float
    unit: str
    timestamp: Optional[str] = None
    is_critical: bool = False


class Procedure(BaseModel):
    name: str
    timestamp: Optional[str] = None
    outcome: Optional[str] = None


class Relation(BaseModel):
    """Relation extracted from unstructured text by MedGemma."""
    from_entity: str = Field(alias="from")
    relation: str      # "treats" | "prescribed_for" | "causes" | …
    to_entity: str = Field(alias="to")

    model_config = {"populate_by_name": True}


# ── Aggregate patient state ──────────────────────────────────────────────────

class PatientState(BaseModel):
    patient_id: str
    demographics: Demographics
    conditions: list[Condition] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    lab_results: list[LabResult] = Field(default_factory=list)
    vitals: list[Vital] = Field(default_factory=list)
    procedures: list[Procedure] = Field(default_factory=list)
    raw_notes: list[str] = Field(default_factory=list)
    extracted_relations: list[Relation] = Field(default_factory=list)
    last_updated: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    def latest_vital(self, vital_type: str) -> Optional[float]:
        """Return the most recent value for a given vital type."""
        matching = [v for v in self.vitals if v.type.upper() == vital_type.upper()]
        if not matching:
            return None
        # Sort by timestamp descending if available, else take last
        timestamped = [v for v in matching if v.timestamp]
        if timestamped:
            return sorted(timestamped, key=lambda v: v.timestamp, reverse=True)[0].value
        return matching[-1].value

    def latest_lab(self, test_name: str) -> Optional[float]:
        """Return the most recent value for a given lab test."""
        matching = [l for l in self.lab_results if l.test.lower() == test_name.lower()]
        if not matching:
            return None
        timestamped = [l for l in matching if l.timestamp]
        if timestamped:
            return sorted(timestamped, key=lambda l: l.timestamp, reverse=True)[0].value
        return matching[-1].value

    def active_conditions(self) -> list[Condition]:
        return [c for c in self.conditions if c.status == ConditionStatus.ACTIVE]


# ── Reasoning outputs ────────────────────────────────────────────────────────

class SeverityResult(BaseModel):
    severity: SeverityColor
    confidence: float = Field(ge=0.0, le=1.0)
    triggered_rules: list[str] = Field(default_factory=list)
    explanation: str = ""
    evidence: list[str] = Field(default_factory=list)
    drug_alerts: list[str] = Field(default_factory=list)
    guideline_alerts: list[str] = Field(default_factory=list)


class ClinicalSummary(BaseModel):
    patient_id: str
    summary_text: str
    severity: SeverityResult
    generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
