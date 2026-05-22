"""
Pipeline B — Synthetic ICU Patient Generator
Since Java/Synthea is not installed, we generate realistic ICU patient data
directly in Python using carefully crafted clinical templates.

Generates 20 patients across 5 ICU scenarios with realistic vitals,
labs, medications, diagnoses, and clinical notes.

Usage:
    python scripts/generate_patients.py
"""

from __future__ import annotations
import json
import random
import logging
from pathlib import Path
from datetime import datetime, timedelta
from faker import Faker

logger = logging.getLogger(__name__)
fake = Faker()
random.seed(42)

PATIENTS_DIR = Path(__file__).parents[2] / "data" / "patients"

# ── Clinical scenario templates ───────────────────────────────────────────────

SCENARIOS = [
    # (name, conditions, meds, severity, lab_profile, vital_profile)
    {
        "name": "septic_shock",
        "severity": "RED",
        "conditions": [
            {"name": "Septic shock", "icd_code": "R65.21", "status": "active"},
            {"name": "Acute kidney injury", "icd_code": "N17.9", "status": "active"},
            {"name": "Type 2 diabetes mellitus", "icd_code": "E11.9", "status": "chronic"},
        ],
        "medications": [
            {"name": "Norepinephrine", "dose": "0.15 mcg/kg/min", "route": "IV", "indication": "Septic shock"},
            {"name": "Meropenem", "dose": "1g q8h", "route": "IV", "indication": "Sepsis"},
            {"name": "Hydrocortisone", "dose": "200 mg/day", "route": "IV", "indication": "Refractory shock"},
            {"name": "Pantoprazole", "dose": "40 mg daily", "route": "IV", "indication": "Stress ulcer prophylaxis"},
        ],
        "labs": [
            {"test": "Lactate", "value": 4.2, "unit": "mmol/L", "flag": "HIGH"},
            {"test": "Creatinine", "value": 3.1, "unit": "mg/dL", "flag": "HIGH"},
            {"test": "WBC", "value": 18.5, "unit": "10^3/uL", "flag": "HIGH"},
            {"test": "Procalcitonin", "value": 28.4, "unit": "ng/mL", "flag": "HIGH"},
            {"test": "Hemoglobin", "value": 9.2, "unit": "g/dL", "flag": "LOW"},
            {"test": "Platelets", "value": 88, "unit": "10^3/uL", "flag": "LOW"},
            {"test": "CRP", "value": 245, "unit": "mg/L", "flag": "HIGH"},
            {"test": "Glucose", "value": 198, "unit": "mg/dL", "flag": "HIGH"},
        ],
        "vitals": {"MAP": 56, "HR": 118, "SpO2": 91, "RR": 26, "Temp": 38.9, "SBP": 82, "DBP": 45},
    },
    {
        "name": "ards_postop",
        "severity": "AMBER",
        "conditions": [
            {"name": "Acute respiratory distress syndrome", "icd_code": "J80", "status": "active"},
            {"name": "Post-operative state", "icd_code": "Z48.89", "status": "active"},
            {"name": "Hypertension", "icd_code": "I10", "status": "chronic"},
        ],
        "medications": [
            {"name": "Propofol", "dose": "10–50 mcg/kg/min", "route": "IV", "indication": "Sedation"},
            {"name": "Fentanyl", "dose": "25 mcg/h", "route": "IV", "indication": "Analgesia"},
            {"name": "Cisatracurium", "dose": "0.1 mg/kg/h", "route": "IV", "indication": "Neuromuscular blockade"},
            {"name": "Piperacillin-tazobactam", "dose": "4.5g q6h", "route": "IV", "indication": "HAP prophylaxis"},
        ],
        "labs": [
            {"test": "PaO2/FiO2", "value": 142, "unit": "mmHg", "flag": "LOW"},
            {"test": "WBC", "value": 13.2, "unit": "10^3/uL", "flag": "HIGH"},
            {"test": "Procalcitonin", "value": 4.1, "unit": "ng/mL", "flag": "HIGH"},
            {"test": "Albumin", "value": 2.4, "unit": "g/dL", "flag": "LOW"},
            {"test": "CRP", "value": 122, "unit": "mg/L", "flag": "HIGH"},
            {"test": "Creatinine", "value": 1.4, "unit": "mg/dL", "flag": "NORMAL"},
            {"test": "Lactate", "value": 1.9, "unit": "mmol/L", "flag": "NORMAL"},
        ],
        "vitals": {"MAP": 68, "HR": 92, "SpO2": 93, "RR": 20, "Temp": 37.8, "SBP": 102, "DBP": 61},
    },
    {
        "name": "stable_diabetic",
        "severity": "GREEN",
        "conditions": [
            {"name": "Type 2 diabetes mellitus", "icd_code": "E11.9", "status": "chronic"},
            {"name": "Hypertension", "icd_code": "I10", "status": "chronic"},
            {"name": "Hyperlipidemia", "icd_code": "E78.5", "status": "chronic"},
        ],
        "medications": [
            {"name": "Insulin glargine", "dose": "20 units", "route": "SC", "indication": "Diabetes"},
            {"name": "Metformin", "dose": "500 mg BID", "route": "PO", "indication": "Diabetes"},
            {"name": "Lisinopril", "dose": "10 mg daily", "route": "PO", "indication": "Hypertension"},
            {"name": "Atorvastatin", "dose": "40 mg daily", "route": "PO", "indication": "Hyperlipidemia"},
        ],
        "labs": [
            {"test": "Glucose", "value": 142, "unit": "mg/dL", "flag": "HIGH"},
            {"test": "HbA1c", "value": 8.1, "unit": "%", "flag": "HIGH"},
            {"test": "Creatinine", "value": 1.1, "unit": "mg/dL", "flag": "NORMAL"},
            {"test": "WBC", "value": 8.4, "unit": "10^3/uL", "flag": "NORMAL"},
            {"test": "Hemoglobin", "value": 13.2, "unit": "g/dL", "flag": "NORMAL"},
            {"test": "Potassium", "value": 4.1, "unit": "mEq/L", "flag": "NORMAL"},
            {"test": "Lactate", "value": 1.1, "unit": "mmol/L", "flag": "NORMAL"},
        ],
        "vitals": {"MAP": 87, "HR": 74, "SpO2": 98, "RR": 16, "Temp": 36.8, "SBP": 128, "DBP": 76},
    },
    {
        "name": "acute_mi_cardiogenic_shock",
        "severity": "RED",
        "conditions": [
            {"name": "ST-elevation myocardial infarction", "icd_code": "I21.9", "status": "active"},
            {"name": "Cardiogenic shock", "icd_code": "R57.0", "status": "active"},
            {"name": "Atrial fibrillation", "icd_code": "I48.91", "status": "active"},
        ],
        "medications": [
            {"name": "Dobutamine", "dose": "5 mcg/kg/min", "route": "IV", "indication": "Cardiogenic shock"},
            {"name": "Heparin", "dose": "1000 units/h", "route": "IV", "indication": "STEMI/Afib"},
            {"name": "Aspirin", "dose": "81 mg", "route": "PO", "indication": "Antiplatelet"},
            {"name": "Amiodarone", "dose": "150 mg bolus then 1 mg/min", "route": "IV", "indication": "Afib with RVR"},
        ],
        "labs": [
            {"test": "Troponin I", "value": 48.2, "unit": "ng/mL", "flag": "HIGH"},
            {"test": "BNP", "value": 1820, "unit": "pg/mL", "flag": "HIGH"},
            {"test": "Lactate", "value": 5.1, "unit": "mmol/L", "flag": "HIGH"},
            {"test": "Creatinine", "value": 2.4, "unit": "mg/dL", "flag": "HIGH"},
            {"test": "CK-MB", "value": 312, "unit": "IU/L", "flag": "HIGH"},
            {"test": "Potassium", "value": 5.6, "unit": "mEq/L", "flag": "HIGH"},
            {"test": "pH", "value": 7.22, "unit": "", "flag": "LOW"},
        ],
        "vitals": {"MAP": 52, "HR": 128, "SpO2": 89, "RR": 28, "Temp": 36.2, "SBP": 74, "DBP": 42},
    },
    {
        "name": "cap_pneumonia",
        "severity": "AMBER",
        "conditions": [
            {"name": "Community-acquired pneumonia", "icd_code": "J18.9", "status": "active"},
            {"name": "Respiratory failure", "icd_code": "J96.01", "status": "active"},
            {"name": "COPD", "icd_code": "J44.1", "status": "chronic"},
        ],
        "medications": [
            {"name": "Ceftriaxone", "dose": "2g daily", "route": "IV", "indication": "CAP"},
            {"name": "Azithromycin", "dose": "500 mg daily", "route": "IV", "indication": "Atypical coverage"},
            {"name": "Dexamethasone", "dose": "6 mg daily", "route": "IV", "indication": "COVID/CAP"},
            {"name": "Ipratropium-albuterol", "dose": "3 mL q4h", "route": "Nebulizer", "indication": "COPD exacerbation"},
        ],
        "labs": [
            {"test": "WBC", "value": 16.8, "unit": "10^3/uL", "flag": "HIGH"},
            {"test": "CRP", "value": 186, "unit": "mg/L", "flag": "HIGH"},
            {"test": "Procalcitonin", "value": 3.8, "unit": "ng/mL", "flag": "HIGH"},
            {"test": "Lactate", "value": 2.1, "unit": "mmol/L", "flag": "NORMAL"},
            {"test": "PaO2", "value": 58, "unit": "mmHg", "flag": "LOW"},
            {"test": "Creatinine", "value": 1.2, "unit": "mg/dL", "flag": "NORMAL"},
            {"test": "Hemoglobin", "value": 11.8, "unit": "g/dL", "flag": "LOW"},
        ],
        "vitals": {"MAP": 72, "HR": 102, "SpO2": 93, "RR": 24, "Temp": 38.5, "SBP": 108, "DBP": 62},
    },
]

# Beds and male/female alternation
BEDS = ["1A", "1B", "2A", "2B", "3A", "3B", "4A", "4B", "5A", "5B",
        "6A", "6B", "7A", "7B", "8A", "8B", "9A", "9B", "10A", "10B"]
AGE_RANGES = {
    "septic_shock":              (55, 80),
    "ards_postop":               (45, 75),
    "stable_diabetic":           (50, 70),
    "acute_mi_cardiogenic_shock":(60, 85),
    "cap_pneumonia":             (50, 78),
}


def _add_noise(value: float, pct: float = 0.08) -> float:
    """Add ±pct% noise to a numeric value."""
    return round(value * (1 + random.uniform(-pct, pct)), 2)


def _generate_vitals_timeseries(base_vitals: dict, hours: int = 24) -> list[dict]:
    """Generate hourly vital sign readings with realistic trends."""
    ts: list[dict] = []
    base_time = datetime.utcnow() - timedelta(hours=hours)
    current = {k: float(v) for k, v in base_vitals.items()}

    for h in range(hours):
        reading = {
            "timestamp": (base_time + timedelta(hours=h)).isoformat(),
        }
        for k, v in current.items():
            reading[k] = round(_add_noise(v, pct=0.05), 1)
            # Slight drift
            current[k] = max(40, v + random.uniform(-0.5, 0.5))
        ts.append(reading)
    return ts


def _generate_note(patient: dict, scenario: dict) -> str:
    """Generate a realistic clinical progress note."""
    cond_names = ", ".join(c["name"] for c in scenario["conditions"][:2])
    med_names  = ", ".join(m["name"] for m in scenario["medications"][:3])
    vitals     = scenario["vitals"]
    labs       = {l["test"]: l["value"] for l in scenario["labs"][:4]}

    lab_str = ", ".join(f"{k}: {v}" for k, v in labs.items())
    return (
        f"ICU Progress Note — {patient['demographics']['name']}, "
        f"{patient['demographics']['age']}y {patient['demographics']['sex']}, "
        f"Bed {patient['demographics']['bed']}\n\n"
        f"S: Patient admitted with {cond_names}. "
        f"{'Hemodynamically unstable.' if scenario['severity'] == 'RED' else 'Hemodynamically stable.'} "
        f"Family updated regarding prognosis.\n\n"
        f"O: Vitals — MAP {vitals['MAP']} mmHg, HR {vitals['HR']} bpm, "
        f"SpO2 {vitals['SpO2']}%, RR {vitals['RR']}, Temp {vitals['Temp']}°C.\n"
        f"Labs — {lab_str}.\n\n"
        f"A: {cond_names}. "
        f"{'Severity: critical, multi-organ involvement.' if scenario['severity'] == 'RED' else 'Severity: moderate, monitored closely.'}\n\n"
        f"P: Continue {med_names}. "
        f"{'Escalate vasopressor support if MAP <65 mmHg.' if scenario['severity'] == 'RED' else 'Wean supplemental oxygen as tolerated.'} "
        f"Reassess q4h. Multidisciplinary team review at 08:00."
    )


# ── Additional stable scenarios for variety ───────────────────────────────────

STABLE_EXTRAS = [
    {
        "name": "post_cardiac_surgery",
        "severity": "GREEN",
        "conditions": [
            {"name": "Post-CABG recovery", "icd_code": "Z95.1", "status": "active"},
            {"name": "Hypertension", "icd_code": "I10", "status": "chronic"},
        ],
        "medications": [
            {"name": "Aspirin", "dose": "81 mg daily", "route": "PO", "indication": "Antiplatelet"},
            {"name": "Metoprolol", "dose": "25 mg BID", "route": "PO", "indication": "Rate control"},
            {"name": "Pantoprazole", "dose": "40 mg daily", "route": "PO", "indication": "GI prophylaxis"},
        ],
        "labs": [
            {"test": "Hemoglobin", "value": 11.4, "unit": "g/dL", "flag": "LOW"},
            {"test": "Creatinine", "value": 1.0, "unit": "mg/dL", "flag": "NORMAL"},
            {"test": "WBC", "value": 9.2, "unit": "10^3/uL", "flag": "NORMAL"},
            {"test": "Troponin I", "value": 0.8, "unit": "ng/mL", "flag": "HIGH"},
            {"test": "Potassium", "value": 4.0, "unit": "mEq/L", "flag": "NORMAL"},
        ],
        "vitals": {"MAP": 82, "HR": 72, "SpO2": 97, "RR": 15, "Temp": 36.9, "SBP": 122, "DBP": 72},
    },
    {
        "name": "chf_monitoring",
        "severity": "GREEN",
        "conditions": [
            {"name": "Congestive heart failure", "icd_code": "I50.9", "status": "chronic"},
            {"name": "Chronic kidney disease stage 3", "icd_code": "N18.3", "status": "chronic"},
        ],
        "medications": [
            {"name": "Furosemide", "dose": "40 mg BID", "route": "IV", "indication": "Volume overload"},
            {"name": "Lisinopril", "dose": "10 mg daily", "route": "PO", "indication": "Heart failure"},
            {"name": "Carvedilol", "dose": "6.25 mg BID", "route": "PO", "indication": "Heart failure"},
        ],
        "labs": [
            {"test": "BNP", "value": 380, "unit": "pg/mL", "flag": "HIGH"},
            {"test": "Creatinine", "value": 1.6, "unit": "mg/dL", "flag": "HIGH"},
            {"test": "Potassium", "value": 4.5, "unit": "mEq/L", "flag": "NORMAL"},
            {"test": "Sodium", "value": 136, "unit": "mEq/L", "flag": "NORMAL"},
            {"test": "Hemoglobin", "value": 12.1, "unit": "g/dL", "flag": "NORMAL"},
        ],
        "vitals": {"MAP": 78, "HR": 80, "SpO2": 96, "RR": 18, "Temp": 36.7, "SBP": 118, "DBP": 68},
    },
    {
        "name": "stroke_rehab",
        "severity": "GREEN",
        "conditions": [
            {"name": "Ischemic stroke", "icd_code": "I63.9", "status": "active"},
            {"name": "Hypertension", "icd_code": "I10", "status": "chronic"},
            {"name": "Dysphagia", "icd_code": "R13.10", "status": "active"},
        ],
        "medications": [
            {"name": "Clopidogrel", "dose": "75 mg daily", "route": "PO", "indication": "Stroke prevention"},
            {"name": "Amlodipine", "dose": "5 mg daily", "route": "PO", "indication": "Hypertension"},
            {"name": "Enoxaparin", "dose": "40 mg daily", "route": "SC", "indication": "DVT prophylaxis"},
        ],
        "labs": [
            {"test": "INR", "value": 1.1, "unit": "", "flag": "NORMAL"},
            {"test": "Glucose", "value": 118, "unit": "mg/dL", "flag": "NORMAL"},
            {"test": "Creatinine", "value": 0.9, "unit": "mg/dL", "flag": "NORMAL"},
            {"test": "WBC", "value": 7.8, "unit": "10^3/uL", "flag": "NORMAL"},
            {"test": "Hemoglobin", "value": 13.5, "unit": "g/dL", "flag": "NORMAL"},
        ],
        "vitals": {"MAP": 90, "HR": 68, "SpO2": 98, "RR": 14, "Temp": 36.6, "SBP": 132, "DBP": 78},
    },
]

AGE_RANGES.update({
    "post_cardiac_surgery": (55, 75),
    "chf_monitoring":       (60, 80),
    "stroke_rehab":         (58, 82),
})


def generate_patients(n_patients: int = 20) -> list[dict]:
    """
    Generate n_patients synthetic ICU patients with realistic severity distribution:
      - 2 RED (critical)     — 10% of ward
      - 4 AMBER (warning)    — 20% of ward
      - 14 GREEN (stable)    — 70% of ward
    """
    PATIENTS_DIR.mkdir(parents=True, exist_ok=True)
    patients = []

    # Build the explicit assignment list
    red_scenarios   = [s for s in SCENARIOS if s["severity"] == "RED"]
    amber_scenarios = [s for s in SCENARIOS if s["severity"] == "AMBER"]
    green_scenarios = [s for s in SCENARIOS if s["severity"] == "GREEN"] + STABLE_EXTRAS

    assignment: list[dict] = []
    # 2 critical patients
    for i in range(2):
        assignment.append(red_scenarios[i % len(red_scenarios)])
    # 4 warning patients
    for i in range(4):
        assignment.append(amber_scenarios[i % len(amber_scenarios)])
    # Fill rest with stable
    remaining = n_patients - len(assignment)
    for i in range(remaining):
        assignment.append(green_scenarios[i % len(green_scenarios)])

    # Shuffle so beds aren't grouped by severity
    random.shuffle(assignment)

    for pid_idx, scenario in enumerate(assignment, start=1):
        sex = "M" if pid_idx % 2 == 1 else "F"
        age_min, age_max = AGE_RANGES.get(scenario["name"], (50, 75))
        age = random.randint(age_min, age_max)
        name = fake.name_male() if sex == "M" else fake.name_female()
        bed = BEDS[(pid_idx - 1) % len(BEDS)]
        admission_date = (datetime.utcnow() - timedelta(days=random.randint(1, 5))).isoformat()

        labs = []
        for l in scenario["labs"]:
            labs.append({
                "test": l["test"],
                "value": _add_noise(l["value"]),
                "unit": l["unit"],
                "flag": l["flag"],
                "timestamp": (datetime.utcnow() - timedelta(hours=random.randint(1, 6))).isoformat(),
            })

        patient = {
            "patient_id": f"P{pid_idx:03d}",
            "scenario": scenario["name"],
            "demographics": {
                "name": name, "age": age, "sex": sex,
                "bed": bed, "admission_date": admission_date,
                "weight_kg": round(random.uniform(55, 100), 1),
            },
            "conditions": scenario["conditions"],
            "medications": scenario["medications"],
            "lab_results": labs,
            "vitals_latest": {k: _add_noise(v, 0.04) for k, v in scenario["vitals"].items()},
            "vitals_timeseries": _generate_vitals_timeseries(scenario["vitals"], hours=24),
            "procedures": [
                {"name": "Central venous catheter", "timestamp": admission_date},
                {"name": "Urinary catheter", "timestamp": admission_date},
            ],
            "expected_severity": scenario["severity"],
        }

        note = _generate_note(patient, scenario)
        patient["raw_notes"] = [note]

        patient_dir = PATIENTS_DIR / f"P{pid_idx:03d}"
        patient_dir.mkdir(parents=True, exist_ok=True)
        (patient_dir / "patient_state.json").write_text(
            json.dumps(patient, indent=2), encoding="utf-8"
        )
        (patient_dir / "progress_note.txt").write_text(note, encoding="utf-8")

        patients.append(patient)
        logger.info(f"  ✅ Generated P{pid_idx:03d} — {scenario['name']} ({scenario['severity']})")

    logger.info(f"\n🏥 Generated {len(patients)} ICU patients in {PATIENTS_DIR}")
    return patients


def load_all_patients() -> list[dict]:
    """Load all patient_state.json files from data/patients/."""
    patients = []
    for d in sorted(PATIENTS_DIR.iterdir()):
        fp = d / "patient_state.json"
        if fp.exists():
            patients.append(json.loads(fp.read_text()))
    return patients
