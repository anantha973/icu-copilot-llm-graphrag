"""
Patients router — /api/patients and /api/patients/{patient_id}

GET /api/patients
    Returns a compact list of all patients: demographics, vitals_latest,
    severity badge, and sparkline timeseries for the overview grid.

GET /api/patients/{patient_id}
    Returns the full patient object for the detail panel.

GET /api/severity/{patient_id}
    Returns severity + triggered_rules + confidence for a single patient.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger("icu.patients")
router = APIRouter(tags=["Patients"])


@router.get("/patients")
async def list_patients(request: Request) -> list[dict[str, Any]]:
    """
    Ward overview payload — one entry per patient.
    Polled every 15 seconds by the frontend.
    """
    state = request.app.state.icu_state
    result = []

    for p in state.patients:
        pid = p["patient_id"]
        sev_data = state.severity_map.get(pid, {
            "severity": "GREEN",
            "triggered_rules": [],
            "confidence": 0.85,
        })
        demo = p.get("demographics", {})
        vitals = p.get("vitals_latest", {})

        # Build compact sparkline data — last 12h hourly readings for MAP, HR, SpO2
        ts = p.get("vitals_timeseries", [])
        sparkline: dict[str, list] = {"timestamps": [], "MAP": [], "HR": [], "SpO2": []}
        for point in ts[-12:]:
            sparkline["timestamps"].append(point.get("timestamp", "")[-8:-3])
            sparkline["MAP"].append(point.get("MAP"))
            sparkline["HR"].append(point.get("HR"))
            sparkline["SpO2"].append(point.get("SpO2"))

        # Graph stats
        G = state.graphs.get(pid)
        graph_stats = {
            "nodes": G.number_of_nodes() if G else 0,
            "edges": G.number_of_edges() if G else 0,
        }

        result.append({
            "patient_id": pid,
            "scenario": p.get("scenario", ""),
            "demographics": demo,
            "vitals_latest": vitals,
            "severity": sev_data["severity"],
            "triggered_rules": sev_data["triggered_rules"],
            "confidence": sev_data["confidence"],
            "sparkline": sparkline,
            "graph_stats": graph_stats,
            "conditions": p.get("conditions", []),
            "conditions_count": len(p.get("conditions", [])),
            "meds_count": len(p.get("medications", [])),
        })

    return result


# ── /api/patients/{patient_id} ────────────────────────────────────────────────

@router.get("/patients/{patient_id}")
async def get_patient(patient_id: str, request: Request) -> dict[str, Any]:
    """
    Full patient detail — called when a clinician opens a patient card.
    Returns everything needed to populate all 4 detail tabs.
    """
    state = request.app.state.icu_state
    patient = next((p for p in state.patients if p["patient_id"] == patient_id), None)
    if patient is None:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")

    sev_data = state.severity_map.get(patient_id, {
        "severity": "GREEN",
        "triggered_rules": [],
        "confidence": 0.85,
    })

    # Full timeseries for Plotly chart (all 24h)
    ts = patient.get("vitals_timeseries", [])
    timeseries_full = {
        "timestamps": [t.get("timestamp", "")[-8:-3] for t in ts],
        "MAP":  [t.get("MAP") for t in ts],
        "HR":   [t.get("HR") for t in ts],
        "SpO2": [t.get("SpO2") for t in ts],
        "RR":   [t.get("RR") for t in ts],
        "Temp": [t.get("Temp") for t in ts],
    }

    G = state.graphs.get(patient_id)
    primekg_links = sum(
        1 for _, d in G.nodes(data=True) if d.get("primekg_node_id")
    ) if G else 0

    return {
        "patient_id": patient_id,
        "scenario": patient.get("scenario", ""),
        "demographics": patient.get("demographics", {}),
        "conditions": patient.get("conditions", []),
        "medications": patient.get("medications", []),
        "lab_results": patient.get("lab_results", []),
        "vitals_latest": patient.get("vitals_latest", {}),
        "procedures": patient.get("procedures", []),
        "timeseries": timeseries_full,
        "severity": sev_data["severity"],
        "triggered_rules": sev_data["triggered_rules"],
        "confidence": sev_data["confidence"],
        "graph_stats": {
            "nodes": G.number_of_nodes() if G else 0,
            "edges": G.number_of_edges() if G else 0,
            "primekg_links": primekg_links,
        },
    }


# ── /api/severity/{patient_id} ────────────────────────────────────────────────

@router.get("/severity/{patient_id}")
async def get_severity(patient_id: str, request: Request) -> dict[str, Any]:
    """Severity data only — used for targeted badge refresh."""
    state = request.app.state.icu_state
    if patient_id not in state.severity_map:
        raise HTTPException(status_code=404, detail=f"Severity data not found for {patient_id}")
    return state.severity_map[patient_id]
