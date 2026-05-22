"""
Layer 2 — Patient Instance Graph Builder
Converts a patient dict (from patient_generator) into a per-patient NetworkX DiGraph.
Patient entity nodes are linked to PrimeKG nodes via fuzzy matching.
"""

from __future__ import annotations
import logging
from typing import Optional

import networkx as nx

logger = logging.getLogger(__name__)


def build_patient_graph(
    patient: dict,
    global_kg: Optional[nx.DiGraph] = None,
    name_index: Optional[dict] = None,
) -> nx.DiGraph:
    """
    Build a per-patient directed graph.

    Args:
        patient:    Patient dict from patient_generator.load_all_patients()
        global_kg:  PrimeKG DiGraph (optional — links patient nodes to global KB)
        name_index: name → node_index dict from primekg_loader.build_name_index()

    Returns:
        NetworkX DiGraph for this patient
    """
    from rapidfuzz import process, fuzz

    pid = patient["patient_id"]
    G = nx.DiGraph(patient_id=pid)

    # ── Patient root node ────────────────────────────────────────────────────
    demo = patient.get("demographics", {})
    G.add_node(
        pid,
        node_type="Patient",
        name=demo.get("name", "Unknown"),
        age=demo.get("age", 0),
        sex=demo.get("sex", "?"),
        bed=demo.get("bed", "?"),
        admission_date=demo.get("admission_date", ""),
        weight_kg=demo.get("weight_kg", 0),
    )

    def _fuzzy_link(entity_name: str) -> Optional[str]:
        """Return best-matching PrimeKG node_id for entity_name, or None."""
        if not (global_kg and name_index):
            return None
        query = entity_name.lower().strip()
        if query in name_index:
            return name_index[query][0]
        candidates = process.extract(
            query, name_index.keys(), scorer=fuzz.WRatio, limit=1, score_cutoff=72
        )
        if candidates:
            matched_name = candidates[0][0]
            nodes = name_index.get(matched_name, [])
            return nodes[0] if nodes else None
        return None

    # ── Condition nodes ───────────────────────────────────────────────────────
    for cond in patient.get("conditions", []):
        nid = f"{pid}_cond_{cond['name'].replace(' ', '_')}"
        primekg_id = _fuzzy_link(cond["name"])
        G.add_node(
            nid,
            node_type="Condition",
            name=cond["name"],
            icd_code=cond.get("icd_code", ""),
            status=cond.get("status", "active"),
            primekg_node_id=primekg_id or "",
        )
        G.add_edge(pid, nid, relation="diagnosed_with")
        if primekg_id and global_kg and primekg_id in global_kg:
            G.add_edge(nid, primekg_id, relation="maps_to_global")

    # ── Medication nodes + drug-drug interactions ─────────────────────────────
    med_nodes: list[tuple[str, str]] = []  # (node_id, med_name)
    for med in patient.get("medications", []):
        nid = f"{pid}_med_{med['name'].replace(' ', '_')}"
        primekg_id = _fuzzy_link(med["name"])
        G.add_node(
            nid,
            node_type="Medication",
            name=med["name"],
            dose=med.get("dose", ""),
            route=med.get("route", ""),
            indication=med.get("indication", ""),
            primekg_node_id=primekg_id or "",
        )
        G.add_edge(pid, nid, relation="prescribed")
        if primekg_id and global_kg and primekg_id in global_kg:
            G.add_edge(nid, primekg_id, relation="maps_to_global")
        med_nodes.append((nid, primekg_id or ""))

    # Pull drug-drug interactions from PrimeKG
    if global_kg and name_index:
        _add_drug_interactions(G, global_kg, med_nodes)

    # ── Lab result nodes ──────────────────────────────────────────────────────
    for lab in patient.get("lab_results", []):
        is_crit = lab.get("flag") in ("HIGH", "LOW", "CRITICAL")
        nid = f"{pid}_lab_{lab['test'].replace('/', '_').replace(' ', '_')}_{lab.get('timestamp', '')[-5:]}"
        G.add_node(
            nid,
            node_type="LabResult",
            test=lab["test"],
            value=lab["value"],
            unit=lab["unit"],
            flag=lab.get("flag", "NORMAL"),
            timestamp=lab.get("timestamp", ""),
            is_critical=is_crit,
        )
        G.add_edge(pid, nid, relation="has_result")
        # Link abnormal labs to related conditions
        _link_lab_to_condition(G, pid, nid, lab)

    # ── Vital nodes ───────────────────────────────────────────────────────────
    latest = patient.get("vitals_latest", {})
    CRITICAL_THRESHOLDS = {
        "MAP": lambda v: v < 65,
        "SpO2": lambda v: v < 90,
        "HR":  lambda v: v > 130,
    }
    for vital_type, value in latest.items():
        is_crit = CRITICAL_THRESHOLDS.get(vital_type, lambda v: False)(value)
        nid = f"{pid}_vital_{vital_type}"
        G.add_node(
            nid,
            node_type="Vital",
            vital_type=vital_type,
            value=value,
            is_critical=is_crit,
        )
        G.add_edge(pid, nid, relation="measured_at")

    # ── Procedure nodes ───────────────────────────────────────────────────────
    for proc in patient.get("procedures", []):
        nid = f"{pid}_proc_{proc['name'].replace(' ', '_')}"
        G.add_node(
            nid,
            node_type="Procedure",
            name=proc["name"],
            timestamp=proc.get("timestamp", ""),
        )
        G.add_edge(pid, nid, relation="underwent")

    logger.debug(
        f"  Built graph for {pid}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
    )
    return G


def _add_drug_interactions(
    G: nx.DiGraph,
    global_kg: nx.DiGraph,
    med_nodes: list[tuple[str, str]],
) -> None:
    """Add drug-drug interaction edges from PrimeKG."""
    for i, (nid_a, primekg_a) in enumerate(med_nodes):
        if not primekg_a or primekg_a not in global_kg:
            continue
        for nid_b, primekg_b in med_nodes[i + 1:]:
            if not primekg_b or primekg_b not in global_kg:
                continue
            # Check for drug_drug edge in either direction
            if global_kg.has_edge(primekg_a, primekg_b):
                data = global_kg[primekg_a][primekg_b]
                if "drug_drug" in data.get("relation", ""):
                    G.add_edge(nid_a, nid_b, relation="contraindicates",
                               source="PrimeKG")
            elif global_kg.has_edge(primekg_b, primekg_a):
                data = global_kg[primekg_b][primekg_a]
                if "drug_drug" in data.get("relation", ""):
                    G.add_edge(nid_a, nid_b, relation="contraindicates",
                               source="PrimeKG")


def _link_lab_to_condition(
    G: nx.DiGraph, pid: str, lab_nid: str, lab: dict
) -> None:
    """Add abnormal_for edges from lab nodes to relevant condition nodes."""
    CONDITION_LAB_MAP = {
        "Lactate":          ["Septic shock", "Septic_shock"],
        "Creatinine":       ["Acute kidney injury", "Acute_kidney_injury"],
        "Troponin I":       ["ST-elevation myocardial infarction", "Cardiogenic shock"],
        "BNP":              ["Cardiogenic shock"],
        "Procalcitonin":    ["Septic shock", "Community-acquired pneumonia"],
        "PaO2/FiO2":        ["Acute respiratory distress syndrome"],
        "WBC":              ["Septic shock", "Community-acquired pneumonia"],
        "HbA1c":            ["Type 2 diabetes mellitus"],
        "Glucose":          ["Type 2 diabetes mellitus"],
    }
    if lab.get("flag") not in ("HIGH", "LOW", "CRITICAL"):
        return
    conditions_to_link = CONDITION_LAB_MAP.get(lab["test"], [])
    for cond_name in conditions_to_link:
        cond_nid_1 = f"{pid}_cond_{cond_name}"
        cond_nid_2 = f"{pid}_cond_{cond_name.replace(' ', '_')}"
        for cond_nid in [cond_nid_1, cond_nid_2]:
            if G.has_node(cond_nid):
                G.add_edge(lab_nid, cond_nid, relation="abnormal_for")
                break
