"""
Layer 2 — Subgraph Retriever
Extracts a human-readable context string from a patient instance graph.
This string is injected into Layer 3 LLM prompts.
"""

from __future__ import annotations
import logging
import networkx as nx

logger = logging.getLogger(__name__)

# Node types that carry clinical signal (exclude cross-graph links for text)
CLINICAL_TYPES = {"Condition", "Medication", "LabResult", "Vital", "Procedure"}


def get_patient_subgraph_text(
    G: nx.DiGraph,
    patient_id: str,
    radius: int = 2,
    max_nodes: int = 40,
    prioritize_critical: bool = True,
) -> str:
    """
    Extract the patient's clinical context as structured text for LLM injection.

    Args:
        G:                  Per-patient NetworkX DiGraph
        patient_id:         Root patient node ID (e.g. "P001")
        radius:             Hop distance from patient root
        max_nodes:          Cap on nodes included
        prioritize_critical: Put critical vitals and abnormal labs first

    Returns:
        Structured text representation of the patient subgraph
    """
    if patient_id not in G:
        logger.warning(f"Patient {patient_id} not found in graph.")
        return f"(No graph data found for {patient_id})"

    # Get all nodes within radius hops
    try:
        ego = nx.ego_graph(G, patient_id, radius=radius, undirected=False)
        subgraph_nodes = list(ego.nodes())
    except Exception as e:
        logger.warning(f"Ego graph failed for {patient_id}: {e}")
        subgraph_nodes = list(G.nodes())

    subgraph_nodes = subgraph_nodes[:max_nodes]

    lines: list[str] = []

    # Patient header
    root = G.nodes.get(patient_id, {})
    lines.append(
        f"=== Patient {patient_id} — {root.get('name', 'Unknown')}, "
        f"{root.get('age', '?')}y {root.get('sex', '?')}, "
        f"Bed {root.get('bed', '?')}, "
        f"Admitted: {root.get('admission_date', 'unknown')[:10]} ==="
    )

    # Collect by type
    conditions, medications, labs, vitals, procedures = [], [], [], [], []

    for nid in subgraph_nodes:
        attrs = G.nodes[nid]
        ntype = attrs.get("node_type", "")
        if ntype == "Condition":
            conditions.append(attrs)
        elif ntype == "Medication":
            medications.append(attrs)
        elif ntype == "LabResult":
            labs.append(attrs)
        elif ntype == "Vital":
            vitals.append(attrs)
        elif ntype == "Procedure":
            procedures.append(attrs)

    # Sort critical first
    if prioritize_critical:
        labs = sorted(labs, key=lambda x: 0 if x.get("flag") in ("HIGH", "LOW", "CRITICAL") else 1)
        vitals = sorted(vitals, key=lambda x: 0 if x.get("is_critical") else 1)

    # Conditions
    if conditions:
        lines.append("\nDIAGNOSES:")
        for c in conditions:
            status = c.get("status", "active")
            lines.append(f"  • {c.get('name', '?')} [{status.upper()}]"
                         + (f" (ICD: {c['icd_code']})" if c.get("icd_code") else ""))

    # Medications
    if medications:
        lines.append("\nMEDICATIONS:")
        for m in medications:
            lines.append(f"  • {m.get('name', '?')} {m.get('dose', '')} {m.get('route', '')}"
                         + (f" — for {m['indication']}" if m.get("indication") else ""))

    # Labs
    if labs:
        lines.append("\nLABORATORY RESULTS:")
        for l in labs:
            flag = l.get("flag", "NORMAL")
            marker = " ⚠️" if flag in ("HIGH", "LOW", "CRITICAL") else ""
            lines.append(f"  • {l.get('test', '?')}: {l.get('value', '?')} {l.get('unit', '')} [{flag}]{marker}")

    # Vitals
    if vitals:
        lines.append("\nVITAL SIGNS:")
        for v in vitals:
            crit = " ⚠️ CRITICAL" if v.get("is_critical") else ""
            lines.append(f"  • {v.get('vital_type', '?')}: {v.get('value', '?')}{crit}")

    # Procedures
    if procedures:
        lines.append("\nPROCEDURES:")
        for p in procedures:
            lines.append(f"  • {p.get('name', '?')}")

    # Drug interactions
    interactions = _get_drug_interactions(G, subgraph_nodes)
    if interactions:
        lines.append("\n⚠️  DRUG INTERACTION ALERTS:")
        for interaction in interactions:
            lines.append(f"  • {interaction}")

    return "\n".join(lines)


def _get_drug_interactions(G: nx.DiGraph, subgraph_nodes: list[str]) -> list[str]:
    """Extract all contraindicates edges within the subgraph."""
    interactions: list[str] = []
    for nid in subgraph_nodes:
        attrs = G.nodes[nid]
        if attrs.get("node_type") != "Medication":
            continue
        for neighbor in G.successors(nid):
            edge_data = G.edges[nid, neighbor]
            if edge_data.get("relation") == "contraindicates":
                target = G.nodes.get(neighbor, {})
                target_name = target.get("name", neighbor)
                interactions.append(
                    f"{attrs.get('name', nid)} ↔ {target_name} (interaction via PrimeKG)"
                )
    return interactions


def get_critical_summary(G: nx.DiGraph, patient_id: str) -> dict:
    """
    Fast summary of critical signals for severity classification.
    Returns dict with critical vitals, abnormal labs, drug alerts.
    """
    result = {
        "critical_vitals": [],
        "abnormal_labs": [],
        "drug_interactions": [],
        "active_conditions": [],
    }

    for nid, attrs in G.nodes(data=True):
        if not nid.startswith(patient_id):
            continue
        ntype = attrs.get("node_type", "")
        if ntype == "Vital" and attrs.get("is_critical"):
            result["critical_vitals"].append({
                "type": attrs["vital_type"],
                "value": attrs["value"],
            })
        elif ntype == "LabResult" and attrs.get("flag") in ("HIGH", "LOW", "CRITICAL"):
            result["abnormal_labs"].append({
                "test": attrs["test"],
                "value": attrs["value"],
                "unit": attrs["unit"],
                "flag": attrs["flag"],
            })
        elif ntype == "Condition" and attrs.get("status") == "active":
            result["active_conditions"].append(attrs["name"])

    # Drug interactions
    for nid in G.nodes():
        if G.nodes[nid].get("node_type") == "Medication":
            for succ in G.successors(nid):
                if G.edges[nid, succ].get("relation") == "contraindicates":
                    a = G.nodes[nid].get("name", nid)
                    b = G.nodes[succ].get("name", succ)
                    result["drug_interactions"].append(f"{a} ↔ {b}")

    return result
