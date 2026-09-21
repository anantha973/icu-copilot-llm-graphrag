"""
Pipeline A — Global KB Retriever
The single interface Layer 3 uses to query both PrimeKG and FAISS guidelines.
Enhanced with Clinical Entity Normalization, Acronym Resolution, and
Relation-Directed Clinical Subgraph Extraction.
"""

from __future__ import annotations
import logging
from typing import Optional

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

# ── High-yield Clinical Acronym & Brand-Name Synonym Normalizer ───────────────
CLINICAL_SYNONYMS: dict[str, str] = {
    # Conditions & Acronyms
    "ckd": "chronic kidney disease",
    "esrd": "end-stage renal disease",
    "renal failure": "kidney failure",
    "acute renal failure": "acute kidney failure",
    "arf": "acute kidney failure",
    "aki": "acute kidney injury",
    "afib": "atrial fibrillation",
    "a-fib": "atrial fibrillation",
    "af": "atrial fibrillation",
    "mi": "myocardial infarction",
    "heart attack": "myocardial infarction",
    "stemi": "myocardial infarction",
    "nstemi": "myocardial infarction",
    "htn": "hypertension",
    "high blood pressure": "hypertension",
    "copd": "chronic obstructive pulmonary disease",
    "chf": "congestive heart failure",
    "heart failure": "heart failure",
    "dvt": "deep vein thrombosis",
    "pe": "pulmonary embolism",
    "stroke": "cerebrovascular accident",
    "cva": "cerebrovascular accident",
    "tia": "transient ischemic attack",
    "t2d": "type 2 diabetes mellitus",
    "t2dm": "type 2 diabetes mellitus",
    "diabetes": "diabetes mellitus",
    "sepsis": "sepsis",
    "ards": "acute respiratory distress syndrome",
    "gi bleed": "gastrointestinal hemorrhage",
    "uti": "urinary tract infection",
    "pneumonia": "pneumonia",
    "pna": "pneumonia",
    "cirrhosis": "cirrhosis",
    "asthma": "asthma",
    "wheezing": "asthma",
    "shortness of breath": "dyspnea",
    "low blood sugar": "hypoglycemia",
    "high blood sugar": "hyperglycemia",
    "chest pain": "angina pectoris",
    "angina": "angina pectoris",
    "irregular heartbeat": "cardiac arrhythmia",
    "swollen legs": "edema",
    "fluid overload": "edema",
    "kidney stones": "nephrolithiasis",
    "stomach ulcer": "peptic ulcer disease",
    "blood clot in lung": "pulmonary embolism",
    "blood clot": "thrombosis",

    # Brand Names -> Canonical Drug Generic Names
    "lasix": "furosemide",
    "plavix": "clopidogrel",
    "coumadin": "warfarin",
    "advil": "ibuprofen",
    "motrin": "ibuprofen",
    "tylenol": "acetaminophen",
    "paracetamol": "acetaminophen",
    "norvasc": "amlodipine",
    "glucophage": "metformin",
    "lopressor": "metoprolol",
    "toprol": "metoprolol",
    "cardizem": "diltiazem",
    "cipro": "ciprofloxacin",
    "cordarone": "amiodarone",
    "lanoxin": "digoxin",
    "lipitor": "atorvastatin",
    "crestor": "rosuvastatin",
    "zestril": "lisinopril",
    "prinivil": "lisinopril",
    "cozaar": "losartan",
    "diovan": "valsartan",
    "zantac": "ranitidine",
    "prilosec": "omeprazole",
    "nexium": "esomeprazole",
    "zofran": "ondansetron",
    "ativan": "lorazepam",
    "valium": "diazepam",
    "versed": "midazolam",
    "propofol": "propofol",
    "diprivan": "propofol",
    "levophed": "norepinephrine",
    "epinephrine": "epinephrine",
    "adrenaline": "epinephrine",
    "dobutrex": "dobutamine",
    "narcan": "naloxone",
    "viagra": "sildenafil",
    "revatio": "sildenafil",
    "nitro": "nitroglycerin",
    "aldactone": "spironolactone",
    "inderal": "propranolol",
    "jardiance": "empagliflozin",
    "lovenox": "enoxaparin",
}

# Clinically actionable relation filters
CLINICAL_EDGE_RELATIONS = {
    "contraindication",
    "indication",
    "off-label use",
    "drug_drug",
    "side effect",
    "drug_effect",
    "disease_phenotype_positive",
    "treats",
}

CLINICAL_DISPLAY_RELATIONS = {
    "contraindication",
    "indication",
    "off-label use",
    "synergistic interaction",
    "side effect",
    "phenotype present",
    "treats",
}

# High-acuity critical / lethal interaction and contraindication rules
CRITICAL_CONTRAINDICATION_RULES = [
    (
        {"nitroglycerin", "isosorbide", "nitrate", "nitroprusside"},
        {"sildenafil", "tadalafil", "vardenafil", "avanafil"},
        "Lethal refractory hypotension via combined cGMP-mediated systemic vasodilation",
    ),
    (
        {"heparin", "enoxaparin", "warfarin", "apixaban", "rivaroxaban", "dabigatran"},
        {"gastrointestinal hemorrhage", "active bleed", "active hemorrhage", "intracranial hemorrhage", "cerebral hemorrhage"},
        "Exacerbation of life-threatening active hemorrhage",
    ),
    (
        {"propranolol", "nadolol", "timolol", "carvedilol", "labetalol", "metoprolol", "atenolol", "bisoprolol"},
        {"asthma", "copd", "severe copd", "bronchospasm", "reactive airway disease", "cardiogenic shock", "severe bradycardia"},
        "Severe bronchospasm via beta-2 receptor blockade in reactive airway disease or hemodynamic collapse in cardiogenic shock",
    ),
    (
        {"metformin"},
        {"acute kidney injury", "severe renal failure", "lactic acidosis", "severe ckd"},
        "Fatal lactic acidosis in severe renal impairment",
    ),
    (
        {"potassium chloride", "spironolactone", "eplerenone"},
        {"hyperkalemia", "severe renal failure"},
        "Fatal cardiac arrhythmias from critical hyperkalemia",
    ),
]


class GlobalKBRetriever:
    """
    Wraps PrimeKG (NetworkX) + NIH guidelines (FAISS).
    Loaded once at application startup, shared across all patients.
    """

    def __init__(
        self,
        primekg: nx.DiGraph,
        name_index: dict[str, list[str]],
        faiss_index,
        guideline_chunks: list[str],
        embedder,
        primekg_faiss_index=None,
        primekg_faiss_map=None,
    ):
        self.primekg = primekg
        self.name_index = name_index
        self.faiss_index = faiss_index
        self.guideline_chunks = guideline_chunks
        self.embedder = embedder
        self.primekg_faiss_index = primekg_faiss_index
        self.primekg_faiss_map = primekg_faiss_map

    # ── Entity Normalization ──────────────────────────────────────────────────

    def normalize_entity(self, entity_name: str) -> str:
        """Resolve clinical acronyms and brand names to canonical ontology names."""
        clean = entity_name.lower().strip()
        return CLINICAL_SYNONYMS.get(clean, clean)

    def _rank_node_candidates(self, node_ids: list[str]) -> list[str]:
        """Prioritize drug, disease, and phenotype nodes over generic proteins/genes."""
        if not self.primekg:
            return node_ids
        priority_types = {"drug", "disease", "effect/phenotype"}
        return sorted(
            node_ids,
            key=lambda nid: 0 if self.primekg.nodes[nid].get("node_type") in priority_types else 1,
        )

    # ── PrimeKG retrieval ────────────────────────────────────────────────────

    def find_primekg_nodes(
        self, entity_name: str, top_k: int = 3
    ) -> list[str]:
        """
        Match entity_name to PrimeKG node indices using normalization + exact + fuzzy match.
        Prioritizes clinical node types (drug, disease) over proteins.
        """
        from rapidfuzz import process, fuzz

        query = self.normalize_entity(entity_name)

        # 1. Exact match in name index
        if query in self.name_index:
            return self._rank_node_candidates(self.name_index[query])[:top_k]

        # 2. Fuzzy match
        candidates = process.extract(
            query,
            self.name_index.keys(),
            scorer=fuzz.WRatio,
            limit=top_k * 2,
            score_cutoff=70,
        )
        nodes: list[str] = []
        for name, score, _ in candidates:
            nodes.extend(self.name_index.get(name, []))

        return self._rank_node_candidates(nodes)[:top_k]

    def find_primekg_nodes_vector(self, query: str, top_k: int = 3) -> list[str]:
        """Use FAISS vector search with query normalization to find PrimeKG nodes."""
        normalized_query = self.normalize_entity(query)

        # If exact match exists after normalization, prioritize it
        if normalized_query in self.name_index:
            return self._rank_node_candidates(self.name_index[normalized_query])[:top_k]

        if not self.primekg_faiss_index or not self.primekg_faiss_map:
            return self.find_primekg_nodes(normalized_query, top_k=top_k)

        query_vec = self.embedder.encode([normalized_query]).astype(np.float32)
        distances, indices = self.primekg_faiss_index.search(query_vec, top_k * 2)

        nodes = []
        for idx in indices[0]:
            if idx in self.primekg_faiss_map:
                nid = self.primekg_faiss_map[idx]
                if nid in self.primekg:
                    nodes.append(nid)

        ranked = self._rank_node_candidates(nodes)
        return ranked[:top_k] if ranked else self.find_primekg_nodes(normalized_query, top_k=top_k)

    def get_primekg_context(
        self,
        conditions: list[str],
        drugs: list[str],
        radius: int = 1,
        max_nodes: int = 40,
    ) -> str:
        """
        Build a high-precision, clinically actionable text representation of the
        PrimeKG subgraph relevant to the patient's conditions and medications.

        Prioritizes:
        1. Direct Drug-Drug interactions (synergistic/antagonistic)
        2. Direct Drug-Disease contraindications and indications
        3. 1-hop adverse effects and phenotypes
        """
        if not self.primekg:
            return "(PrimeKG knowledge graph not loaded.)"

        # Resolve seed nodes
        seed_conditions: dict[str, str] = {}  # nid -> label
        seed_drugs: dict[str, str] = {}       # nid -> label

        for cond in conditions:
            matched = self.find_primekg_nodes_vector(cond, top_k=1)
            if matched and matched[0] in self.primekg:
                seed_conditions[matched[0]] = cond

        for drug in drugs:
            matched = self.find_primekg_nodes_vector(drug, top_k=1)
            if matched and matched[0] in self.primekg:
                seed_drugs[matched[0]] = drug

        all_seeds = {**seed_conditions, **seed_drugs}
        if not all_seeds:
            return "(No relevant PrimeKG nodes identified for current conditions/medications.)"

        direct_alerts: list[str] = []
        contraindications: list[str] = []
        indications: list[str] = []
        side_effects: list[str] = []

        # 1. Direct Seed-to-Seed Interactions (Highest Priority)
        for u in all_seeds:
            for v in all_seeds:
                if u != v and self.primekg.has_edge(u, v):
                    edata = self.primekg.edges[u, v]
                    rel = edata.get("display_relation", edata.get("relation", "interacts with"))
                    u_name = self.primekg.nodes[u].get("node_name", u)
                    v_name = self.primekg.nodes[v].get("node_name", v)
                    direct_alerts.append(f"• [DIRECT ALERT] {u_name} --[{rel}]--> {v_name}")

        # 2. Direct 1-hop clinical relations from/to each seed node
        for nid in all_seeds:
            node_name = self.primekg.nodes[nid].get("node_name", nid)
            # Outgoing edges
            for _, v, edata in self.primekg.edges(nid, data=True):
                rel = edata.get("display_relation", edata.get("relation", ""))
                v_name = self.primekg.nodes[v].get("node_name", v)
                v_type = self.primekg.nodes[v].get("node_type", "")
                if rel in ("contraindication",) or edata.get("relation") == "contraindication":
                    contraindications.append(f"• {node_name} CONTRAINDICATED with {v_name} ({v_type})")
                elif rel in ("indication", "treats", "off-label use") or edata.get("relation") in ("indication", "off-label use"):
                    indications.append(f"• {node_name} [{rel.upper()}] for {v_name} ({v_type})")
                elif rel in ("side effect", "drug_effect") or edata.get("relation") == "drug_effect":
                    side_effects.append(f"• {node_name} risk: {v_name}")

            # Incoming edges
            for u, _, edata in self.primekg.in_edges(nid, data=True):
                rel = edata.get("display_relation", edata.get("relation", ""))
                u_name = self.primekg.nodes[u].get("node_name", u)
                u_type = self.primekg.nodes[u].get("node_type", "")
                if rel in ("contraindication",) or edata.get("relation") == "contraindication":
                    contraindications.append(f"• {u_name} ({u_type}) CONTRAINDICATED with {node_name}")
                elif rel in ("indication", "treats", "off-label use") or edata.get("relation") in ("indication", "off-label use"):
                    indications.append(f"• {u_name} ({u_type}) [{rel.upper()}] for {node_name}")
                elif rel in ("side effect", "drug_effect") or edata.get("relation") == "drug_effect":
                    side_effects.append(f"• {node_name} risk: {u_name}")

        # Deduplicate while preserving insertion order
        direct_alerts = list(dict.fromkeys(direct_alerts))
        contraindications = list(dict.fromkeys(contraindications))
        side_effects = list(dict.fromkeys(side_effects))
        indications = list(dict.fromkeys(indications))

        lines: list[str] = ["=== PrimeKG Medical Knowledge (Verified Clinical Subgraph) ==="]

        if direct_alerts:
            lines.append("\n[Direct Patient Drug/Disease Interactions]")
            lines.extend(direct_alerts[:12])

        if contraindications:
            lines.append("\n[Contraindications]")
            lines.extend(contraindications[:12])

        if side_effects:
            lines.append("\n[Known Adverse Effects / Clinical Risks]")
            lines.extend(side_effects[:10])

        if indications:
            lines.append("\n[Approved Indications]")
            lines.extend(indications[:8])

        # If strict clinical relations were empty, fallback to 1-hop connected nodes
        if len(lines) <= 1:
            fallback_edges = []
            for nid in all_seeds:
                for _, v, edata in list(self.primekg.edges(nid, data=True))[:5]:
                    u_n = self.primekg.nodes[nid].get("node_name", nid)
                    v_n = self.primekg.nodes[v].get("node_name", v)
                    rel = edata.get("display_relation", edata.get("relation", "related to"))
                    fallback_edges.append(f"• {u_n} --[{rel}]--> {v_n}")
            lines.extend(fallback_edges[:max_nodes])

        return "\n".join(lines) if len(lines) > 1 else "(PrimeKG subgraph empty.)"

    # ── FAISS guideline retrieval ─────────────────────────────────────────────

    def get_guideline_context(self, query: str, top_k: int = 3) -> str:
        """
        Retrieve top-k guideline chunks relevant to the query via FAISS.
        """
        query_vec = self.embedder.encode([query]).astype(np.float32)
        distances, indices = self.faiss_index.search(query_vec, top_k)

        chunks: list[str] = []
        for idx in indices[0]:
            if 0 <= idx < len(self.guideline_chunks):
                chunk = self.guideline_chunks[idx].strip()
                if len(chunk) > 40 and not chunk.startswith("@"):
                    chunks.append(chunk[:600])

        if not chunks:
            return "(No relevant guideline chunks found.)"
        return "=== Clinical Guidelines ===\n" + "\n\n---\n".join(chunks[:2])

    # ── Safety Interlock Engine ───────────────────────────────────────────────

    def evaluate_safety_interlock(
        self,
        conditions: list[str],
        drugs: list[str],
        query: str,
        clinical_context: str,
        candidate_drugs: list[str] | None = None,
    ) -> tuple[str, str]:
        """
        Deterministic clinical safety interlock based on knowledge graph relations
        and high-acuity pharmacology rules.

        Returns:
            (verdict, rationale)
            verdict: "CONTRAINDICATED" | "CAUTION / MONITORING REQUIRED" | "PERMITTED / INDICATED"
        """
        import re

        # Extract or prioritize candidate drugs the clinician is asking about
        cand_list: list[str] = list(candidate_drugs) if candidate_drugs else []
        if not cand_list:
            # Detect candidate drug names mentioned in query
            for term, nid_list in self.name_index.items():
                if len(term) >= 4 and re.search(r"\b" + re.escape(term) + r"\b", query, re.IGNORECASE):
                    if self.primekg and any(self.primekg.nodes[nid].get("node_type") == "drug" for nid in nid_list if nid in self.primekg):
                        cand_list.append(term)

        # Existing background medications on the patient's chart
        cand_lower = {c.lower() for c in cand_list}
        existing_meds = [d for d in drugs if d.lower() not in cand_lower]

        all_entities = {self.normalize_entity(c).lower() for c in conditions} | {
            self.normalize_entity(d).lower() for d in drugs
        }
        query_norm = query.lower()

        # 1. Rule-based severe lethal interactions (Critical Contraindications)
        for group_a, group_b, reason in CRITICAL_CONTRAINDICATION_RULES:
            match_a = any(any(k in item for k in group_a) for item in all_entities) or any(
                k in query_norm for k in group_a
            )
            match_b = any(any(k in item for k in group_b) for item in all_entities) or any(
                k in query_norm for k in group_b
            )
            if match_a and match_b:
                return "CONTRAINDICATED", f"Critical Interaction: {reason}"

        if not self.primekg:
            return "PERMITTED / INDICATED", "No knowledge graph loaded for verification"

        # Resolve node seeds
        seed_conds: list[str] = []
        for c in conditions:
            m = self.find_primekg_nodes_vector(c, top_k=1)
            if m and m[0] in self.primekg:
                seed_conds.append(m[0])

        # If evaluating specific candidate drug(s) inquired by the clinician:
        if cand_list:
            seed_cand_drugs: list[str] = []
            for d in cand_list:
                m = self.find_primekg_nodes_vector(d, top_k=1)
                if m and m[0] in self.primekg:
                    seed_cand_drugs.append(m[0])

            seed_existing_meds: list[str] = []
            for d in existing_meds:
                m = self.find_primekg_nodes_vector(d, top_k=1)
                if m and m[0] in self.primekg:
                    seed_existing_meds.append(m[0])

            # 2a. Candidate drug contraindications with patient conditions
            for d_nid in seed_cand_drugs:
                d_name = self.primekg.nodes[d_nid].get("node_name", d_nid)
                for c_nid in seed_conds:
                    c_name = self.primekg.nodes[c_nid].get("node_name", c_nid)
                    for u, v in [(d_nid, c_nid), (c_nid, d_nid)]:
                        if self.primekg.has_edge(u, v) and self.primekg.edges[u, v].get("relation") == "contraindication":
                            return "CONTRAINDICATED", f"Contraindication: {c_name} contraindicated with {d_name}"

            # 2b. Candidate drug contraindications with existing medications
            for d_nid in seed_cand_drugs:
                d_name = self.primekg.nodes[d_nid].get("node_name", d_nid)
                for m_nid in seed_existing_meds:
                    m_name = self.primekg.nodes[m_nid].get("node_name", m_nid)
                    for u, v in [(d_nid, m_nid), (m_nid, d_nid)]:
                        if self.primekg.has_edge(u, v) and self.primekg.edges[u, v].get("relation") == "contraindication":
                            return "CONTRAINDICATED", f"Direct Contraindication: {d_name} with {m_name}"

            # 3. Approved Indications (GREEN FLAG)
            for d_nid in seed_cand_drugs:
                d_name = self.primekg.nodes[d_nid].get("node_name", d_nid)
                for c_nid in seed_conds:
                    c_name = self.primekg.nodes[c_nid].get("node_name", c_nid)
                    for u, v in [(d_nid, c_nid), (c_nid, d_nid)]:
                        if self.primekg.has_edge(u, v) and self.primekg.edges[u, v].get("relation") in ("indication", "treats", "off-label use"):
                            return "PERMITTED / INDICATED", f"Approved Indication: {d_name} for {c_name}"

            # 4. Moderate Drug-Drug Interactions with Existing Regimen (YELLOW FLAG / CAUTION)
            for d_nid in seed_cand_drugs:
                d_name = self.primekg.nodes[d_nid].get("node_name", d_nid)
                for m_nid in seed_existing_meds:
                    m_name = self.primekg.nodes[m_nid].get("node_name", m_nid)
                    for u, v in [(d_nid, m_nid), (m_nid, d_nid)]:
                        if self.primekg.has_edge(u, v):
                            edata = self.primekg.edges[u, v]
                            rel = edata.get("display_relation", edata.get("relation", ""))
                            if rel in ("synergistic interaction", "antagonistic interaction"):
                                return "CAUTION / MONITORING REQUIRED", f"Direct Drug Interaction: {d_name} --[{rel}]--> {m_name}"

            # 5. Safe / Permitted without interactions (GREEN FLAG)
            target_name = cand_list[0].capitalize()
            return "PERMITTED / INDICATED", f"No contraindications or adverse interactions identified for {target_name}"

        # If no specific candidate drug (general chart interaction review):
        seed_drugs: list[str] = []
        for d in drugs:
            m = self.find_primekg_nodes_vector(d, top_k=1)
            if m and m[0] in self.primekg:
                seed_drugs.append(m[0])

        all_seeds = seed_conds + seed_drugs

        # Check existing meds against each other
        for u in all_seeds:
            for v in all_seeds:
                if u != v and self.primekg.has_edge(u, v):
                    edata = self.primekg.edges[u, v]
                    rel = edata.get("display_relation", edata.get("relation", ""))
                    u_name = self.primekg.nodes[u].get("node_name", u)
                    v_name = self.primekg.nodes[v].get("node_name", v)
                    if rel in ("contraindication",):
                        return "CONTRAINDICATED", f"Direct Contraindication: {u_name} with {v_name}"
                    if rel in ("synergistic interaction", "antagonistic interaction"):
                        return "CAUTION / MONITORING REQUIRED", f"Direct Drug Interaction: {u_name} --[{rel}]--> {v_name}"

        return "PERMITTED / INDICATED", "No contraindications identified in clinical knowledge base"

    # ── Combined context ──────────────────────────────────────────────────────

    def get_full_context(
        self,
        conditions: list[str],
        drugs: list[str],
        query: str,
        guideline_top_k: int = 3,
        candidate_drugs: list[str] | None = None,
    ) -> dict[str, str]:
        """
        Main API called by Layer 3 reasoning engine.

        Returns:
            {
              "graph_context": str,      # PrimeKG subgraph text
              "guideline_context": str,  # FAISS retrieved guideline chunks
              "safety_verdict": str,     # Deterministic safety classification
              "safety_rationale": str,   # Deterministic safety reason
            }
        """
        graph_ctx = self.get_primekg_context(conditions, drugs)
        guideline_ctx = self.get_guideline_context(query, top_k=guideline_top_k)
        combined_text = graph_ctx + "\n\n" + guideline_ctx
        safety_verdict, safety_rationale = self.evaluate_safety_interlock(
            conditions, drugs, query, combined_text, candidate_drugs=candidate_drugs
        )
        return {
            "graph_context": graph_ctx,
            "guideline_context": guideline_ctx,
            "safety_verdict": safety_verdict,
            "safety_rationale": safety_rationale,
        }

