"""
Pipeline A — Global KB Retriever
The single interface Layer 3 uses to query both PrimeKG and FAISS guidelines.
"""

from __future__ import annotations
import logging
from typing import Optional

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


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

    # ── PrimeKG retrieval ────────────────────────────────────────────────────

    def find_primekg_nodes(
        self, entity_name: str, top_k: int = 3
    ) -> list[str]:
        """
        Fuzzy-match entity_name to PrimeKG node indices.
        Uses rapidfuzz for speed.
        """
        from rapidfuzz import process, fuzz

        query = entity_name.lower().strip()

        # Exact match first
        if query in self.name_index:
            return self.name_index[query][:top_k]

        # Fuzzy match
        candidates = process.extract(
            query,
            self.name_index.keys(),
            scorer=fuzz.WRatio,
            limit=top_k,
            score_cutoff=70,
        )
        nodes: list[str] = []
        for name, score, _ in candidates:
            nodes.extend(self.name_index.get(name, []))
        return nodes[:top_k]

    def find_primekg_nodes_vector(self, query: str, top_k: int = 3) -> list[str]:
        """Use FAISS vector search to find PrimeKG nodes."""
        if not self.primekg_faiss_index or not self.primekg_faiss_map:
            return self.find_primekg_nodes(query, top_k=top_k)
            
        import numpy as np
        query_vec = self.embedder.encode([query]).astype(np.float32)
        distances, indices = self.primekg_faiss_index.search(query_vec, top_k)
        
        nodes = []
        for idx in indices[0]:
            if idx in self.primekg_faiss_map:
                nodes.append(self.primekg_faiss_map[idx])
        return nodes

    def get_primekg_context(
        self,
        conditions: list[str],
        drugs: list[str],
        radius: int = 2,
        max_nodes: int = 40,
    ) -> str:
        """
        Build a text representation of the PrimeKG subgraph relevant to
        the given conditions and drugs.
        """
        seed_nodes: list[str] = []
        for entity in conditions + drugs:
            seed_nodes.extend(self.find_primekg_nodes_vector(entity, top_k=2))

        if not seed_nodes:
            return "(No PrimeKG matches found for given conditions/drugs.)"

        seen: set[str] = set()
        for nid in seed_nodes:
            if nid not in self.primekg:
                continue
            try:
                ego = nx.ego_graph(self.primekg, nid, radius=radius, undirected=True)
                seen.update(ego.nodes())
            except Exception:
                pass
            if len(seen) >= max_nodes:
                break

        subgraph_nodes = list(seen)[:max_nodes]
        sub = self.primekg.subgraph(subgraph_nodes)

        lines: list[str] = ["=== PrimeKG Medical Knowledge ==="]
        for u, v, data in sub.edges(data=True):
            u_name = self.primekg.nodes[u].get("node_name", u)
            v_name = self.primekg.nodes[v].get("node_name", v)
            rel = data.get("display_relation", data.get("relation", "related_to"))
            lines.append(f"{u_name} --[{rel}]--> {v_name}")

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
                chunks.append(self.guideline_chunks[idx])

        if not chunks:
            return "(No relevant guideline chunks found.)"
        return "=== Clinical Guidelines ===\n" + "\n\n---\n".join(chunks)

    # ── Combined context ──────────────────────────────────────────────────────

    def get_full_context(
        self,
        conditions: list[str],
        drugs: list[str],
        query: str,
        guideline_top_k: int = 3,
    ) -> dict[str, str]:
        """
        Main API called by Layer 3 reasoning engine.

        Returns:
            {
              "graph_context": str,      # PrimeKG subgraph text
              "guideline_context": str,  # FAISS retrieved guideline chunks
            }
        """
        graph_ctx = self.get_primekg_context(conditions, drugs)
        guideline_ctx = self.get_guideline_context(query, top_k=guideline_top_k)
        return {
            "graph_context": graph_ctx,
            "guideline_context": guideline_ctx,
        }
