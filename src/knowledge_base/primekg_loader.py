"""
Pipeline A — PrimeKG Loader
Loads PrimeKG CSV files into a NetworkX DiGraph, serializes to pickle.

Usage:
    python scripts/build_global_kb.py
    
Or import:
    from src.knowledge_base.primekg_loader import load_primekg
"""

from __future__ import annotations
import os
import pickle
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import networkx as nx
from tqdm import tqdm

logger = logging.getLogger(__name__)

PRIMEKG_DIR = Path(__file__).parents[2] / "data" / "primekg"
NODES_CSV = PRIMEKG_DIR / "nodes.csv"
EDGES_CSV  = PRIMEKG_DIR / "kg.csv"
GRAPH_PKL  = PRIMEKG_DIR / "primekg_graph.pkl"
PRIMEKG_FAISS_INDEX = PRIMEKG_DIR / "primekg_faiss.index"
PRIMEKG_FAISS_MAP = PRIMEKG_DIR / "primekg_faiss_map.pkl"


def download_primekg(force: bool = False) -> None:
    """
    Download PrimeKG CSVs from Harvard Dataverse if not present.
    Files are ~400MB total.
    """
    import requests

    PRIMEKG_DIR.mkdir(parents=True, exist_ok=True)

    urls = {
        "kg.csv":    "https://dataverse.harvard.edu/api/access/datafile/6180620",
        "nodes.csv": "https://dataverse.harvard.edu/api/access/datafile/6180617",
    }

    for filename, url in urls.items():
        dest = PRIMEKG_DIR / filename
        if dest.exists() and not force:
            logger.info(f"  ✅ {filename} already present, skipping download.")
            continue
        logger.info(f"  ⬇️  Downloading {filename}…")
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(dest, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc=filename
            ) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        logger.info(f"  ✅ {filename} downloaded.")


def build_primekg_graph(save: bool = True) -> nx.DiGraph:
    """
    Build a NetworkX DiGraph from PrimeKG CSVs.

    Nodes: one per unique entity, keyed by node_index with attributes:
        - node_id, node_type, node_name, node_source

    Edges: directed, keyed by (x_index, y_index) with relation type.

    Returns the graph (and optionally pickles it).
    """
    if not NODES_CSV.exists() or not EDGES_CSV.exists():
        raise FileNotFoundError(
            f"PrimeKG CSVs not found in {PRIMEKG_DIR}. "
            "Run download_primekg() first or place files manually."
        )

    logger.info("📖 Loading PrimeKG nodes…")
    # PrimeKG nodes.csv from Harvard Dataverse often uses tab separators.
    try:
        nodes_df = pd.read_csv(NODES_CSV, sep='\t', low_memory=False)
        if nodes_df.shape[1] < 2:
            raise ValueError("Possible wrong separator")
    except Exception:
        nodes_df = pd.read_csv(NODES_CSV, low_memory=False, on_bad_lines='skip')
    logger.info(f"   {len(nodes_df):,} nodes loaded.")

    logger.info("📖 Loading PrimeKG edges…")
    edges_df = pd.read_csv(EDGES_CSV, low_memory=False, on_bad_lines='skip')
    logger.info(f"   {len(edges_df):,} edges loaded.")

    G = nx.DiGraph()

    logger.info("🔨 Adding nodes…")
    for _, row in tqdm(nodes_df.iterrows(), total=len(nodes_df), desc="Nodes"):
        G.add_node(
            str(row["node_index"]),
            node_id=str(row.get("node_id", "")),
            node_type=str(row.get("node_type", "")),
            node_name=str(row.get("node_name", "")),
            node_source=str(row.get("node_source", "")),
        )

    logger.info("🔨 Adding edges…")
    for _, row in tqdm(edges_df.iterrows(), total=len(edges_df), desc="Edges"):
        G.add_edge(
            str(row["x_index"]),
            str(row["y_index"]),
            relation=str(row.get("relation", "")),
            display_relation=str(row.get("display_relation", "")),
            x_type=str(row.get("x_type", "")),
            y_type=str(row.get("y_type", "")),
        )

    logger.info(f"✅ PrimeKG graph built: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges.")

    if save:
        logger.info(f"💾 Serializing graph to {GRAPH_PKL}…")
        PRIMEKG_DIR.mkdir(parents=True, exist_ok=True)
        with open(GRAPH_PKL, "wb") as f:
            pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("✅ Graph saved.")

    return G


def load_primekg(rebuild: bool = False) -> nx.DiGraph:
    """
    Load the PrimeKG graph from pickle (fast) or rebuild from CSVs.

    Args:
        rebuild: Force rebuild from CSVs even if pickle exists.

    Returns:
        NetworkX DiGraph.
    """
    if GRAPH_PKL.exists() and not rebuild:
        logger.info(f"⚡ Loading PrimeKG from cache: {GRAPH_PKL}")
        with open(GRAPH_PKL, "rb") as f:
            G = pickle.load(f)
        logger.info(f"✅ PrimeKG loaded: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges.")
        return G

    logger.info("🔨 Rebuilding PrimeKG graph from CSVs…")
    return build_primekg_graph(save=True)


# ── Name → node_index lookup helpers ────────────────────────────────────────

def build_name_index(G: nx.DiGraph) -> dict[str, list[str]]:
    """
    Build a lowercase name → list[node_index] lookup dict.
    Used by fuzzy matching in primekg_linker.py.
    """
    name_index: dict[str, list[str]] = {}
    for node_id, attrs in G.nodes(data=True):
        name = attrs.get("node_name", "").lower().strip()
        if name:
            name_index.setdefault(name, []).append(node_id)
    return name_index


def get_subgraph_around(
    G: nx.DiGraph,
    node_ids: list[str],
    radius: int = 2,
    max_nodes: int = 50,
) -> nx.DiGraph:
    """
    Return a subgraph of G within `radius` hops of the given node_ids.
    Capped at max_nodes to keep context small.
    """
    seen: set[str] = set()
    for nid in node_ids:
        if nid in G:
            ego = nx.ego_graph(G, nid, radius=radius, undirected=True)
            seen.update(ego.nodes())
            if len(seen) >= max_nodes:
                break
    subgraph_nodes = list(seen)[:max_nodes]
    return G.subgraph(subgraph_nodes).copy()

def build_primekg_faiss(G: nx.DiGraph, embedder, force_rebuild: bool = False):
    """Embed all PrimeKG node names into a FAISS index for semantic search."""
    import faiss
    import numpy as np

    if PRIMEKG_FAISS_INDEX.exists() and PRIMEKG_FAISS_MAP.exists() and not force_rebuild:
        logger.info("⚡ Loading PrimeKG FAISS index from cache…")
        index = faiss.read_index(str(PRIMEKG_FAISS_INDEX))
        with open(PRIMEKG_FAISS_MAP, "rb") as f:
            idx_to_node = pickle.load(f)
        return index, idx_to_node

    logger.info("🔨 Building PrimeKG FAISS index from scratch…")
    
    node_indices = []
    texts_to_embed = []
    
    for node_idx, attrs in G.nodes(data=True):
        name = attrs.get("node_name", "")
        if name:
            texts_to_embed.append(name.lower())
            node_indices.append(node_idx)
            
    logger.info(f"⏳ Embedding {len(texts_to_embed)} PrimeKG nodes. This will take a few minutes…")
    embeddings = embedder.encode(texts_to_embed)
    
    logger.info("⏳ Building FAISS index…")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    idx_to_node = {i: node_idx for i, node_idx in enumerate(node_indices)}
    
    logger.info(f"💾 Saving PrimeKG FAISS index to {PRIMEKG_FAISS_INDEX}…")
    faiss.write_index(index, str(PRIMEKG_FAISS_INDEX))
    with open(PRIMEKG_FAISS_MAP, "wb") as f:
        pickle.dump(idx_to_node, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    logger.info("✅ PrimeKG FAISS index ready.")
    return index, idx_to_node
