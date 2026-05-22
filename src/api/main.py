"""
ICU Clinical Copilot — FastAPI Backend
Layer 4: replaces Streamlit with a clean JSON API + WebSocket push.

Run:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
    # OR
    bash scripts/run_dashboard.sh
"""

from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# ── Path setup ────────────────────────────────────────────────────────────────
# Ensure src/ is importable regardless of cwd
SRC_DIR = Path(__file__).parents[1]
PROJ_DIR = Path(__file__).parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from api.routers import patients as patients_router
from api.routers import graph as graph_router
from api.routers import chat as chat_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("icu.main")

# ── App State ─────────────────────────────────────────────────────────────────

@dataclass
class AppState:
    """
    Single-instance application state — replaces st.cache_resource.
    All fields are populated during the lifespan startup hook.
    """
    patients: list[dict] = field(default_factory=list)
    graphs: dict = field(default_factory=dict)        # patient_id → nx.DiGraph
    severity_map: dict = field(default_factory=dict)  # patient_id → {severity, triggers}
    model_manager: Optional[object] = None
    kb_retriever: Optional[object] = None
    models_loaded: bool = False
    kb_loaded: bool = False
    ws_clients: list[WebSocket] = field(default_factory=list)

# Module-level singleton — routers import this directly
state = AppState()

# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all heavy resources at startup; nothing to clean up on shutdown."""
    logger.info("🏥 ICU Copilot starting…")

    # 1. Load patients
    try:
        from patient_simulation.patient_generator import load_all_patients
        state.patients = load_all_patients()
        logger.info(f"✅ Loaded {len(state.patients)} patients")
    except Exception as e:
        logger.warning(f"⚠️  Could not load patients: {e}. Run scripts/generate_patients.py first.")
        state.patients = []

    # 2. Build patient graphs
    try:
        from knowledge_graph.graph_builder import build_patient_graph
        state.graphs = {p["patient_id"]: build_patient_graph(p) for p in state.patients}
        logger.info(f"✅ Built {len(state.graphs)} patient graphs")
    except Exception as e:
        logger.warning(f"⚠️  Graph building failed: {e}")
        state.graphs = {}

    # 3. Compute severity (rule-based — fast, no LLM needed)
    try:
        from llm_reasoning.severity_classifier import rule_based_severity
        for p in state.patients:
            pid = p["patient_id"]
            sev, triggers = rule_based_severity(p)
            state.severity_map[pid] = {
                "severity": sev,
                "triggered_rules": triggers,
                "confidence": 0.90 if sev != "GREEN" else 0.85,
            }
        logger.info("✅ Severity map computed (rule-based)")
    except Exception as e:
        logger.warning(f"⚠️  Severity computation failed: {e}")

    # 4. Load LLM models — lazy: only if GGUF files present
    try:
        from llm_reasoning.model_loader import ModelManager
        mgr = ModelManager(n_ctx=2048, n_batch=512, verbose=False)
        loaded = mgr.load()
        state.model_manager = mgr
        state.models_loaded = loaded
        if loaded:
            logger.info("✅ LLM models loaded")
        else:
            logger.info("ℹ️  LLM models not found — running in rule-only mode")
    except Exception as e:
        logger.warning(f"⚠️  Model loader failed: {e}")
        state.model_manager = None
        state.models_loaded = False

    # 5. Load Global KB (Guidelines FAISS always; PrimeKG only if SKIP_PRIMEKG is not set)
    import os
    skip_primekg = os.environ.get("SKIP_PRIMEKG", "0") == "1"
    if skip_primekg:
        logger.info("⏭️  PrimeKG skipped (SKIP_PRIMEKG=1) — guidelines-only mode")

    try:
        from knowledge_base.guideline_indexer import index_guidelines
        from knowledge_base.retriever import GlobalKBRetriever
        import faiss
        import pickle

        faiss_idx, chunks, embedder = index_guidelines()

        primekg = None
        name_idx = {}
        primekg_faiss_idx = None
        primekg_faiss_map = None

        if not skip_primekg:
            from knowledge_base.primekg_loader import GRAPH_PKL, load_primekg, build_name_index, PRIMEKG_FAISS_INDEX, PRIMEKG_FAISS_MAP
            if GRAPH_PKL.exists():
                primekg = load_primekg()
                name_idx = build_name_index(primekg)
            if PRIMEKG_FAISS_INDEX.exists() and PRIMEKG_FAISS_MAP.exists():
                primekg_faiss_idx = faiss.read_index(str(PRIMEKG_FAISS_INDEX))
                with open(PRIMEKG_FAISS_MAP, "rb") as f:
                    primekg_faiss_map = pickle.load(f)

        state.kb_retriever = GlobalKBRetriever(
            primekg=primekg,
            name_index=name_idx,
            faiss_index=faiss_idx,
            guideline_chunks=chunks,
            embedder=embedder,
            primekg_faiss_index=primekg_faiss_idx,
            primekg_faiss_map=primekg_faiss_map,
        )
        state.kb_loaded = True
        mode = "guidelines only" if skip_primekg else "PrimeKG + FAISS guidelines"
        logger.info(f"✅ Global KB loaded ({mode})")
    except Exception as e:
        logger.warning(f"⚠️  Global KB not loaded: {e}. Run scripts/build_global_kb.py.")
        state.kb_retriever = None
        state.kb_loaded = False

    logger.info("🚀 ICU Copilot ready — http://localhost:8000")
    yield
    logger.info("👋 ICU Copilot shutting down")


# ── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="ICU Clinical Copilot API",
    description="GraphRAG-powered ICU patient monitoring backend",
    version="1.0.0",
    lifespan=lifespan,
)

# Attach state to app for router access
app.state.icu_state = state

# ── Static files (serves index.html) ─────────────────────────────────────────
# check_dir=False prevents StaticFiles from raising RuntimeError on an empty
# directory at startup — safe once index.html is dropped into static/.
STATIC_DIR = PROJ_DIR / "frontend"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR), check_dir=False), name="static")

# ── Include routers ───────────────────────────────────────────────────────────
app.include_router(patients_router.router, prefix="/api")
app.include_router(graph_router.router, prefix="/api")
app.include_router(chat_router.router, prefix="/api")


# ── Root: serve the dashboard SPA ────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"error": "index.html not found in frontend/"}


# ── Health endpoint ───────────────────────────────────────────────────────────
@app.get("/api/health")
async def health(request: Request):
    state = request.app.state.icu_state
    return {
        "status": "ok",
        "patients_count": len(state.patients),
        "graphs_count": len(state.graphs),
        "models_loaded": state.models_loaded,
        "kb_loaded": state.kb_loaded,
    }


# ── WebSocket: alert push channel ─────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    state.ws_clients.append(ws)
    logger.info(f"🔌 WS client connected ({len(state.ws_clients)} total)")
    try:
        # Keep-alive: echo pings and forward server-push alerts
        while True:
            try:
                data = await asyncio.wait_for(ws.receive_text(), timeout=30.0)
                if data == "ping":
                    await ws.send_text("pong")
            except asyncio.TimeoutError:
                # Send periodic keep-alive
                await ws.send_json({"type": "heartbeat"})
    except WebSocketDisconnect:
        pass
    finally:
        if ws in state.ws_clients:
            state.ws_clients.remove(ws)
        logger.info(f"🔌 WS client disconnected ({len(state.ws_clients)} total)")


async def broadcast_alert(patient_id: str, severity: str, message: str):
    """Call this from background tasks to push alerts to all connected browsers."""
    payload = {
        "type": "alert",
        "patient_id": patient_id,
        "severity": severity,
        "message": message,
    }
    dead = []
    for ws in state.ws_clients:
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in state.ws_clients:
            state.ws_clients.remove(ws)
