"""
ICU Clinical Copilot Dashboard
Layer 4 — Streamlit Application (Complete Redesign based on code.html)

Run: streamlit run src/dashboard/app2.py
"""

from __future__ import annotations
import sys
import json
import logging
import time
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import networkx as nx

sys.path.insert(0, str(Path(__file__).parents[1])) # Add src to path

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ICU Cockpit - Knowledge Graph",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap');

/* Global Styles */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #031427 !important;
    color: #d3e4fe !important;
}

.stApp {
    background-color: #031427;
}

/* Hide Default Streamlit UI Elements */
[data-testid="stHeader"] {
    display: none !important;
}
[data-testid="baseButton-headerNoPadding"] {
    display: none !important;
}

/* Reverse Flex to move sidebar to the right */
[data-testid="stAppViewContainer"] {
    flex-direction: row-reverse;
}

/* Style Right Sidebar (Copilot) */
[data-testid="stSidebar"] {
    background-color: rgba(15, 23, 42, 0.9) !important;
    backdrop-filter: blur(12px) !important;
    border-left: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-right: none !important;
    min-width: 340px !important;
    max-width: 340px !important;
}

/* Main Content Adjustments */
[data-testid="stAppViewBlockContainer"] {
    padding-top: 80px !important; /* Make room for fixed top nav */
    padding-left: 48px !important;
    padding-right: 48px !important;
    max-width: 100% !important;
}

/* Typography matching DESIGN.md */
h1 {
    font-size: 24px !important;
    font-weight: 600 !important;
    color: #d3e4fe !important;
    line-height: 32px !important;
    letter-spacing: -0.02em !important;
    margin-bottom: 4px !important;
    padding-bottom: 0 !important;
}
.subtitle {
    font-size: 14px;
    color: #c6c6cd;
    margin-bottom: 24px;
    font-weight: 400;
}
h2 {
    font-size: 18px !important;
    font-weight: 600 !important;
    color: #d3e4fe !important;
    margin-top: 24px !important;
    margin-bottom: 12px !important;
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 0 !important;
}

/* Custom Top Nav Bar */
.top-nav {
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 56px;
    background: rgba(15, 23, 42, 0.85);
    backdrop-filter: blur(16px);
    border-bottom: 1px solid rgba(255,255,255,0.1);
    z-index: 999999;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0 24px;
}
.nav-brand { font-size: 18px; font-weight: 700; color: #38bdf8; letter-spacing: -0.02em; }
.nav-links { display: flex; gap: 32px; height: 100%; align-items: center; }
.nav-link { color: #94a3b8; font-size: 14px; cursor: pointer; height: 100%; display: flex; align-items: center; transition: color 0.2s; }
.nav-link:hover { color: #d3e4fe; }
.nav-link.active { color: #38bdf8; border-bottom: 2px solid #38bdf8; padding-top: 2px; font-weight: 500; }
.nav-icons { display: flex; gap: 16px; color: #38bdf8; align-items: center; }
.nav-icon { cursor: pointer; transition: color 0.2s; }
.nav-icon:hover { color: #d3e4fe; }
.user-avatar { width: 32px; height: 32px; border-radius: 50%; border: 1px solid rgba(255,255,255,0.2); background: #1e293b; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 700; color: #d3e4fe; }

/* Graph Container */
.graph-container {
    background-color: #102034;
    border: 1px solid #45464d;
    border-radius: 12px;
    height: 450px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
    margin-bottom: 32px;
}

/* GraphRAG Alerts */
.alert-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.alert-card { background: #1b2b3f; border: 1px solid #45464d; border-radius: 8px; padding: 16px; display: flex; flex-direction: column; gap: 6px; }
.alert-card.critical { border-color: rgba(239, 68, 68, 0.4); }
.alert-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
.alert-badge { font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 4px; letter-spacing: 0.05em; }
.badge-critical { background: rgba(239, 68, 68, 0.15); color: #ffb4ab; }
.badge-guideline { background: rgba(78, 222, 163, 0.15); color: #4edea3; }
.alert-time { color: #c6c6cd; font-size: 13px; }
.alert-title { color: #d3e4fe; font-size: 13px; font-weight: 600; font-family: 'Inter', monospace; }
.alert-desc { color: #c6c6cd; font-size: 13px; line-height: 1.5; }

/* Streamlit Chat Inputs overrides to match Copilot styling */
.stChatMessage {
    background: rgba(30, 41, 59, 0.6) !important;
    border: 1px solid rgba(56, 189, 248, 0.15) !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
    color: #cbd5e1 !important;
    font-size: 13px !important;
    line-height: 1.6 !important;
}
.stChatInputContainer {
    background: #0f172a !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 6px !important;
    padding-bottom: 12px !important;
}

/* Streamlit Selectbox override */
div[data-baseweb="select"] > div {
    background-color: #1e293b;
    border-color: #45464d;
    color: #d3e4fe;
}
</style>
""", unsafe_allow_html=True)

# ── Custom Top Navigation Injection ───────────────────────────────────────────

st.markdown("""
<div class="top-nav">
    <div class="nav-brand">ICU Cockpit</div>
    <div class="nav-links">
        <span class="nav-link">Summary</span>
        <span class="nav-link">Vitals</span>
        <span class="nav-link active">Knowledge Graph</span>
        <span class="nav-link">AI Analysis</span>
    </div>
    <div class="nav-icons">
        <span class="material-symbols-outlined nav-icon">notifications</span>
        <span class="material-symbols-outlined nav-icon">emergency</span>
        <span class="material-symbols-outlined nav-icon">settings</span>
        <div class="user-avatar">MD</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Data Loading & Logic ──────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_patients() -> list[dict]:
    try:
        from pipeline_b.patient_generator import load_all_patients
        return load_all_patients()
    except Exception as e:
        logger.error(f"Error loading patients: {e}")
        return []

@st.cache_resource
def load_model_manager():
    try:
        from layer3.model_loader import ModelManager
        mgr = ModelManager(n_ctx=2048, n_batch=512, verbose=False)
        mgr.load()
        return mgr
    except Exception:
        return None

@st.cache_data(ttl=60)
def build_graphs_cached(patient_ids: tuple) -> dict:
    try:
        from pipeline_b.patient_generator import load_all_patients
        from layer2.graph_builder import build_patient_graph
        patients = load_all_patients()
        return {p["patient_id"]: build_patient_graph(p) for p in patients}
    except Exception:
        return {}

def render_graph_html(G: nx.DiGraph, patient_id: str) -> str:
    try:
        from pyvis.network import Network
    except ImportError:
        return "<p style='color: #d3e4fe; padding: 20px;'>pyvis not installed</p>"

    # Background color matches the .graph-container exactly to be seamless
    net = Network(height="450px", width="100%", bgcolor="#102034", font_color="#d3e4fe", directed=True)

    TYPE_COLORS = {
        "Patient":   "#38bdf8", # Sky blue
        "Condition": "#ffb4ab", # Error
        "Medication":"#7bd0ff", # Secondary
        "LabResult": "#c4e7ff", # Secondary fixed
        "Vital":     "#4edea3", # Tertiary
        "Procedure": "#6ffbbe", # Tertiary fixed
    }

    for nid, attrs in G.nodes(data=True):
        if len(nid) > 40:
            continue
        ntype = attrs.get("node_type", "")
        label = attrs.get("name") or attrs.get("vital_type") or attrs.get("test") or nid
        color = TYPE_COLORS.get(ntype, "#c6c6cd")
        
        size = 20 if nid == patient_id else 12
        if attrs.get("is_critical") or attrs.get("flag") in ("HIGH", "LOW"):
            size = 18
            color = "#ef4444"

        tooltip = f"{ntype}: {label}"
        net.add_node(nid, label=str(label)[:25], color=color, size=size, title=tooltip, font={"size": 12, "face": "Inter"})

    for u, v, data in G.edges(data=True):
        if len(u) > 40 or len(v) > 40:
            continue
        rel = data.get("relation", "")
        color = "#ffb4ab" if rel == "contraindicates" else "#45464d"
        net.add_edge(u, v, title=rel, color=color, arrows="to", width=1.5)

    net.set_options("""
    {
      "physics": {"enabled": true, "barnesHut": {"springLength": 160, "damping": 0.1}},
      "interaction": {"hover": true, "tooltipDelay": 100}
    }
    """)
    return net.generate_html()


def main():
    patients = load_patients()
    if not patients:
        st.error("No patients found. Ensure pipeline_b is generating data.")
        return

    pid_tuple = tuple(p["patient_id"] for p in patients)
    G_all = build_graphs_cached(pid_tuple)

    # ── Right Sidebar: AI Copilot ──────────────────────────────────────────────
    with st.sidebar:
        # Copilot Header exactly matching code.html
        st.markdown("""
        <div style="padding: 4px 0 16px 0; border-bottom: 1px solid rgba(255,255,255,0.1); margin-bottom: 16px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span class="material-symbols-outlined" style="color: #38bdf8; font-size: 20px;">temp_preferences_custom</span>
                <h2 style="margin: 0 !important; font-size: 15px !important; padding: 0 !important; color: #38bdf8 !important;">AI Clinical Copilot</h2>
            </div>
            <div style="color: #94a3b8; font-size: 12px; margin-top: 4px;">Analytical Insights</div>
        </div>
        """, unsafe_allow_html=True)

        # Patient selector seamlessly integrated into the sidebar
        selected_pid = st.selectbox("Active Patient Context", [p["patient_id"] for p in patients], label_visibility="collapsed")
        
        selected_patient = next((p for p in patients if p["patient_id"] == selected_pid), patients[0])
        demo = selected_patient.get("demographics", {})
        
        chat_key = f"chat_{selected_pid}"
        if chat_key not in st.session_state:
            # Default Copilot message matching code.html mock
            st.session_state[chat_key] = [
                {"role": "assistant", "content": f"I noticed a new connection in the Knowledge Graph. Patient {selected_pid}'s recent telemetry and persistent hypotension strongly correlate with the Sepsis-3 node. Would you like to review the Hour-1 bundle checklist?"}
            ]

        chat_container = st.container(height=500)
        with chat_container:
            for msg in st.session_state[chat_key]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        if prompt := st.chat_input("Ask Copilot..."):
            st.session_state[chat_key].append({"role": "user", "content": prompt})
            st.rerun()

    # Post-rerun AI logic for sidebar chat
    if "chat_key" in locals() and st.session_state[chat_key][-1]["role"] == "user":
        mgr = load_model_manager()
        if mgr and mgr.models_ready:
            with st.spinner("Analyzing graph..."):
                response = mgr.gemma4_generate(f"User: {st.session_state[chat_key][-1]['content']}\nAnswer briefly as a clinical AI.", max_tokens=150)
                st.session_state[chat_key].append({"role": "assistant", "content": response.strip()})
                st.rerun()
        else:
            st.session_state[chat_key].append({"role": "assistant", "content": "LLM models not loaded in `models/` directory. Cannot process query."})
            st.rerun()


    # ── Main Content Canvas ────────────────────────────────────────────────────
    
    st.markdown("<h1>Knowledge Graph Real-time Analysis</h1>", unsafe_allow_html=True)
    st.markdown(f"<div class='subtitle'>Sepsis-3 Pathway Active Monitoring — {demo.get('name','')} (Bed {demo.get('bed','?')})</div>", unsafe_allow_html=True)

    # 1. Graph Container
    G_patient = G_all.get(selected_pid, nx.DiGraph())
    
    st.markdown('<div class="graph-container">', unsafe_allow_html=True)
    if G_patient.number_of_nodes() > 0:
        html_str = render_graph_html(G_patient, selected_pid)
        # Inject the generated HTML into the styled container
        st.components.v1.html(html_str, height=450, scrolling=False)
    else:
        st.info("Graph is empty — patient data may not be fully generated.")
    st.markdown('</div>', unsafe_allow_html=True)

    # 2. GraphRAG Alerts
    st.markdown('<h2><span class="material-symbols-outlined" style="color: #7bd0ff;">psychiatry</span> GraphRAG Reasoning Path</h2>', unsafe_allow_html=True)
    
    # Render the exact static layout from code.html, combined with dynamic triggers if we want to expand later.
    # To perfectly match the requested design redesign, we render the hardcoded alert layout as requested in code.html.
    st.markdown("""
    <div class="alert-grid">
        <!-- Alert Card 1 -->
        <div class="alert-card critical">
            <div class="alert-header">
                <span class="alert-badge badge-critical">CRITICAL PATH</span>
                <span class="alert-time">Just now</span>
            </div>
            <div class="alert-title">Hemodynamic Instability linked to Suspected Infection</div>
            <div class="alert-desc">Knowledge graph connects sustained low MAP despite fluid bolus directly to positive blood cultures (pending), activating Sepsis-3 clinical pathway node.</div>
        </div>
        <!-- Alert Card 2 -->
        <div class="alert-card">
            <div class="alert-header">
                <span class="alert-badge badge-guideline">GUIDELINE MATCH</span>
                <span class="alert-time">-5m</span>
            </div>
            <div class="alert-title">Vasopressor Initiation Recommendation</div>
            <div class="alert-desc">Graph traces lack of response to crystalloids to Surviving Sepsis guidelines, recommending immediate escalation to Norepinephrine node.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
