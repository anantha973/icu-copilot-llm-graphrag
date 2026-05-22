"""
ICU Clinical Copilot Dashboard
Layer 4 — Streamlit Application

Run: streamlit run src/dashboard/app.py
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

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[2]))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ICU Clinical Copilot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Reference UI Styles */
.patient-card {
  background: #0d1221;
  border: 0.5px solid rgba(255,255,255,0.08);
  border-radius: 8px;
  padding: 10px;
  cursor: pointer;
  transition: border-color 0.15s;
  position: relative;
  margin-bottom: 8px;
}
.patient-card:hover { border-color: rgba(56,189,248,0.35); }
.patient-card.selected { border-color: #38bdf8; background: #0f1a2e; }
.patient-card.critical { border-color: #f87171; background: #150d0d; animation: critblink 1s ease-in-out infinite alternate; }
@keyframes critblink { from{border-color:#f87171} to{border-color:rgba(248,113,113,0.35)} }
.patient-card.warning { border-color: #fb923c; background: #160f08; }

.card-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 7px; }
.bed-id { font-size: 11px; font-weight: 500; color: #94a3b8; }
.pt-name { font-size: 13px; font-weight: 500; color: #e2e8f0; }
.pt-meta { font-size: 10px; color: #475569; }
.status-dot { width: 8px; height: 8px; border-radius: 50%; margin-top: 2px; }
.dot-stable { background: #4ade80; }
.dot-warning { background: #fb923c; animation: pulse-dot 1.2s infinite; }
.dot-critical { background: #f87171; animation: pulse-dot 0.6s infinite; }
@keyframes pulse-dot { 0%,100%{opacity:1} 50%{opacity:0.3} }

.vitals-row { display: grid; grid-template-columns: 1fr 1fr; gap: 4px; margin-top: 6px; }
.vital { background: rgba(255,255,255,0.03); border-radius: 5px; padding: 4px 6px; }
.vital .v-lbl { font-size: 9px; color: #475569; text-transform: uppercase; letter-spacing: 0.05em; }
.vital .v-val { font-size: 13px; font-weight: 500; color: #cbd5e1; line-height: 1.2; }
.vital .v-val.alert { color: #f87171; }
.vital .v-val.warn { color: #fb923c; }
.vital .v-val.ok { color: #4ade80; }

.card-footer { margin-top: 6px; font-size: 9px; color: #334155; display: flex; justify-content: space-between; }

/* AI explanation box */
.explanation-box {
    background: #0d1221;
    border: 0.5px solid rgba(139,92,246,0.35);
    border-left: 3px solid #8b5cf6;
    border-radius: 8px;
    padding: 16px;
    margin: 10px 0;
    font-size: 14px;
    line-height: 1.7;
    color: #e2e8f0;
}

/* Section headers */
.section-header {
    font-size: 10px; color: #475569; text-transform: uppercase; letter-spacing: 0.08em;
    margin-bottom: 6px;
}

/* Metric row */
.metric-row {
    display: flex;
    gap: 16px;
    margin: 8px 0;
}
.metric-box {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 8px 14px;
    flex: 1;
    text-align: center;
}
.metric-value { font-size: 22px; font-weight: 700; color: #e2e8f0; }
.metric-label { font-size: 11px; color: #8b949e; }
</style>
""", unsafe_allow_html=True)


# ── Session state + data loading ──────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_patients() -> list[dict]:
    from pipeline_b.patient_generator import load_all_patients
    return load_all_patients()


@st.cache_resource
def load_global_kb():
    """Load PrimeKG + FAISS index (cached across sessions)."""
    from pipeline_a.guideline_indexer import index_guidelines
    from pipeline_a.primekg_loader import (
        GRAPH_PKL, load_primekg, build_name_index
    )
    from pipeline_a.global_kb_retriever import GlobalKBRetriever

    # Guidelines (always available)
    faiss_idx, chunks, embedder = index_guidelines()

    # PrimeKG (may not be built yet)
    try:
        if GRAPH_PKL.exists():
            primekg = load_primekg()
            name_idx = build_name_index(primekg)
        else:
            primekg, name_idx = None, {}
    except Exception:
        primekg, name_idx = None, {}

    return GlobalKBRetriever(
        primekg=primekg,
        name_index=name_idx,
        faiss_index=faiss_idx,
        guideline_chunks=chunks,
        embedder=embedder,
    )


@st.cache_resource
def load_model_manager():
    from layer3.model_loader import ModelManager
    mgr = ModelManager(n_ctx=2048, n_batch=512, verbose=False)
    mgr.load()
    return mgr


@st.cache_data(ttl=60)
def build_graphs_cached(patient_ids: tuple) -> dict:
    """Cache key is just patient IDs — actual data loaded inside."""
    from pipeline_b.patient_generator import load_all_patients
    from layer2.graph_builder import build_patient_graph
    patients = load_all_patients()
    return {p["patient_id"]: build_patient_graph(p) for p in patients}


@st.cache_data(ttl=60)
def get_severity_cached(patient_ids: tuple) -> dict:
    from pipeline_b.patient_generator import load_all_patients
    from layer3.severity_classifier import rule_based_severity
    patients = load_all_patients()
    results = {}
    for p in patients:
        sev, trig = rule_based_severity(p)
        results[p["patient_id"]] = {
            "severity": sev,
            "triggered_rules": trig,
            "confidence": 0.90 if sev != "GREEN" else 0.85,
        }
    return results


# ── Helper functions ──────────────────────────────────────────────────────────

SEVERITY_EMOJI = {"RED": "🔴", "AMBER": "🟡", "GREEN": "🟢"}
SEVERITY_LABEL = {"RED": "CRITICAL", "AMBER": "WATCH", "GREEN": "STABLE"}

VITAL_CRITICAL = {"MAP": 65, "SpO2": 90, "HR": 130}
VITAL_WARN     = {"MAP": 70, "SpO2": 94, "HR": 110}


def vital_color_class(vtype: str, value: float) -> str:
    low = {"MAP", "SpO2"}
    crit = VITAL_CRITICAL.get(vtype, None)
    warn = VITAL_WARN.get(vtype, None)
    if crit is not None:
        if (vtype in low and value < crit) or (vtype not in low and value > crit):
            return "alert"
        if warn is not None:
            if (vtype in low and value < warn) or (vtype not in low and value > warn):
                return "warn"
    return "ok"


def trend_arrow(timeseries: list[dict], key: str) -> str:
    if len(timeseries) < 4:
        return "→"
    recent = [t[key] for t in timeseries[-4:] if key in t]
    if len(recent) < 2:
        return "→"
    delta = recent[-1] - recent[0]
    if abs(delta) < 1:
        return "→"
    return "↑" if delta > 0 else "↓"


def render_vitals_chart(timeseries: list[dict]) -> go.Figure:
    """Render sparkline charts for key vitals."""
    if not timeseries:
        return go.Figure()

    timestamps = [t.get("timestamp", "")[-8:-3] for t in timeseries]
    fig = go.Figure()

    vital_cfg = [
        ("MAP",  "#ef4444", 65),
        ("SpO2", "#3b82f6", 90),
        ("HR",   "#f59e0b", 130),
    ]

    for key, color, threshold in vital_cfg:
        values = [t.get(key) for t in timeseries if key in t]
        if not values:
            continue
        fig.add_trace(go.Scatter(
            x=timestamps[-len(values):],
            y=values,
            name=key,
            line=dict(color=color, width=2),
            mode="lines",
        ))
        fig.add_hline(
            y=threshold, line_dash="dot",
            line_color=color, opacity=0.4,
            annotation_text=f"{key} threshold",
            annotation_font_size=9,
        )

    fig.update_layout(
        height=220,
        margin=dict(l=0, r=0, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,17,23,1)",
        font=dict(color="#8b949e", size=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left"),
        xaxis=dict(showgrid=False, color="#30363d"),
        yaxis=dict(showgrid=True, gridcolor="#21262d", color="#8b949e"),
    )
    return fig


def render_graph_html(G: nx.DiGraph, patient_id: str) -> str:
    """Render patient graph as Pyvis HTML string."""
    try:
        from pyvis.network import Network
    except ImportError:
        return "<p>pyvis not installed</p>"

    net = Network(height="400px", width="100%", bgcolor="#0d1117",
                  font_color="#e6edf3", directed=True)

    TYPE_COLORS = {
        "Patient":   "#6366f1",
        "Condition": "#ef4444",
        "Medication":"#f59e0b",
        "LabResult": "#3b82f6",
        "Vital":     "#22c55e",
        "Procedure": "#8b5cf6",
    }

    for nid, attrs in G.nodes(data=True):
        if len(nid) > 40:  # skip long PrimeKG node IDs
            continue
        ntype = attrs.get("node_type", "")
        label = attrs.get("name") or attrs.get("vital_type") or attrs.get("test") or nid
        color = TYPE_COLORS.get(ntype, "#6b7280")

        # Make critical nodes bigger/brighter
        size = 20 if nid == patient_id else 14
        if attrs.get("is_critical") or attrs.get("flag") in ("HIGH", "LOW"):
            size = 18
            color = "#ef4444"

        tooltip = f"{ntype}: {label}"
        if attrs.get("value"):
            tooltip += f" = {attrs['value']} {attrs.get('unit','')}"
        if attrs.get("flag") and attrs["flag"] != "NORMAL":
            tooltip += f" [{attrs['flag']}]"

        net.add_node(nid, label=str(label)[:20], color=color,
                     size=size, title=tooltip, font={"size": 11})

    for u, v, data in G.edges(data=True):
        if len(u) > 40 or len(v) > 40:
            continue
        rel = data.get("relation", "")
        color = "#ef4444" if rel == "contraindicates" else "#30363d"
        net.add_edge(u, v, title=rel, color=color,
                     arrows="to", width=1.5)

    net.set_options("""
    {
      "physics": {"enabled": true, "barnesHut": {"springLength": 120}},
      "interaction": {"hover": true, "tooltipDelay": 100}
    }
    """)
    return net.generate_html()


# ── Main App ──────────────────────────────────────────────────────────────────

def main():
    if "current_view" not in st.session_state:
        st.session_state.current_view = "Overview"

    # Header
    col_title, col_refresh = st.columns([6, 1])
    with col_title:
        st.markdown("## 🏥 ICU Clinical Copilot")
        st.caption("AI-powered patient monitoring · GraphRAG-grounded explanations")
    with col_refresh:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # Navbar (Website style)
    col_nav1, col_nav2, _ = st.columns([2, 2, 8])
    with col_nav1:
        if st.button("📊 Ward Overview", use_container_width=True, type="primary" if st.session_state.current_view == "Overview" else "secondary"):
            st.session_state.current_view = "Overview"
            st.rerun()
    with col_nav2:
        if st.button("🧑‍⚕️ Patient Detail", use_container_width=True, type="primary" if st.session_state.current_view == "Detail" else "secondary"):
            st.session_state.current_view = "Detail"
            st.rerun()
    st.divider()

    # Load data
    with st.spinner("Loading patients…"):
        patients = load_patients()

    if not patients:
        st.error("No patients found. Run: python scripts/build_global_kb.py")
        return

    with st.spinner("Loading knowledge base…"):
        kb = load_global_kb()

    with st.spinner("Building patient graphs…"):
        pid_tuple = tuple(p["patient_id"] for p in patients)
        G_all = build_graphs_cached(pid_tuple)

    sev_map = get_severity_cached(tuple(p["patient_id"] for p in patients))

    # Sidebar: patient list
    with st.sidebar:
        st.markdown("### 🧑‍⚕️ Patient List")
        st.caption(f"{len(patients)} patients monitored")
        st.divider()

        selected_pid = st.session_state.get("selected_pid", patients[0]["patient_id"])

        for p in patients:
            pid = p["patient_id"]
            sev = sev_map[pid]["severity"]
            demo = p.get("demographics", {})
            label = f"{SEVERITY_EMOJI[sev]} {pid} — {demo.get('name', '').split()[0]} (Bed {demo.get('bed', '?')})"
            if st.button(label, key=f"side_{pid}", use_container_width=True):
                st.session_state["selected_pid"] = pid
                st.session_state.current_view = "Detail"
                st.rerun()

        st.divider()
        st.markdown("**Legend**")
        st.markdown("🔴 CRITICAL · 🟡 WATCH · 🟢 STABLE")
        primekg_status = "✅ Loaded" if kb.primekg else "⚠️ Not loaded (run build_global_kb.py)"
        st.caption(f"PrimeKG: {primekg_status}")

    # Main area: Navigation logic
    if st.session_state.current_view == "Overview":
        st.markdown("### Patient Overview")

        cols = st.columns(4)
        for i, p in enumerate(patients):
            pid = p["patient_id"]
            sev = sev_map[pid]["severity"]
            demo = p.get("demographics", {})
            vitals = p.get("vitals_latest", {})

            card_class = "critical" if sev == "RED" else "warning" if sev == "AMBER" else ""
            dot_class = "dot-critical" if sev == "RED" else "dot-warning" if sev == "AMBER" else "dot-stable"

            with cols[i % 4]:
                card_content = f"""
                <div class="patient-card {card_class}" id="card-{pid}">
                    <div class="card-header">
                        <div>
                            <div class="bed-id">Bed {demo.get('bed', '?')}</div>
                            <div class="pt-name">{demo.get('name','').split()[0]}</div>
                            <div class="pt-meta">{demo.get('sex','?')}, {demo.get('age','?')} · {pid}</div>
                        </div>
                        <div class="status-dot {dot_class}"></div>
                    </div>
                    <div class="vitals-row">
                        <div class="vital"><div class="v-lbl">HR</div><div class="v-val {vital_color_class('HR', vitals.get('HR',0))}">{vitals.get('HR','—')}</div></div>
                        <div class="vital"><div class="v-lbl">SpO₂</div><div class="v-val {vital_color_class('SpO2', vitals.get('SpO2',0))}">{vitals.get('SpO2','—')}</div></div>
                        <div class="vital"><div class="v-lbl">MAP</div><div class="v-val {vital_color_class('MAP', vitals.get('MAP',0))}">{vitals.get('MAP','—')}</div></div>
                        <div class="vital"><div class="v-lbl">Temp</div><div class="v-val {vital_color_class('Temp', vitals.get('Temp',0))}">{vitals.get('Temp','—')}</div></div>
                    </div>
                    <div class="card-footer"><span>RR: {vitals.get('RR','—')}/min</span><span>ID: {pid}</span></div>
                </div>
                """
                st.markdown(card_content, unsafe_allow_html=True)
                if st.button(f"View {pid}", key=f"btn_{pid}", use_container_width=True):
                    st.session_state["selected_pid"] = pid
                    st.session_state.current_view = "Detail"
                    st.rerun()

    elif st.session_state.current_view == "Detail":
        # Expanded patient view
        selected_pid = st.session_state.get("selected_pid", patients[0]["patient_id"])
        selected = next((p for p in patients if p["patient_id"] == selected_pid), patients[0])
        G_patient = G_all.get(selected_pid, nx.DiGraph())
        sev_data = sev_map.get(selected_pid, {})
        sev = sev_data.get("severity", "GREEN")

        st.markdown(f"### {SEVERITY_EMOJI[sev]} Patient {selected_pid} — Expanded View")
        demo = selected.get("demographics", {})
        st.caption(
            f"{demo.get('name','?')} · {demo.get('age','?')}y {demo.get('sex','?')} · "
            f"Bed {demo.get('bed','?')} · Admitted: {demo.get('admission_date','')[:10]}"
        )

        tab1, tab2, tab3, tab4 = st.tabs(["📋 Summary", "📈 Vitals", "🔗 Graph", "🤖 AI Explanation"])

        # Tab 1: Summary
        with tab1:
            col_l, col_r = st.columns([2, 1])

            with col_l:
                st.markdown('<div class="section-header">Active Diagnoses</div>', unsafe_allow_html=True)
                for c in selected.get("conditions", []):
                    status_color = "#ef4444" if c["status"] == "active" else "#6b7280"
                    st.markdown(
                        f'<span style="color:{status_color};">●</span> **{c["name"]}** '
                        f'<span style="color:#8b949e; font-size:12px;">[{c["status"]}]</span>',
                        unsafe_allow_html=True,
                    )

                st.divider()
                st.markdown('<div class="section-header">Medications</div>', unsafe_allow_html=True)
                for m in selected.get("medications", []):
                    st.markdown(
                        f'💊 **{m["name"]}** {m.get("dose","")} {m.get("route","")}'
                        + (f'  \n  <span style="color:#8b949e; font-size:12px;">for {m["indication"]}</span>'
                           if m.get("indication") else ""),
                        unsafe_allow_html=True,
                    )

            with col_r:
                st.markdown('<div class="section-header">Latest Labs</div>', unsafe_allow_html=True)
                for l in selected.get("lab_results", [])[:8]:
                    flag = l.get("flag", "NORMAL")
                    color = "#ef4444" if flag in ("HIGH", "CRITICAL") else "#f59e0b" if flag == "LOW" else "#22c55e"
                    st.markdown(
                        f'<span style="color:{color}; font-size:13px;">● {l["test"]}: '
                        f'**{l["value"]}** {l["unit"]} [{flag}]</span>',
                        unsafe_allow_html=True,
                    )

        # Tab 2: Vitals
        with tab2:
            ts = selected.get("vitals_timeseries", [])
            if ts:
                fig = render_vitals_chart(ts)
                st.plotly_chart(fig, use_container_width=True)

            st.divider()
            vitals = selected.get("vitals_latest", {})
            cols_v = st.columns(5)
            vital_pairs = [("MAP", "mmHg"), ("HR", "bpm"), ("SpO2", "%"), ("RR", "/min"), ("Temp", "°C")]
            for i, (vt, unit) in enumerate(vital_pairs):
                val = vitals.get(vt, "—")
                arrow = trend_arrow(ts, vt) if ts else "→"
                vc = vital_color_class(vt, float(val)) if isinstance(val, (int, float)) else "vital-normal"
                with cols_v[i]:
                    st.markdown(
                        f'<div class="metric-box">'
                        f'<div class="metric-value {vc}">{val}{arrow}</div>'
                        f'<div class="metric-label">{vt} ({unit})</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        # Tab 3: Graph
        with tab3:
            st.caption("Interactive patient knowledge graph. Hover nodes for details.")
            if G_patient.number_of_nodes() > 0:
                html_str = render_graph_html(G_patient, selected_pid)
                st.components.v1.html(html_str, height=440, scrolling=False)
            else:
                st.info("Graph is empty — patient data may not be loaded.")

            col_stats1, col_stats2, col_stats3 = st.columns(3)
            with col_stats1:
                st.metric("Graph Nodes", G_patient.number_of_nodes())
            with col_stats2:
                st.metric("Graph Edges", G_patient.number_of_edges())
            with col_stats3:
                linked = sum(1 for _, d in G_patient.nodes(data=True)
                             if d.get("primekg_node_id"))
                st.metric("PrimeKG Links", linked)

        # Tab 4: AI Explanation
        with tab4:
            from layer2.subgraph_retriever import get_patient_subgraph_text
            patient_ctx = get_patient_subgraph_text(G_patient, selected_pid)

            # Severity display
            sev_colors = {"RED": "#ef4444", "AMBER": "#f59e0b", "GREEN": "#22c55e"}
            st.markdown(
                f'<div style="font-size:24px; font-weight:700; color:{sev_colors[sev]};">'
                f'{SEVERITY_EMOJI[sev]} {sev} — {SEVERITY_LABEL[sev]}'
                f'</div>',
                unsafe_allow_html=True,
            )
            triggers = sev_data.get("triggered_rules", [])
            if triggers:
                st.markdown("**Triggered rules:**")
                for t in triggers:
                    st.markdown(f"  - {t}")

            st.divider()

            # Patient context display
            with st.expander("📊 Patient Graph Context (raw)", expanded=False):
                st.code(patient_ctx, language=None)

            # LLM Explanation
            st.markdown('<div class="section-header">AI Clinical Explanation</div>', unsafe_allow_html=True)

            # Load model manager (lazy)
            try:
                mgr = load_model_manager()
                models_loaded = mgr.models_ready
            except Exception:
                mgr = None
                models_loaded = False

            if not models_loaded:
                st.markdown(
                    '<div class="explanation-box">'
                    '⚠️ <strong>LLM models not loaded.</strong><br>'
                    'Place GGUF files in <code>models/</code> to enable AI explanations.<br><br>'
                    '<strong>Expected files:</strong><br>'
                    '• <code>medgemma-4b-q4_k_m.gguf</code> (from HuggingFace: google/medgemma-4b-it)<br>'
                    '• <code>gemma4-e4b-uncensored-q4_k_m.gguf</code> (from HuggingFace: HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive)<br><br>'
                    'Rule-based severity classification is active and accurate.<br>'
                    'AI narrative explanation requires the LLM models.'
                    '</div>',
                    unsafe_allow_html=True,
                )
            else:
                chat_key = f"chat_{selected_pid}"
                if chat_key not in st.session_state:
                    st.session_state[chat_key] = [
                        {"role": "assistant", "content": f"How can I help you analyze the clinical state of Patient {selected_pid}?"}
                    ]

                chat_container = st.container(height=450)
                with chat_container:
                    for msg in st.session_state[chat_key]:
                        with st.chat_message(msg["role"]):
                            st.markdown(msg["content"])

                if prompt := st.chat_input("Ask a clinical question about this patient..."):
                    st.session_state[chat_key].append({"role": "user", "content": prompt})
                    with chat_container:
                        with st.chat_message("user"):
                            st.markdown(prompt)
                        with st.chat_message("assistant"):
                            with st.spinner("Analyzing with Gemma 4..."):
                                conditions = [c["name"] for c in selected.get("conditions", [])]
                                meds = [m["name"] for m in selected.get("medications", [])]
                                kb_ctx = kb.get_full_context(conditions, meds, prompt)
                                knowledge_ctx = kb_ctx["graph_context"] + "\n\n" + kb_ctx["guideline_context"]
                                
                                history_text = ""
                                for m in st.session_state[chat_key][:-1][-6:]: # Last 6 messages
                                    history_text += f"{m['role'].capitalize()}: {m['content']}\n\n"
                                
                                chat_prompt = f"""You are an expert ICU Clinical AI Copilot.
Answer the user's clinical question based ONLY on the patient's data and medical knowledge provided. Be concise and clinical.

<patient_data>
{patient_ctx}
</patient_data>

<medical_knowledge>
{knowledge_ctx}
</medical_knowledge>

<conversation_history>
{history_text}
</conversation_history>

User: {prompt}
Assistant:"""
                                response = mgr.gemma4_generate(chat_prompt, max_tokens=600, temperature=0.3)
                                st.markdown(response.strip())
                                st.session_state[chat_key].append({"role": "assistant", "content": response.strip()})

            # Always show PrimeKG context (doesn't need LLM)
            conditions = [c["name"] for c in selected.get("conditions", [])]
            meds = [m["name"] for m in selected.get("medications", [])]

            with st.expander("🧠 PrimeKG Medical Knowledge Context", expanded=False):
                kb_ctx = kb.get_full_context(
                    conditions, meds,
                    f"Why is {selected_pid} {sev}?"
                )
                st.markdown("**Graph context:**")
                st.code(kb_ctx["graph_context"][:2000], language=None)
                st.markdown("**Guideline context:**")
                st.code(kb_ctx["guideline_context"][:1000], language=None)


if __name__ == "__main__":
    main()
