"""
Graph router — /api/graph/{patient_id}

FIX: Use cdn_resources="in_line" so ALL JavaScript and CSS from vis-network
is embedded directly inside the HTML string. This is the only approach that
works reliably inside an iframe srcdoc, because srcdoc iframes have no base
URL and cannot resolve any relative file paths like lib/bindings/utils.js.
"""

from __future__ import annotations

import logging

import networkx as nx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("icu.graph")
router = APIRouter(tags=["Graph"])

# ── Semantic node colors ──────────────────────────────────────────────────────
TYPE_COLORS = {
    "Patient":    "#38bdf8",   # cyan  — patient hub node
    "Condition":  "#f87171",   # red   — diagnoses
    "Medication": "#7bd0ff",   # blue  — drugs
    "LabResult":  "#fbbf24",   # amber — labs
    "Vital":      "#34d399",   # green — vitals
    "Procedure":  "#a78bfa",   # purple — procedures
}

# CSS injected into the pyvis HTML to enforce dark background and full-height canvas
_DARK_CSS = """<style>
html, body {
  margin: 0; padding: 0;
  background: #0d1117 !important;
  width: 100%; height: 100%;
  overflow: hidden;
  font-family: Inter, -apple-system, sans-serif;
}
h1, h2, center { display: none !important; }
.card { border: none !important; background: transparent !important; }
.card-body { padding: 0 !important; }
#mynetwork {
  width: 100% !important;
  height: 100vh !important;
  background: #0d1117 !important;
  border: none !important;
  position: absolute !important;
  top: 0; left: 0;
}
</style>"""


def _render_graph_html(G: nx.DiGraph, patient_id: str) -> str:
    """
    Convert a NetworkX DiGraph to a fully self-contained Pyvis HTML string.
    cdn_resources='in_line' embeds all JS/CSS — no relative paths, no CDN failures.
    """
    try:
        from pyvis.network import Network
    except ImportError:
        return "<p style='color:#9AA0A6;padding:24px;font-family:Inter,sans-serif'>pyvis not installed. Run: pip install pyvis</p>"

    if G.number_of_nodes() == 0:
        return "<p style='color:#9AA0A6;padding:24px;font-family:Inter,sans-serif'>Graph is empty — patient data not fully loaded yet.</p>"

    net = Network(
        height="100vh",
        width="100%",
        bgcolor="#0d1117",
        font_color="#d3e4fe",
        directed=True,
        cdn_resources="in_line",   # ← THE KEY FIX: embeds vis-network JS/CSS inline
    )

    for nid, attrs in G.nodes(data=True):
        if len(str(nid)) > 40:
            continue  # skip excessively long PrimeKG IDs

        ntype = attrs.get("node_type", "")
        label = (
            attrs.get("name")
            or attrs.get("vital_type")
            or attrs.get("test")
            or str(nid)
        )
        color = TYPE_COLORS.get(ntype, "#6b7280")
        size = 26 if str(nid) == patient_id else 14

        # Critical / flagged nodes — highlight in red
        if attrs.get("is_critical") or attrs.get("flag") in ("HIGH", "LOW", "CRITICAL"):
            size = 20
            color = "#ef4444"

        tooltip = f"<b style='color:#f3f4f6'>{ntype}</b>: {label}"
        if attrs.get("value"):
            tooltip += f"<br/>Value: <b>{attrs['value']}</b> {attrs.get('unit', '')}"
        if attrs.get("flag") and attrs["flag"] != "NORMAL":
            tooltip += f"<br/>Flag: <b style='color:#fbbf24'>{attrs['flag']}</b>"

        net.add_node(
            str(nid),
            label=str(label)[:28],
            color=color,
            size=size,
            title=tooltip,
            font={"size": 12, "face": "Inter", "color": "#d3e4fe"},
        )

    for u, v, data in G.edges(data=True):
        if len(str(u)) > 40 or len(str(v)) > 40:
            continue
        rel = data.get("relation", "")
        edge_color = "#ef4444" if rel == "contraindicates" else "#374151"
        net.add_edge(str(u), str(v), title=rel or "", color=edge_color, arrows="to", width=1.5)

    net.set_options("""{
      "physics": {
        "enabled": true,
        "barnesHut": {
          "springLength": 160,
          "damping": 0.15,
          "gravitationalConstant": -3500,
          "centralGravity": 0.15
        },
        "stabilization": {"iterations": 150, "updateInterval": 25}
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 60,
        "navigationButtons": false,
        "keyboard": false,
        "zoomView": true,
        "dragView": true
      },
      "nodes": {
        "shape": "dot",
        "borderWidth": 2,
        "font": {"face": "Inter", "color": "#d3e4fe", "size": 12}
      },
      "edges": {
        "color": {"inherit": false},
        "smooth": {"type": "continuous"},
        "font": {"size": 10, "color": "#6b7280", "face": "Inter"}
      }
    }""")

    html = net.generate_html(notebook=False)

    # Inject dark theme CSS override before </head>
    if "</head>" in html:
        html = html.replace("</head>", _DARK_CSS + "\n</head>")
    else:
        html = _DARK_CSS + html

    return html


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.get("/graph/{patient_id}")
async def get_graph(patient_id: str, request: Request):
    """
    Returns { html: str, nodes: int, edges: int }.
    Frontend injects html directly into iframe srcdoc.
    """
    state = request.app.state.icu_state
    G = state.graphs.get(patient_id)

    if G is None:
        raise HTTPException(status_code=404, detail=f"Graph not found for {patient_id}")

    html = _render_graph_html(G, patient_id)
    return JSONResponse({
        "html": html,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
    })
