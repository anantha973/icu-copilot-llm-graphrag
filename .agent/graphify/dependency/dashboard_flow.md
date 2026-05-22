# ICU Dashboard Execution Flow

This document maps the architectural dependencies and data execution flow of the primary dashboard interface (`src/dashboard/app.py`).

## 1. Core Data Ingestion (Session State Initialization)
Upon launching, the dashboard loads three core memory layers synchronously:

- **Patient State:** `load_patients()`
  - **Dependency:** `pipeline_b.patient_generator.load_all_patients`
  - **Function:** Loads all simulated or live ICU patient states, vitals, and demographics.

- **Global Knowledge Base (GraphRAG):** `load_global_kb()`
  - **Dependencies:** 
    - `pipeline_a.guideline_indexer.index_guidelines` (FAISS indexing for medical guidelines)
    - `pipeline_a.primekg_loader.load_primekg` (PrimeKG Medical Knowledge Graph)
    - `pipeline_a.global_kb_retriever.GlobalKBRetriever` (The core semantic retrieval engine)
  - **Function:** Initializes the medical grounding memory that acts as the truth engine for AI reasoning.

- **LLM Engine:** `load_model_manager()`
  - **Dependency:** `layer3.model_loader.ModelManager`
  - **Function:** Loads `medgemma` and `gemma4` GGUF models. Runs lazily to avoid blocking UI rendering.

## 2. Graph & Severity Computation (Cached Layer)
For each patient, the dashboard pre-computes specific patient states:

- **Patient Graph Extraction:** `build_graphs_cached()`
  - **Dependency:** `layer2.graph_builder.build_patient_graph`
  - **Function:** Generates a NetworkX DiGraph mapping the patient's individual conditions, meds, and critical vitals.

- **Rule-Based Triage:** `get_severity_cached()`
  - **Dependency:** `layer3.severity_classifier.rule_based_severity`
  - **Function:** Pre-computes CRITICAL (Red), WATCH (Amber), or STABLE (Green) statuses based on rigid vitals/lab triggers.

## 3. UI Presentation Flow
The frontend operates primarily across two views, managed by Streamlit's `session_state.current_view`.

### A. Ward Overview (High-Density Triage)
- **Visuals:** Grid of patient cards displaying critical vitals (HR, SpO2, MAP, Temp).
- **Logic:** Calls `vital_color_class()` to highlight abnormal parameters instantly based on the pre-computed severity.

### B. Patient Detail (Expanded Deep Dive)
- **Summary:** Lists active diagnoses, current meds, and latest labs.
- **Vitals:** Plots a `plotly` sparkline chart (`render_vitals_chart`) and computes numerical trend arrows.
- **Interactive Graph:** Leverages `pyvis` via `render_graph_html` to render the individual `layer2` patient graph visually.
- **AI Explanation Engine:** 
  - Retrieves patient text context via `layer2.subgraph_retriever.get_patient_subgraph_text`.
  - Merges it with PrimeKG medical guidelines (`kb.get_full_context()`).
  - Feeds the combined `patient_data` + `medical_knowledge` into Gemma 4 for conversational clinical assistance.

## Architectural Notes
- The dashboard successfully isolates presentation logic from cognitive retrieval (`layer2`) and reasoning (`layer3`).
- Heavy operations are safely protected behind `@st.cache_data` and `@st.cache_resource`, ensuring responsive UX despite intensive backend graphing.
