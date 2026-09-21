"""
Rigorous 20-Question Common Clinical Evaluation Script — ICU Copilot
Runs 20 common nurse/patient questions across Cardiology, Nephrology, Endocrine, Pulmonary, and Critical Care.
Captures full GraphRAG context, Gemma 4 generation, and latency.
"""

import sys
import json
import time
from pathlib import Path

# Setup paths
PROJ_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(PROJ_DIR / "src"))

from knowledge_base.primekg_loader import (
    load_primekg,
    build_name_index,
    PRIMEKG_FAISS_INDEX,
    PRIMEKG_FAISS_MAP,
)
from knowledge_base.guideline_indexer import index_guidelines
from knowledge_base.retriever import GlobalKBRetriever
from llm_reasoning.model_loader import ModelManager
import faiss
import pickle

QUESTIONS = [
    # ── Category 1: Cardiovascular
    {
        "id": 1,
        "category": "Cardiovascular",
        "conditions": ["hypertension"],
        "drugs": ["lisinopril"],
        "query": "Can the patient take potassium supplements or salt substitutes while on lisinopril?",
    },
    {
        "id": 2,
        "category": "Cardiovascular",
        "conditions": ["heart attack"],
        "drugs": ["aspirin"],
        "query": "Can we add Plavix (clopidogrel) for dual antiplatelet therapy after a heart attack?",
    },
    {
        "id": 3,
        "category": "Cardiovascular",
        "conditions": ["atrial fibrillation"],
        "drugs": ["metoprolol"],
        "query": "Is there a risk if we add digoxin for rate control in this AFib patient already on metoprolol?",
    },
    {
        "id": 4,
        "category": "Cardiovascular",
        "conditions": ["heart failure"],
        "drugs": ["carvedilol"],
        "query": "Can we suddenly increase beta blocker carvedilol during acute decompensated heart failure with fluid retention?",
    },
    {
        "id": 5,
        "category": "Cardiovascular",
        "conditions": ["chest pain", "angina"],
        "drugs": ["nitroglycerin"],
        "query": "Can this patient taking nitroglycerin also take Viagra (sildenafil) for erectile dysfunction?",
    },
    # ── Category 2: Renal & Urinary
    {
        "id": 6,
        "category": "Renal & Urinary",
        "conditions": ["CKD", "chronic kidney disease"],
        "drugs": ["metformin"],
        "query": "Can we continue metformin if the patient's kidney failure worsens?",
    },
    {
        "id": 7,
        "category": "Renal & Urinary",
        "conditions": ["acute kidney injury"],
        "drugs": ["vancomycin"],
        "query": "What is the risk of administering gentamicin or vancomycin in a patient with acute kidney damage?",
    },
    {
        "id": 8,
        "category": "Renal & Urinary",
        "conditions": ["kidney stones"],
        "drugs": ["calcium carbonate"],
        "query": "Is hydrochlorothiazide indicated or contraindicated for recurrent calcium kidney stones?",
    },
    {
        "id": 9,
        "category": "Renal & Urinary",
        "conditions": ["swollen legs", "fluid overload"],
        "drugs": ["lasix"],
        "query": "What electrolytes should the nurse monitor while giving Lasix (furosemide) for swollen legs?",
    },
    {
        "id": 10,
        "category": "Renal & Urinary",
        "conditions": ["renal failure"],
        "drugs": ["spironolactone"],
        "query": "What is the danger of giving spironolactone (Aldactone) to a patient with severe kidney failure?",
    },
    # ── Category 3: Endocrine & Diabetes
    {
        "id": 11,
        "category": "Endocrine & Diabetes",
        "conditions": ["type 2 diabetes", "hypertension"],
        "drugs": ["insulin", "metoprolol"],
        "query": "Can beta blockers like metoprolol mask the warning signs of low blood sugar (hypoglycemia) in a diabetic patient on insulin?",
    },
    {
        "id": 12,
        "category": "Endocrine & Diabetes",
        "conditions": ["type 2 diabetes"],
        "drugs": ["metformin"],
        "query": "What happens to blood glucose if this diabetic patient is started on high-dose prednisone (steroids)?",
    },
    {
        "id": 13,
        "category": "Endocrine & Diabetes",
        "conditions": ["type 2 diabetes"],
        "drugs": ["glipizide"],
        "query": "What is the risk if a patient taking glipizide skips meals or receives excessive dosage?",
    },
    {
        "id": 14,
        "category": "Endocrine & Diabetes",
        "conditions": ["type 2 diabetes"],
        "drugs": ["empagliflozin"],
        "query": "What is the risk of euglycemic diabetic ketoacidosis (DKA) with SGLT2 inhibitors like Jardiance (empagliflozin)?",
    },
    # ── Category 4: Pulmonary & Respiratory
    {
        "id": 15,
        "category": "Pulmonary & Respiratory",
        "conditions": ["asthma", "wheezing"],
        "drugs": ["albuterol"],
        "query": "Can we give non-selective beta blocker propranolol (Inderal) to a patient with asthma?",
    },
    {
        "id": 16,
        "category": "Pulmonary & Respiratory",
        "conditions": ["COPD"],
        "drugs": ["tiotropium"],
        "query": "Why must high-flow oxygen be used cautiously in patients with severe COPD and emphysema?",
    },
    {
        "id": 17,
        "category": "Pulmonary & Respiratory",
        "conditions": ["pneumonia"],
        "drugs": ["atorvastatin"],
        "query": "Is there a drug interaction if we start clarithromycin for pneumonia in a patient taking atorvastatin (Lipitor)?",
    },
    # ── Category 5: Critical Care & Hematology
    {
        "id": 18,
        "category": "Critical Care & Hematology",
        "conditions": ["blood clot in lung", "pulmonary embolism"],
        "drugs": ["heparin"],
        "query": "Can we bridge heparin to Coumadin (warfarin) for pulmonary embolism, and what is the bleeding risk?",
    },
    {
        "id": 19,
        "category": "Critical Care & Hematology",
        "conditions": ["sepsis"],
        "drugs": ["norepinephrine"],
        "query": "Is norepinephrine (Levophed) the first-line vasopressor for septic shock after fluid resuscitation?",
    },
    {
        "id": 20,
        "category": "Critical Care & Hematology",
        "conditions": ["stomach ulcer", "gi bleed"],
        "drugs": ["pantoprazole"],
        "query": "Can we give blood thinners like heparin or Lovenox (enoxaparin) to a patient with active stomach ulcer bleeding?",
    },
]


def run_evaluation():
    print("==================================================================")
    print("  ICU COPILOT — 20-QUESTION CLINICAL EVALUATION BENCH")
    print("==================================================================")

    # 1. Initialize retriever
    print("\n⏳ Initializing Knowledge Base Retriever…")
    t0 = time.time()
    primekg = load_primekg()
    name_idx = build_name_index(primekg)

    primekg_faiss_idx = None
    primekg_faiss_map = None
    if PRIMEKG_FAISS_INDEX.exists() and PRIMEKG_FAISS_MAP.exists():
        primekg_faiss_idx = faiss.read_index(str(PRIMEKG_FAISS_INDEX))
        with open(PRIMEKG_FAISS_MAP, "rb") as f:
            primekg_faiss_map = pickle.load(f)

    faiss_idx, chunks, embedder = index_guidelines()

    retriever = GlobalKBRetriever(
        primekg=primekg,
        name_index=name_idx,
        faiss_index=faiss_idx,
        guideline_chunks=chunks,
        embedder=embedder,
        primekg_faiss_index=primekg_faiss_idx,
        primekg_faiss_map=primekg_faiss_map,
    )
    print(f"✅ KB Retriever ready in {time.time() - t0:.2f}s")

    # 2. Load Gemma 4
    print("⏳ Loading Gemma 4 E4B on Apple Silicon Metal…")
    mgr = ModelManager(n_ctx=2048, n_batch=512, verbose=False)
    if not mgr.load() or not mgr.gemma4:
        print("❌ ERROR: Gemma 4 failed to load!")
        return

    print("✅ Gemma 4 loaded and ready.\n")

    results = []
    total_start = time.time()

    for idx, q in enumerate(QUESTIONS, start=1):
        print(f"\n[{idx}/20] Processing Q{q['id']} ({q['category']}): '{q['query']}'")
        retrieval_t0 = time.time()
        ctx = retriever.get_full_context(
            conditions=q["conditions"],
            drugs=q["drugs"],
            query=q["query"],
        )
        retrieval_time = time.time() - retrieval_t0

        graph_ctx = ctx["graph_context"]
        guideline_ctx = ctx["guideline_context"]

        prompt = (
            "You are an expert ICU Clinical AI Copilot. "
            "Answer the clinician's question based ONLY on the provided context. "
            "Be precise, concise, and clinically rigorous. "
            "Cite specific interaction alerts and guideline points.\n\n"
            f"<medical_knowledge>\n{graph_ctx}\n\n{guideline_ctx}\n</medical_knowledge>\n\n"
            f"Patient Profile: Conditions: {', '.join(q['conditions'])}. Medications: {', '.join(q['drugs'])}.\n"
            f"Question: {q['query']}"
        )

        gen_t0 = time.time()
        answer = mgr.gemma4_generate(prompt, max_tokens=220, temperature=0.2)
        gen_time = time.time() - gen_t0

        print(f"  ⚡ Retrieval: {retrieval_time:.2f}s | Generation: {gen_time:.2f}s")
        print(f"  📝 Answer preview: {answer.strip()[:140]}…")

        results.append({
            "id": q["id"],
            "category": q["category"],
            "conditions": q["conditions"],
            "drugs": q["drugs"],
            "query": q["query"],
            "retrieval_time_s": round(retrieval_time, 2),
            "generation_time_s": round(gen_time, 2),
            "graph_context": graph_ctx,
            "guideline_context": guideline_ctx,
            "answer": answer.strip(),
        })

    total_duration = time.time() - total_start
    print("\n==================================================================")
    print(f"🎉 COMPLETED ALL 20 QUESTIONS in {total_duration:.2f}s (Avg {total_duration/20:.2f}s/query)")
    print("==================================================================")

    out_file = PROJ_DIR / "data" / "eval_20_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"💾 Full results saved to {out_file}")


if __name__ == "__main__":
    run_evaluation()
