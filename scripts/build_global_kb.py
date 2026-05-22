"""
scripts/build_global_kb.py
One-time setup script: download PrimeKG, build NetworkX graph, build FAISS index.
Run once before starting the dashboard.

Usage:
    python scripts/build_global_kb.py [--force]
"""

import sys
import logging
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build ICU Copilot global knowledge base")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if cached")
    parser.add_argument("--skip-primekg", action="store_true", help="Skip PrimeKG download/build")
    parser.add_argument("--skip-guidelines", action="store_true", help="Skip guideline indexing")
    args = parser.parse_args()

    print("\n🏥 ICU Clinical Copilot — Global KB Builder")
    print("=" * 50)

    # ── Step 1: PrimeKG ───────────────────────────────────────────────────────
    if not args.skip_primekg:
        print("\n📦 STEP 1: PrimeKG Medical Knowledge Graph")
        from knowledge_base.primekg_loader import download_primekg, load_primekg, build_name_index, build_primekg_faiss
        from knowledge_base.guideline_indexer import _make_embedder

        print("  Downloading PrimeKG CSVs (may take a few minutes on first run)…")
        try:
            download_primekg(force=args.force)
            G = load_primekg(rebuild=args.force)
            name_idx = build_name_index(G)
            print(f"  ✅ PrimeKG ready: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
            print(f"  ✅ Name index: {len(name_idx):,} unique entity names")
            
            print("  Building PrimeKG FAISS Vector Index…")
            embedder = _make_embedder()
            build_primekg_faiss(G, embedder, force_rebuild=args.force)
        except FileNotFoundError as e:
            print(f"  ⚠️  {e}")
            print("  → Continuing without PrimeKG (fallback mode active)")
    else:
        print("\n⏭️  STEP 1: PrimeKG (skipped)")

    # ── Step 2: NIH Guidelines ────────────────────────────────────────────────
    if not args.skip_guidelines:
        print("\n📚 STEP 2: NIH Guideline FAISS Index")
        from knowledge_base.guideline_indexer import index_guidelines
        index, chunks, _ = index_guidelines(force_rebuild=args.force)
        print(f"  ✅ FAISS index ready: {len(chunks)} guideline chunks indexed")
    else:
        print("\n⏭️  STEP 2: Guidelines (skipped)")

    # ── Step 3: Generate patients ─────────────────────────────────────────────
    print("\n🧑‍⚕️  STEP 3: Synthetic Patient Generation")
    from patient_simulation.patient_generator import generate_patients, PATIENTS_DIR
    if PATIENTS_DIR.exists() and any(PATIENTS_DIR.iterdir()) and not args.force:
        print(f"  ✅ Patients already generated in {PATIENTS_DIR} (use --force to regenerate)")
    else:
        patients = generate_patients(n_patients=20)
        print(f"  ✅ {len(patients)} patients generated")

    print("\n✅ Global KB build complete. Ready to run the dashboard.")
    print("   Next: streamlit run src/dashboard/app.py\n")


if __name__ == "__main__":
    main()
