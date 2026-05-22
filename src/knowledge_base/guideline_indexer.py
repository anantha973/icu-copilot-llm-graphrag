"""
Pipeline A — NIH Guideline Indexer
Downloads PDFs → extracts text → chunks → embeds → builds FAISS index.

Usage:
    python scripts/build_global_kb.py   (calls index_guidelines())
"""

from __future__ import annotations
import logging
import pickle
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

GUIDELINES_DIR = Path(__file__).parents[2] / "data" / "guidelines"
FAISS_INDEX_PATH = GUIDELINES_DIR / "faiss_index.pkl"
CHUNKS_PATH = GUIDELINES_DIR / "chunks.pkl"

# ── Public NIH / clinical guideline PDFs ─────────────────────────────────────
# These are freely available, public domain PDFs.
GUIDELINE_URLS: dict[str, str] = {
    "surviving_sepsis_2021.pdf": (
        "https://www.sccm.org/getattachment/Surviving-Sepsis-Campaign/"
        "Research-Trials/Surviving-sepsis-guidelines-2021/SSC-Guidelines-2021.pdf"
    ),
    "aki_kdigo_summary.pdf": (
        "https://kdigo.org/wp-content/uploads/2016/10/"
        "KDIGO-2012-AKI-Guideline-English.pdf"
    ),
    "ADA-diabetes-guidelines-2026.pdf": (
        "https://diabetesjournals.org/clinical/article-pdf/42/1/7/740348/diaclin_42_1_7.pdf"
    ),
    "jones-et-al-2025-2025-aha-acc-aanp-aapa-abc-accp-acpm-ags-ama-aspc-nma-pcna-sgim-guideline-for-the-prevention-detection.pdf": (
        "https://www.ahajournals.org/doi/pdf/10.1161/HYP.0000000000000065"
    ),
}

# If PDFs cannot be downloaded, we embed these plain-text clinical rules
# so the system always has guideline context.
FALLBACK_GUIDELINE_TEXT = """
=== Surviving Sepsis Campaign 2021 — Key Recommendations ===
1. MAP target: maintain ≥65 mmHg in septic shock.
2. Lactate: resuscitate to normalize lactate (target <2 mmol/L).
3. Blood cultures before antibiotics; administer antibiotics within 1 hour.
4. Norepinephrine: first-line vasopressor for septic shock.
5. Crystalloid 30 mL/kg IV within 3 hours for sepsis-induced hypoperfusion.
6. Procalcitonin: guide duration of antibiotic therapy.
7. Corticosteroids (hydrocortisone 200 mg/day): if norepinephrine dose >0.25 mcg/kg/min.

=== Acute Kidney Injury (KDIGO 2012) — Key Thresholds ===
Stage 1 AKI: Creatinine ×1.5–1.9 baseline OR urine output <0.5 mL/kg/h for 6–12h.
Stage 2 AKI: Creatinine ×2–2.9 baseline OR urine output <0.5 mL/kg/h for ≥12h.
Stage 3 AKI: Creatinine ×3 baseline OR ≥4 mg/dL OR RRT.

=== ICU Mechanical Ventilation — Lung Protective Strategy ===
Tidal volume: 6 mL/kg predicted body weight.
Plateau pressure: ≤30 cmH2O.
PEEP: titrate to FiO2 per ARDSnet table.
SpO2 target: 88–95% (permissive hypoxemia in ARDS).

=== Vasopressor Dosing Reference ===
Norepinephrine: 0.01–3 mcg/kg/min IV (first-line in septic shock).
Vasopressin: 0.03–0.04 units/min IV (adjunct to norepinephrine).
Epinephrine: 0.01–1 mcg/kg/min IV (adjunct or cardiac arrest).
Dopamine: 2–20 mcg/kg/min IV (inotropic effect at >5 mcg/kg/min).
Phenylephrine: 0.5–5 mcg/kg/min IV (avoid in septic shock with low CO).

=== Common ICU Electrolyte Thresholds ===
Potassium: 3.5–5.0 mEq/L (critical: <2.5 or >6.5).
Sodium: 136–145 mEq/L (critical: <120 or >160).
Magnesium: 1.7–2.2 mg/dL (supplement if <1.5 during arrhythmia risk).
Phosphate: 2.5–4.5 mg/dL (refeeding syndrome risk if <1.0).
"""


def _download_pdf(url: str, dest: Path) -> bool:
    """Download a PDF from url to dest. Returns True on success."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    }
    try:
        import requests
        with requests.get(url, stream=True, timeout=30, headers=headers) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return True
    except Exception as e:
        logger.warning(f"    ⚠️  Could not download {url}: {e}")
        return False


def _extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract plain text from a PDF using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(pdf_path))
        return "\n".join(page.get_text() for page in doc)
    except Exception as e:
        logger.warning(f"    ⚠️  PDF text extraction failed for {pdf_path}: {e}")
        return ""


def _chunk_text(text: str, chunk_size: int = 120, overlap: int = 20) -> list[str]:
    """
    Split text into overlapping word-count chunks.
    Default 120 words (~512 chars) to keep each chunk focused.
    Also splits on double-newlines (paragraph boundaries) first.
    """
    # Split into paragraphs first, then chunk each
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    all_chunks: list[str] = []
    for para in paragraphs:
        words = para.split()
        if len(words) <= chunk_size:
            all_chunks.append(para)
        else:
            step = max(1, chunk_size - overlap)
            for i in range(0, len(words), step):
                chunk = " ".join(words[i : i + chunk_size])
                if chunk.strip():
                    all_chunks.append(chunk.strip())
    return all_chunks


class _FastEmbedder:
    """Thin wrapper around fastembed so its API matches what retriever expects."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding
        self._model = TextEmbedding(model_name=model_name)
        # Store dim for FAISS
        sample = list(self._model.embed(["test"]))
        import numpy as np
        self.dim = np.array(sample[0]).shape[0]

    def encode(self, texts: list[str]) -> "np.ndarray":
        import numpy as np
        embeddings = list(self._model.embed(texts))
        return np.array(embeddings, dtype=np.float32)


def _make_embedder() -> "_FastEmbedder":
    """Return a cached fastembed embedder (lightweight, no torchvision dependency)."""
    return _FastEmbedder()


def index_guidelines(force_rebuild: bool = False) -> tuple:
    """
    Build FAISS index over NIH guideline texts.

    Returns:
        (faiss_index, chunks: list[str], embedder)
    """
    import numpy as np

    # Try loading from cache first
    if FAISS_INDEX_PATH.exists() and CHUNKS_PATH.exists() and not force_rebuild:
        logger.info("⚡ Loading guideline FAISS index from cache…")
        import faiss
        with open(FAISS_INDEX_PATH, "rb") as f:
            index = pickle.load(f)
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
        embedder = _make_embedder()
        logger.info(f"✅ FAISS index loaded: {len(chunks)} chunks.")
        return index, chunks, embedder

    GUIDELINES_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all text
    all_text = FALLBACK_GUIDELINE_TEXT  # always include embedded rules
    downloaded_any = False

    for filename, url in GUIDELINE_URLS.items():
        dest = GUIDELINES_DIR / filename
        if not dest.exists():
            logger.info(f"  ⬇️  Downloading {filename}…")
            ok = _download_pdf(url, dest)
        else:
            ok = True
            logger.info(f"  ✅ {filename} already present.")

        if ok and dest.exists():
            text = _extract_text_from_pdf(dest)
            if text:
                all_text += f"\n\n=== {filename} ===\n" + text
                downloaded_any = True

    if not downloaded_any:
        logger.warning("  ⚠️  No PDFs downloaded — using fallback embedded guidelines only.")

    # Chunk
    logger.info("✂️  Chunking guideline text…")
    chunks = _chunk_text(all_text, chunk_size=120, overlap=20)
    logger.info(f"   {len(chunks)} chunks created.")

    # Embed
    logger.info("🧠 Embedding chunks with fastembed all-MiniLM-L6-v2…")
    import faiss

    embedder = _make_embedder()
    embeddings = embedder.encode(chunks)

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    logger.info(f"✅ FAISS index built: {index.ntotal} vectors, dim={dim}.")

    # Cache
    with open(FAISS_INDEX_PATH, "wb") as f:
        pickle.dump(index, f)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    logger.info("💾 FAISS index cached.")

    return index, chunks, embedder
