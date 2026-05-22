"""
Layer 3 — LLM Model Loader
Loads MedGemma 4B and Gemma 4 E4B Uncensored via llama-cpp-python.
Both models are kept in memory simultaneously (~5GB Q4).
"""

from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parents[2] / "models"

# GGUF filenames — exact names as downloaded
MEDGEMMA_FILENAME   = "medgemma-4b-it-q4_k_m.gguf"
GEMMA4_E4B_FILENAME = "gemma-4b-uncensored-q4_k_m.gguf"
GEMMA4_E2B_FILENAME = "gemma4-e2b-q4_k_m.gguf"  # fallback


class ModelManager:
    """
    Manages both GGUF models. Loads at startup, provides simple inference API.
    Falls back gracefully if model files are not yet present.
    """

    def __init__(
        self,
        medgemma_path: Optional[str] = None,
        gemma4_path: Optional[str] = None,
        n_ctx: int = 2048,
        n_batch: int = 512,
        verbose: bool = False,
    ):
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.verbose = verbose

        self._medgemma_path = Path(medgemma_path) if medgemma_path else MODELS_DIR / MEDGEMMA_FILENAME
        self._gemma4_path   = Path(gemma4_path)   if gemma4_path   else MODELS_DIR / GEMMA4_E4B_FILENAME

        self.medgemma = None
        self.gemma4   = None
        self._using_fallback = False

    def load(self) -> bool:
        """Load both models. Returns True if at least one model loaded."""
        try:
            from llama_cpp import Llama
        except ImportError:
            logger.warning("llama-cpp-python not installed. Running in rule-only mode.")
            self._using_fallback = True
            return False

        loaded_any = False

        # Load MedGemma
        if self._medgemma_path.exists():
            logger.info(f"⏳ Loading MedGemma from {self._medgemma_path.name}…")
            self.medgemma = Llama(
                model_path=str(self._medgemma_path),
                n_ctx=self.n_ctx,
                n_batch=self.n_batch,
                n_gpu_layers=-1,   # full Metal offload on Apple Silicon
                verbose=self.verbose,
            )
            logger.info("✅ MedGemma 4B loaded.")
            loaded_any = True
        else:
            logger.warning(
                f"⚠️  MedGemma not found at {self._medgemma_path}. "
                "Place medgemma-4b-q4_k_m.gguf in models/ to enable summarization."
            )

        # Load Gemma 4 E4B
        if self._gemma4_path.exists():
            logger.info(f"⏳ Loading Gemma 4 E4B from {self._gemma4_path.name}…")
            self.gemma4 = Llama(
                model_path=str(self._gemma4_path),
                n_ctx=self.n_ctx,
                n_batch=self.n_batch,
                n_gpu_layers=-1,   # full Metal offload on Apple Silicon
                verbose=self.verbose,
            )
            logger.info("✅ Gemma 4 E4B Uncensored loaded.")
            loaded_any = True
        else:
            # Try E2B as fallback
            e2b_path = MODELS_DIR / GEMMA4_E2B_FILENAME
            if e2b_path.exists():
                logger.info(f"⏳ Loading Gemma 4 E2B (fallback) from {e2b_path.name}…")
                self.gemma4 = Llama(
                    model_path=str(e2b_path),
                    n_ctx=self.n_ctx,
                    n_batch=self.n_batch,
                    n_gpu_layers=-1,
                    verbose=self.verbose,
                )
                logger.info("✅ Gemma 4 E2B loaded (fallback mode).")
                loaded_any = True
            else:
                logger.warning(
                    f"⚠️  Gemma 4 not found. "
                    "Place gemma4-e4b-uncensored-q4_k_m.gguf in models/ to enable reasoning."
                )

        if not loaded_any:
            self._using_fallback = True
            logger.warning("⚠️  No GGUF models loaded. Running in rule-only mode (no LLM).")

        return loaded_any

    # ── Inference helpers ─────────────────────────────────────────────────────

    def medgemma_generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> str:
        """Run MedGemma inference. Returns raw text response."""
        if self.medgemma is None:
            return self._rule_only_summary(prompt)
        response = self.medgemma.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response["choices"][0]["message"]["content"]

    def gemma4_generate(
        self,
        prompt: str,
        max_tokens: int = 600,
        temperature: float = 0.4,
    ) -> str:
        """Run Gemma 4 E4B reasoning inference. Returns raw text response."""
        if self.gemma4 is None:
            return self._rule_only_reasoning(prompt)
        response = self.gemma4.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response["choices"][0]["message"]["content"]

    def gemma4_fn(self) -> callable:
        """Return a simple callable (str → str) for the severity classifier."""
        return lambda prompt: self.gemma4_generate(prompt)

    # ── Fallback responses (no LLM) ───────────────────────────────────────────

    @staticmethod
    def _rule_only_summary(prompt: str) -> str:
        return (
            "[LLM not loaded — rule-based summary only] "
            "Patient data has been processed. Install GGUF model files in models/ "
            "to enable AI-generated clinical summaries."
        )

    @staticmethod
    def _rule_only_reasoning(prompt: str) -> str:
        import json
        return json.dumps({
            "severity": "GREEN",
            "confidence": 0.5,
            "explanation": "LLM not loaded. Severity determined by rule engine only.",
            "evidence": [],
            "drug_alerts": [],
            "guideline_alerts": [],
        })

    @property
    def models_ready(self) -> bool:
        return self.medgemma is not None or self.gemma4 is not None
