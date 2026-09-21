"""
Chat router — POST /api/chat

Streams LLM responses word-by-word using FastAPI StreamingResponse.
Uses llama-cpp-python stream=True for real-time token streaming.

Request body:
    {
        "patient_id": "P001",
        "message": "Why is this patient critical?",
        "history": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }

Response: text/plain stream of tokens as they are generated.
If models are not loaded, returns a plain-text explanation immediately.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("icu.chat")
router = APIRouter(tags=["Chat"])


# ── Request schema ────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str   # "user" | "assistant"
    content: str

class ChatRequest(BaseModel):
    patient_id: str
    message: str
    history: list[ChatMessage] = Field(default_factory=list)


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert ICU Clinical Decision Support AI Copilot.\n"
    "CRITICAL CLINICAL COMMUNICATION RULES:\n"
    "- Never use conversational pleasantries or filler openings (do NOT say 'Yes', 'Certainly', 'Sure', 'Hello').\n"
    "- Provide clear, concise, highly structured clinical answers that an intensivist can scan in 5 seconds.\n"
    "- ALWAYS use structured sections with bullet points and bold lead phrases.\n"
    "- Never write long, dense paragraphs or run-on sentences.\n"
    "- Ground all reasoning strictly in the provided patient data and clinical evidence."
)


def _build_prompt(
    state,
    patient: dict,
    kb_context: str,
    safety_verdict: str,
    safety_rationale: str,
    history: list[ChatMessage],
    message: str,
) -> list[dict]:
    """Construct the chat message list for llama-cpp-python create_chat_completion."""
    from knowledge_graph.subgraph_retriever import get_patient_subgraph_text

    G = state.graphs.get(patient["patient_id"])
    patient_ctx = get_patient_subgraph_text(G, patient["patient_id"]) if G else "(Graph not available)"

    safety_block = ""
    if safety_verdict:
        safety_block = (
            f"\n<clinical_safety_interlock>\n"
            f"SAFETY INTERLOCK STATUS: [{safety_verdict}]\n"
            f"SAFETY BASIS: {safety_rationale}\n"
            f"MANDATORY INSTRUCTION: A safety status banner for [{safety_verdict}] is ALREADY displayed to the user.\n"
            f"Do NOT output conversational pleasantries. Do NOT repeat 'CLINICAL VERDICT: ...'.\n"
            f"Structure your response strictly into these three concise sections with clear bullet points:\n\n"
            f"**Clinical Assessment**:\n"
            f"(One clear, direct sentence stating the clinical decision and primary reason)\n\n"
            f"**Key Mechanisms & Risks**:\n"
            f"• **[Primary Risk / Mechanism]**: (1 concise sentence explaining pharmacological or organ risk)\n"
            f"• **[Patient Vitals / Labs]**: (1 concise sentence citing relevant patient data points)\n\n"
            f"**Recommended Bedside Action**:\n"
            f"• **[Immediate Action]**: (1 actionable step, e.g. withhold drug, order ABG/ECG)\n"
            f"• **[Safe Alternative / Protocol]**: (1 clear alternative drug or monitoring protocol)\n"
            f"</clinical_safety_interlock>\n"
        )
    else:
        safety_block = (
            f"\n<clinical_response_instruction>\n"
            f"Format your response clearly and concisely for an ICU clinician:\n"
            f"- Structure your answer with clear section headings (**Clinical Assessment**, **Key Findings**, **Recommended Bedside Action**).\n"
            f"- Use concise bullet points with bold lead phrases (e.g. • **Finding**: explanation).\n"
            f"- Cite specific patient vitals, lab values, and scores.\n"
            f"- Do NOT write long or dense paragraphs. Keep each point to 1-2 direct sentences.\n"
            f"</clinical_response_instruction>\n"
        )

    system_content = (
        f"{SYSTEM_PROMPT}\n\n"
        f"<patient_data>\n{patient_ctx}\n</patient_data>\n"
        f"{safety_block}\n"
        f"<medical_knowledge>\n{kb_context}\n</medical_knowledge>"
    )

    messages = [{"role": "system", "content": system_content}]

    # Include last 6 turns of history
    for msg in history[-6:]:
        messages.append({"role": msg.role, "content": msg.content})

    messages.append({"role": "user", "content": message})
    return messages


def _get_kb_context(state, patient: dict, message: str) -> tuple[str, str, str, list[str]]:
    """Retrieve PrimeKG + guideline context and deterministic safety verdict for this patient + query."""

    if state.kb_retriever is None:
        return "(Knowledge base not loaded — run scripts/build_global_kb.py)", "", "", []

    conditions = [c["name"] for c in patient.get("conditions", [])]
    meds = [m["name"] for m in patient.get("medications", [])]

    # Extract entities from user query using MedGemma
    query_conditions = []
    query_meds = []
    if state.model_manager and state.model_manager.medgemma:
        from llm_reasoning.reasoner import ENTITY_EXTRACTION_PROMPT
        prompt = ENTITY_EXTRACTION_PROMPT.format(clinical_text=message)
        try:
            import json, re
            raw = state.model_manager.medgemma_generate(prompt, max_tokens=150, temperature=0.1)
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            extracted = json.loads(match.group()) if match else json.loads(raw)
            for c in extracted.get("conditions", []):
                if c.get("name"):
                    query_conditions.append(c["name"])
            for m in extracted.get("medications", []):
                if m.get("name"):
                    query_meds.append(m["name"])
        except Exception as e:
            logger.warning(f"Failed to extract entities from query: {e}")

    # Fallback dictionary matching for medications if LLM missed it
    if not query_meds and state.kb_retriever and state.kb_retriever.name_index:
        import re
        for term, nid_list in state.kb_retriever.name_index.items():
            if len(term) >= 4 and re.search(r"\b" + re.escape(term) + r"\b", message, re.IGNORECASE):
                if state.kb_retriever.primekg and any(state.kb_retriever.primekg.nodes[nid].get("node_type") == "drug" for nid in nid_list if nid in state.kb_retriever.primekg):
                    query_meds.append(term)

    # Combine patient records with query-mentioned entities
    all_conditions = list(dict.fromkeys([c for c in (conditions + query_conditions) if c]))
    all_meds = list(dict.fromkeys([m for m in (meds + query_meds) if m]))

    try:
        ctx = state.kb_retriever.get_full_context(all_conditions, all_meds, message, candidate_drugs=query_meds)
        kb_context = ctx["graph_context"] + "\n\n" + ctx["guideline_context"]
        safety_verdict = ctx.get("safety_verdict", "")
        safety_rationale = ctx.get("safety_rationale", "")
        return kb_context, safety_verdict, safety_rationale, query_meds
    except Exception as e:
        logger.warning(f"KB retrieval failed: {e}")
        return "(KB retrieval error)", "", "", query_meds


_llm_lock = asyncio.Lock()


async def _stream_llm(messages: list[dict], gemma4) -> AsyncIterator[str]:
    """
    Async generator that wraps the synchronous llama-cpp-python streaming call.
    Runs the blocking iteration in a thread pool to avoid blocking the event loop.
    Uses an asyncio.Lock to serialize concurrent requests, preventing KV cache buffer collision.
    Applies zero-tolerance output guardrails to eliminate any sycophantic opening tokens.
    """
    import re
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    async with _llm_lock:
        def _run_sync():
            try:
                for chunk in gemma4.create_chat_completion(
                    messages=messages,
                    max_tokens=600,
                    temperature=0.4,
                    stream=True,
                ):
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        loop.call_soon_threadsafe(queue.put_nowait, delta)
            except Exception as e:
                logger.error(f"LLM streaming error: {e}")
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        future = loop.run_in_executor(None, _run_sync)

        buffer = ""
        initial_check_done = False

        while True:
            token = await queue.get()
            if token is None:
                if buffer:
                    yield buffer
                break

            if not initial_check_done:
                buffer += token
                # When buffer contains enough characters or newline/colon, sanitize leading sycophancy
                if len(buffer) >= 20 or "\n" in buffer or ":" in buffer:
                    cleaned = re.sub(
                        r"^\s*(Yes|Certainly|Sure|Of course|Hello)[,\.!\s]*",
                        "",
                        buffer,
                        flags=re.IGNORECASE,
                    )
                    # Strip redundant repeated clinical verdict if model outputs it
                    cleaned = re.sub(
                        r"^\s*(?:1\.\s*)?CLINICAL VERDICT\s*:\s*\[?[A-Z\s/]+\]?\s*\n*",
                        "",
                        cleaned,
                        flags=re.IGNORECASE,
                    )
                    initial_check_done = True
                    buffer = ""
                    if cleaned:
                        yield cleaned
            else:
                yield token

        await future


# ── Conversational & Intent Classification ──────────────────────────────────────

GREETING_PATTERNS = {
    "hey", "hi", "hello", "good morning", "good afternoon", "good evening",
    "greetings", "yo", "help", "who are you", "what can you do", "test", "start"
}

PHATIC_PATTERNS = {
    "good", "ok", "okay", "k", "great", "cool", "nice", "fine", "alright", "all right",
    "thanks", "thank you", "thx", "ty", "got it", "understood", "sounds good",
    "perfect", "sure", "yep", "yeah", "yes", "nope", "no", "bye", "goodbye"
}

def is_phatic_remark(message: str) -> bool:
    clean = message.lower().strip(" !.,?':;\n\t")
    if clean in PHATIC_PATTERNS:
        return True
    words = clean.split()
    if len(words) <= 3 and any(w in PHATIC_PATTERNS for w in words):
        if not any(med in clean for med in ["take", "give", "dose", "drug", "med", "why", "what", "can"]):
            return True
    return False

def is_conversational_greeting(message: str) -> bool:
    clean = message.lower().strip(" !.,?':;\n\t")
    if clean in GREETING_PATTERNS:
        return True
    words = clean.split()
    if len(words) <= 2 and words[0] in {"hi", "hey", "hello"}:
        return True
    return False

def is_medication_or_safety_query(message: str, query_meds: list[str]) -> bool:
    if query_meds:
        return True
    q_lower = message.lower()
    med_keywords = [
        "can", "take", "give", "prescribe", "safe", "safety", "interact",
        "contraindicat", "adverse", "risk", "dose", "drug", "medication",
        "medicine", "compatibility", "administer", "co-administer", "review",
        "chart", "alert", "warning", "receive", "start", "stop"
    ]
    return any(kw in q_lower for kw in med_keywords)


# ── POST /api/chat ────────────────────────────────────────────────────────────

@router.post("/chat")
async def chat(req: ChatRequest, request: Request):
    """
    Stream AI clinical explanation for a patient query.
    Returns text/plain with tokens as they are generated.
    """
    state = request.app.state.icu_state

    # Find patient
    patient = next((p for p in state.patients if p["patient_id"] == req.patient_id), None)
    if patient is None:
        async def _err():
            yield f"Patient {req.patient_id} not found."
        return StreamingResponse(_err(), media_type="text/plain")

    p_name = patient.get("demographics", {}).get("name", req.patient_id)
    bed = patient.get("demographics", {}).get("bed", "1")

    # 1. Fast path: Conversational greetings
    if is_conversational_greeting(req.message):
        sev = state.severity_map.get(req.patient_id, {})
        sev_label = sev.get("severity", "STABLE")
        conditions = ", ".join(c["name"] for c in patient.get("conditions", [])) or "None recorded"
        meds = ", ".join(m["name"] for m in patient.get("medications", [])) or "None recorded"

        async def _greet():
            yield (
                f"Hello Doctor. ICU Copilot is ready for **{p_name}** (Bed {bed}).\n\n"
                f"• **Current Acuity**: **{sev_label}**\n"
                f"• **Primary Conditions**: {conditions}\n"
                f"• **Active Medications**: {meds}\n\n"
                f"How can I assist you? You can ask:\n"
                f"1. *\"Can this patient take [Drug]?\"* (Real-time safety & contraindication check)\n"
                f"2. *\"Why is this patient in {sev_label} status?\"* (Acuity & rule breakdown)\n"
                f"3. *\"What are the key lab abnormalities and monitoring priorities?\"*"
            )
        return StreamingResponse(
            _greet(),
            media_type="text/plain",
            headers={"X-Patient-ID": req.patient_id, "X-Safety-Verdict": "GREETING"},
        )

    # 2. Fast path: Phatic remarks ("good", "ok", "thanks", "sounds good")
    if is_phatic_remark(req.message):
        async def _ack():
            yield f"Understood, Doctor. Standing by for any clinical orders, risk checks, or inquiries for **{p_name}** (Bed {bed})."
        return StreamingResponse(
            _ack(),
            media_type="text/plain",
            headers={"X-Patient-ID": req.patient_id, "X-Safety-Verdict": "CONVERSATIONAL"},
        )

    # Models not loaded — return instant fallback
    mgr = state.model_manager
    if mgr is None or not state.models_loaded or mgr.gemma4 is None:
        fallback_lines = [
            "⚠️ LLM models are not loaded.\n\n",
            "Place GGUF files in models/ to enable AI explanations:\n",
            "• medgemma-4b-it-q4_k_m.gguf\n",
            "• gemma-4b-uncensored-q4_k_m.gguf\n\n",
            "Rule-based severity classification is active and accurate.",
        ]

        # Even for fallback: show severity context inline
        sev = state.severity_map.get(req.patient_id, {})
        if sev:
            sev_label = sev["severity"]
            triggers = ", ".join(sev["triggered_rules"]) or "none"
            fallback_lines.insert(0, f"**{sev_label}** — Rule triggers: {triggers}\n\n")

        async def _fallback():
            for line in fallback_lines:
                yield line
        return StreamingResponse(_fallback(), media_type="text/plain")

    # Retrieve Knowledge Base context
    kb_context, raw_verdict, safety_rationale, query_meds = _get_kb_context(state, patient, req.message)

    # Determine if safety interlock is contextually relevant to this query
    is_med_query = is_medication_or_safety_query(req.message, query_meds)
    enforce_safety = False
    active_verdict = ""

    # Only enforce safety interlock when the clinician is asking about medications, safety, or contraindications
    if is_med_query:
        if raw_verdict in ("CONTRAINDICATED", "CAUTION / MONITORING REQUIRED", "PERMITTED / INDICATED"):
            enforce_safety = True
            active_verdict = raw_verdict

    # Build prompt
    messages = _build_prompt(
        state,
        patient,
        kb_context,
        active_verdict if enforce_safety else "",
        safety_rationale if enforce_safety else "",
        req.history,
        req.message,
    )

    async def _stream_with_context():
        if enforce_safety:
            if active_verdict == "CONTRAINDICATED":
                yield f"🚨 **CRITICAL SAFETY INTERLOCK: [CONTRAINDICATED]**\n*{safety_rationale}*\n\n"
            elif active_verdict == "CAUTION / MONITORING REQUIRED":
                yield f"⚠️ **SAFETY NOTICE: [CAUTION / MONITORING REQUIRED]**\n*{safety_rationale}*\n\n"
            elif active_verdict == "PERMITTED / INDICATED":
                yield f"✅ **SAFETY CHECK: [PERMITTED / INDICATED]**\n*{safety_rationale}*\n\n"

        async for token in _stream_llm(messages, mgr.gemma4):
            yield token

        # Collapsible Sources & Citations Toggle (PrimeKG & Clinical Guidelines)
        if kb_context and not kb_context.startswith("(") and len(kb_context.strip()) > 20:
            escaped_ctx = kb_context.replace("<", "&lt;").replace(">", "&gt;").strip()
            yield (
                f"\n\n<details class=\"reasoning-context\">\n"
                f"<summary>Sources & Reasoning Context (PrimeKG & Guidelines)</summary>\n"
                f"<pre><code>{escaped_ctx}</code></pre>\n"
                f"</details>"
            )

    return StreamingResponse(
        _stream_with_context(),
        media_type="text/plain",
        headers={
            "X-Patient-ID": req.patient_id,
            "X-Safety-Verdict": active_verdict or "NONE",
        },
    )
