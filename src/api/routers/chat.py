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
    "You are an expert ICU Clinical AI Copilot. "
    "Answer the clinician's question based ONLY on the provided patient data and medical knowledge. "
    "Be concise, precise, and clinical. Cite specific data points. "
    "Never hallucinate or add information not present in the context."
)


def _build_prompt(state, patient: dict, kb_context: str, history: list[ChatMessage], message: str) -> list[dict]:
    """Construct the chat message list for llama-cpp-python create_chat_completion."""
    from knowledge_graph.subgraph_retriever import get_patient_subgraph_text

    G = state.graphs.get(patient["patient_id"])
    patient_ctx = get_patient_subgraph_text(G, patient["patient_id"]) if G else "(Graph not available)"

    system_content = (
        f"{SYSTEM_PROMPT}\n\n"
        f"<patient_data>\n{patient_ctx}\n</patient_data>\n\n"
        f"<medical_knowledge>\n{kb_context}\n</medical_knowledge>"
    )

    messages = [{"role": "system", "content": system_content}]

    # Include last 6 turns of history
    for msg in history[-6:]:
        messages.append({"role": msg.role, "content": msg.content})

    messages.append({"role": "user", "content": message})
    return messages


def _get_kb_context(state, patient: dict, message: str) -> str:
    """Retrieve PrimeKG + guideline context for this patient + query."""

    if state.kb_retriever is None:
        return "(Knowledge base not loaded — run scripts/build_global_kb.py)"

    conditions = [c["name"] for c in patient.get("conditions", [])]
    meds = [m["name"] for m in patient.get("medications", [])]

    # Extract entities from user query using MedGemma
    query_entities = []
    if state.model_manager and state.model_manager.medgemma:
        from llm_reasoning.reasoner import ENTITY_EXTRACTION_PROMPT
        prompt = ENTITY_EXTRACTION_PROMPT.format(clinical_text=message)
        try:
            import json, re
            raw = state.model_manager.medgemma_generate(prompt, max_tokens=150, temperature=0.1)
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            extracted = json.loads(match.group()) if match else json.loads(raw)
            for c in extracted.get("conditions", []): query_entities.append(c.get("name", ""))
            for m in extracted.get("medications", []): query_entities.append(m.get("name", ""))
        except Exception as e:
            logger.warning(f"Failed to extract entities from query: {e}")
            
    # Filter empty strings
    query_entities = [e for e in query_entities if e]
    all_conditions = conditions + query_entities

    try:
        ctx = state.kb_retriever.get_full_context(all_conditions, meds, message)
        kb_context = ctx["graph_context"] + "\n\n" + ctx["guideline_context"]
        print(f"\n=== KB CONTEXT FOR LLM ===\n{kb_context}\n==========================\n")
        return kb_context
    except Exception as e:
        logger.warning(f"KB retrieval failed: {e}")
        return "(KB retrieval error)"


async def _stream_llm(messages: list[dict], gemma4) -> AsyncIterator[str]:
    """
    Async generator that wraps the synchronous llama-cpp-python streaming call.
    Runs the blocking iteration in a thread pool to avoid blocking the event loop.

    The Future returned by run_in_executor is stored and awaited after the token
    loop so that any thread-level exception is surfaced rather than silently lost.
    Without awaiting, if the LLM thread crashes before placing the None sentinel,
    the generator would stall forever on queue.get().
    """
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue[str | None] = asyncio.Queue()

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
            # Always place sentinel so the consumer loop can exit
            loop.call_soon_threadsafe(queue.put_nowait, None)

    # Kick off the blocking LLM call; keep the Future so we can await it
    future = loop.run_in_executor(None, _run_sync)

    while True:
        token = await queue.get()
        if token is None:
            break
        yield token

    # Await the Future — surfaces any thread exception and ensures clean teardown
    await future


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

    # Build prompt with patient context + KB
    kb_context = _get_kb_context(state, patient, req.message)
    messages = _build_prompt(state, patient, kb_context, req.history, req.message)

    async def _stream_with_context():
        async for token in _stream_llm(messages, mgr.gemma4):
            yield token
            
        escaped_ctx = kb_context.replace("<", "&lt;").replace(">", "&gt;")
        yield f"\n\n<details class=\"reasoning-context\">\n<summary>Sources & Reasoning Context</summary>\n<pre><code>{escaped_ctx}</code></pre>\n</details>"

    return StreamingResponse(
        _stream_with_context(),
        media_type="text/plain",
        headers={"X-Patient-ID": req.patient_id},
    )
