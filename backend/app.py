"""Context Management RAG starter — extended chat backend.

This is the Foundations chat backend with stubs for retrieval-augmented generation.

TODO:
  1. Update SYSTEM_PROMPT with citation rules.
  2. In /api/chat: retrieve top-K chunks for the user's question.
  3. Format chunks as a numbered context block.
  4. Build the user_content with CONTEXT + QUESTION.
  5. Parse citation numbers from the answer; return them to the frontend.
"""
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

# Make the parent directory importable so we can use indexer.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from anthropic import Anthropic
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from indexer import load_index, search

load_dotenv()  # ANTHROPIC_API_KEY from .env

# Logging on by default at INFO so the retrieval trace (indexer's rag.retrieval
# logger) and the per-request gate + citation decisions below are visible while
# debugging answer quality. Quiet it with RAG_LOG_LEVEL=WARNING.
logging.basicConfig(
    level=os.environ.get("RAG_LOG_LEVEL", "INFO").upper(),
    format="%(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("rag.chat")

app = Flask(__name__)
CORS(app)
client = Anthropic()

# Load the index once at startup. Fails fast if no index — run `python indexer.py` first.
INDEX = load_index()
log.info("loaded %d chunks from index", len(INDEX))


# ════════════════════════════════════════════════════════════════
# TODO — update SYSTEM_PROMPT with citation rules.
# Suggestions:
#   - Answer ONLY from the provided context.
#   - Cite each factual claim with [n] using the numbers in the context.
#   - If the context doesn't contain the answer, say so explicitly.
# ════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the \
sources provided in the CONTEXT block of each message.

Rules:
- Base every statement strictly on the provided context. Do not use outside knowledge, \
and do not guess or infer beyond what the sources state.
- Cite each factual claim with a bracketed source number, e.g. [1] or [2][3], using the \
numbers shown in the context. Place the citation immediately after the claim it supports.
- Only use citation numbers that appear in the context. Never invent a number.
- If the context does not contain enough information to answer, say so explicitly \
(e.g. "The provided sources don't contain an answer to that.") and do not fabricate one.
- If only part of the question is supported, answer that part and clearly state what the \
sources do not cover.
- Lead with the direct answer and keep it concise: no preamble, no restating the question, \
and no closing summary. Be as brief as the question allows while still citing every claim with [n].

After the answer, append a machine-readable support block in EXACTLY this form, with nothing after it:
<<<CITATIONS>>>
{"1": "<span copied verbatim from source [1]>", "2": "<span copied verbatim from source [2]>"}
Include one entry for every [n] you cited. Each value must be copied EXACTLY, character-for-character, \
from that numbered source and must support the claim you cited it for — do not paraphrase, summarize, or \
invent it. This block is stripped out before your answer is shown to the user."""


# Relevance gate. Calibrated on the Gemini index (in-domain top scores ~0.73–0.79,
# out-of-domain ~0.49–0.61): if the best dense-cosine match is below MIN_TOP_SCORE
# we treat the question as out-of-scope and abstain WITHOUT calling the LLM — a
# cheap, hard guarantee against answering when nothing relevant was retrieved
# (also screens off-topic and injected prompts). Re-calibrate per embedding
# backend; env-tunable.
TOP_K = int(os.environ.get("RETRIEVAL_K", "8"))
MIN_TOP_SCORE = float(os.environ.get("RETRIEVAL_MIN_SCORE", "0.66"))
CHAT_MODEL = os.environ.get("CHAT_MODEL", "claude-sonnet-4-6")
# Rough $/answer estimate surfaced in meta. Sonnet 4.6 list price is $3 / $15 per
# MTok (input / output); override per model via env. Embedding cost is separate
# (folded into retrieval_ms as latency only), so this is the LLM inference cost.
COST_PER_INPUT_TOKEN = float(os.environ.get("COST_PER_INPUT_TOKEN", "3e-6"))
COST_PER_OUTPUT_TOKEN = float(os.environ.get("COST_PER_OUTPUT_TOKEN", "15e-6"))
# Grounding verification: the model appends a per-[n] verbatim quote block; a
# citation is kept only if its quote is a substring of the cited source. A quote
# shorter than this (after whitespace/case normalization) is treated as no real
# support, so a trivial 1-word "quote" can't pass the check. Env-tunable.
CITATION_BLOCK_MARKER = "<<<CITATIONS>>>"
MIN_QUOTE_CHARS = int(os.environ.get("MIN_QUOTE_CHARS", "12"))
ABSTAIN_REPLY = (
    "The provided 14 CFR sources don't contain anything relevant to that question, "
    "so I can't answer it from the corpus."
)


@app.route("/api/chat", methods=["POST"])
def chat():
    user_message = request.json["message"]
    log.info("chat q=%r", user_message)
    # Stream the answer token-by-token when the client asks for it (the web UI
    # sends `Accept: text/event-stream`); fall back to a single JSON body for
    # programmatic callers (e.g. test_api.py). Both paths run the identical
    # retrieval → gate → verify pipeline — only answer delivery differs.
    if "text/event-stream" in request.headers.get("Accept", ""):
        return Response(
            _chat_stream(user_message),
            mimetype="text/event-stream",
            # Defeat any response buffering between here and the browser so deltas
            # arrive as they're produced (no-cache for proxies, X-Accel for nginx).
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return jsonify(_chat_once(user_message))


def _retrieve(user_message):
    """Retrieve top-K chunks; return (hits, retrieval_ms, t_start).

    indexer's rag.retrieval logger traces which channels ran (dense/BM25/router/
    rerank) and the exact chunks. t_start is captured just before retrieval so a
    later total_ms spans the whole request (retrieval round-trip + LLM + verify),
    matching what the non-streaming path reported before.
    """
    t_start = time.perf_counter()
    hits = search(user_message, INDEX, k=TOP_K)
    retrieval_ms = (time.perf_counter() - t_start) * 1000  # incl. query-embedding round-trip
    return hits, retrieval_ms, t_start


def _gate_ok(hits):
    """Relevance gate: True iff the best dense-cosine match clears the bar.

    Gate on dense_score (raw cosine), not the hybrid `score`, which is rank-based
    and not comparable to the calibrated cosine threshold. Below the bar we abstain
    up front — no LLM call, no chance to hallucinate an answer the corpus can't
    support. Returns (ok, top_dense) so the caller can log the score it decided on.
    """
    top_dense = max((h["dense_score"] for h in hits), default=0.0)
    return bool(hits) and top_dense >= MIN_TOP_SCORE, top_dense


def _build_user_content(user_message, hits):
    """Numbered context block + question, so the model can ground and cite [n]."""
    context = "\n\n".join(f"[{i + 1}] {h['text']}" for i, h in enumerate(hits))
    return f"CONTEXT:\n{context}\n\nQUESTION:\n{user_message}"


def _message_text(msg) -> str:
    """Concatenate the text blocks of an Anthropic message (skip any non-text)."""
    return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")


def _abstain_payload(user_message, retrieval_ms) -> dict:
    """The out-of-scope refusal — identical whether sent as JSON or an SSE `done`."""
    return {
        "reply": ABSTAIN_REPLY,
        "citations": [],
        "abstained": True,
        "meta": {
            "question": user_message,
            "retrieval_ms": round(retrieval_ms),
            "total_ms": round(retrieval_ms),
        },
    }


def _sse(event: dict) -> str:
    """Serialize one event as an SSE `data:` frame (double-newline terminated)."""
    return f"data: {json.dumps(event)}\n\n"


def _emittable_prefix(raw: str, marker: str) -> int:
    """Length of `raw` that is safe to stream without leaking the citation marker.

    Everything before a fully-present marker is safe (the tail is the machine-only
    support block, never shown to the user). Before the marker appears we withhold
    the longest suffix of `raw` that is a prefix of `marker`, so a marker split
    across deltas is never partially shown; that held-back tail is delivered in the
    final `done` event's authoritative `reply` instead.
    """
    idx = raw.find(marker)
    if idx != -1:
        return idx
    for k in range(min(len(marker) - 1, len(raw)), 0, -1):
        if raw.endswith(marker[:k]):
            return len(raw) - k
    return len(raw)


def _finalize(raw, usage, hits, user_message, retrieval_ms, llm_ms, t_start) -> dict:
    """Verify citations against the model's supporting quotes and assemble the
    final answer payload (reply, citations, grounding flags, meta).

    Shared by the streaming and non-streaming paths — the only thing that differs
    upstream is how `raw` (the model's full output) and `usage` were obtained.

    The model appends a machine-readable support block (see SYSTEM_PROMPT); split
    it off so `answer` is the display text and `quotes` maps each cited [n] to the
    verbatim span the model claims supports it. `quote_block` is False when no
    parseable block was emitted — verification then degrades to in-range-only.
    """
    answer, quotes, quote_block = _split_citation_block(raw)

    # Token usage + latency (total = retrieval, incl. embedding round-trip, + LLM).
    total_ms = (time.perf_counter() - t_start) * 1000
    cost_usd = round(
        usage.input_tokens * COST_PER_INPUT_TOKEN
        + usage.output_tokens * COST_PER_OUTPUT_TOKEN,
        6,
    )
    log.info(
        "llm model=%s in=%d out=%d cost=$%.4f | latency retrieval=%.0fms llm=%.0fms total=%.0fms",
        CHAT_MODEL, usage.input_tokens, usage.output_tokens, cost_usd,
        retrieval_ms, llm_ms, total_ms,
    )

    # ────────────────────────────────────────────────────────────
    # Citation extraction + grounding verification
    #
    # A marker [n] is kept only if it is in range AND (when the model emitted a
    # support block) the quote it gave for [n] is a verbatim substring of source
    # [n]. Kept markers carry their verified quote; every dropped marker is
    # neutralized to [?] in the returned reply. This is the deterministic
    # supporting-quote hard gate — NLI/entailment soft-gating is still roadmap
    # (see docs/ARCHITECTURE.md §1.2).
    # ────────────────────────────────────────────────────────────
    used = [int(x) for x in re.findall(r"\[(\d+)\]", answer)]
    in_range = {n for n in used if 1 <= n <= len(hits)}
    if quote_block:
        supported = {n for n in in_range if _quote_supported(quotes.get(n, ""), hits[n - 1]["text"])}
    else:
        supported = in_range  # no quotes to check against — keep the in-range guard only

    citations = _build_citations(answer, hits, supported, quotes)
    invalid = sorted(n for n in used if not (1 <= n <= len(hits)))   # out-of-range / invented
    unsupported = sorted(n for n in in_range if n not in supported)  # quote failed the substring check
    bad = set(invalid) | set(unsupported)

    meta = {
        # Echo the exact question this answer was produced for. A downstream
        # eval/grader can assert row["question"] == meta["question"] to catch a
        # mis-zip (the question column drifting out of sync with the answer it's
        # graded against) instead of silently scoring the wrong pair.
        "question": user_message,
        "model": CHAT_MODEL,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cost_usd": cost_usd,
        "retrieval_ms": round(retrieval_ms),
        "llm_ms": round(llm_ms),
        "total_ms": round(total_ms),
        # True when each kept [n] was checked against a verbatim quote from the
        # cited passage (not merely resolved to a retrieved chunk) — lets the UI
        # say "verified against the cited passage" honestly.
        "citations_verified": quote_block,
    }
    # Prompt caching is off today (single-turn, sub-floor system prompt), so these
    # are 0; read them defensively so cost_usd stays honest if caching is enabled.
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    if cache_read:
        meta["cached_input_tokens"] = cache_read

    # Trace which chunks the answer cited and which markers were dropped, so a
    # wrong answer can be traced to what retrieval + verification actually kept.
    if citations:
        log.info(
            "answer cited %d of %d retrieved chunks (verified=%s):",
            len(citations), len(hits), quote_block,
        )
        for c in citations:
            h = hits[c["n"] - 1]
            snippet = " ".join(h.get("text", "").split())[:140]
            log.info(
                "  [%d] %s src=%s chunk=%s :: %s",
                c["n"],
                h.get("cfr_citation") or h.get("section") or "?",
                h.get("source"), h.get("chunk_index"), snippet,
            )
    else:
        # A non-abstained answer with no kept citation is ungrounded: the gate let
        # it through on retrieval score, but nothing the model cited survived
        # verification. Surface it (grounded=False, UI notice) rather than
        # returning it as if it were a normal cited answer.
        log.warning("non-abstained answer carries no verified citations — ungrounded")
    if invalid:
        log.warning("answer used out-of-range citation(s): %s (neutralized to [?])", invalid)
    if unsupported:
        log.warning(
            "answer citation(s) failed the supporting-quote check: %s (neutralized to [?])",
            unsupported,
        )

    # Neutralize every dropped [n] (out-of-range or unsupported) in the returned
    # answer so a consumer of `reply` can't mistake it for a real, grounded
    # citation. Kept markers are left untouched; dropped ones become a literal
    # [?], which the frontend renders as a flagged badge.
    safe_answer = answer
    if bad:
        safe_answer = re.sub(
            r"\[(\d+)\]",
            lambda m: m.group(0) if int(m.group(1)) in supported else "[?]",
            answer,
        )

    return {
        "reply": safe_answer,
        "citations": citations,
        "grounded": bool(citations),
        "invalid_citations": invalid,
        "unsupported_citations": unsupported,
        "meta": meta,
    }


def _chat_once(user_message) -> dict:
    """Non-streaming path: retrieve, gate, one blocking LLM call, verify, return
    the full JSON payload in one shot."""
    hits, retrieval_ms, t_start = _retrieve(user_message)
    ok, top_dense = _gate_ok(hits)
    if not ok:
        log.info(
            "gate top_dense=%.3f < %.2f -> ABSTAIN (no LLM call) | retrieval=%.0fms in=0 out=0",
            top_dense, MIN_TOP_SCORE, retrieval_ms,
        )
        return _abstain_payload(user_message, retrieval_ms)
    log.info(
        "gate top_dense=%.3f >= %.2f -> ANSWER (%d chunks in context)",
        top_dense, MIN_TOP_SCORE, len(hits),
    )

    # Single-turn: only this question + the retrieved context is sent — no prior
    # conversation is threaded in — so input_tokens ≈ system + context + question.
    t_llm = time.perf_counter()
    resp = client.messages.create(
        model=CHAT_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_content(user_message, hits)}],
    )
    llm_ms = (time.perf_counter() - t_llm) * 1000
    return _finalize(_message_text(resp), resp.usage, hits, user_message, retrieval_ms, llm_ms, t_start)


def _chat_stream(user_message):
    """Streaming path: same pipeline, but emit answer text as SSE `delta` events as
    the model produces it, then a final `done` event with the verified citations
    and meta (which can only be computed once the whole answer is in hand).

    Live text is provisional: the machine-only <<<CITATIONS>>> block is withheld,
    and citation verification / [?] neutralization run on the complete text in
    `_finalize`, so the `done` event carries the authoritative `reply` the client
    should settle on (it may differ from what streamed only where a [n] was dropped
    to [?]). Yielding `error` is the only way to signal a mid-flight failure, since
    the 200 response is already committed once the first byte is sent.
    """
    try:
        hits, retrieval_ms, t_start = _retrieve(user_message)
        ok, top_dense = _gate_ok(hits)
        if not ok:
            log.info(
                "gate top_dense=%.3f < %.2f -> ABSTAIN (no LLM call) | retrieval=%.0fms in=0 out=0",
                top_dense, MIN_TOP_SCORE, retrieval_ms,
            )
            yield _sse({"type": "done", **_abstain_payload(user_message, retrieval_ms)})
            return
        log.info(
            "gate top_dense=%.3f >= %.2f -> ANSWER (%d chunks in context)",
            top_dense, MIN_TOP_SCORE, len(hits),
        )

        t_llm = time.perf_counter()
        raw = ""
        emitted = 0
        with client.messages.stream(
            model=CHAT_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_user_content(user_message, hits)}],
        ) as stream:
            for delta in stream.text_stream:
                raw += delta
                # Emit only what can't be part of the trailing <<<CITATIONS>>>
                # block, holding back a possible partial marker until it resolves.
                safe = _emittable_prefix(raw, CITATION_BLOCK_MARKER)
                if safe > emitted:
                    yield _sse({"type": "delta", "text": raw[emitted:safe]})
                    emitted = safe
            final = stream.get_final_message()
        llm_ms = (time.perf_counter() - t_llm) * 1000

        payload = _finalize(
            _message_text(final), final.usage, hits, user_message, retrieval_ms, llm_ms, t_start,
        )
        yield _sse({"type": "done", **payload})
    except Exception:
        log.exception("streaming chat failed")
        yield _sse({"type": "error", "message": "The server hit an error while generating the answer."})


def _normalize(s: str) -> str:
    """Collapse whitespace and lowercase — a reflow-robust key for substring tests."""
    return " ".join((s or "").split()).lower()


def _quote_supported(quote: str, passage: str) -> bool:
    """True iff `quote` (normalized) is a non-trivial substring of `passage`.

    The length floor stops a one-word "quote" from trivially matching a long
    passage and passing the grounding check with no real support.
    """
    q = _normalize(quote)
    return len(q) >= MIN_QUOTE_CHARS and q in _normalize(passage)


def _split_citation_block(raw: str) -> "tuple[str, dict[int, str], bool]":
    """Split model output into (display_answer, {n: quote}, block_present).

    The model appends, after the answer:
        <<<CITATIONS>>>
        {"1": "verbatim quote", ...}
    Returns the answer with that block stripped, the parsed marker→quote map, and
    whether a parseable block was found. On a missing/garbled block we return the
    text as-is with an empty map, so verification degrades to in-range-only rather
    than dropping every citation on a formatting slip.
    """
    idx = raw.rfind(CITATION_BLOCK_MARKER)
    if idx == -1:
        return raw.strip(), {}, False
    answer = raw[:idx].rstrip()
    tail = raw[idx + len(CITATION_BLOCK_MARKER):].strip()
    if tail.startswith("```"):  # tolerate a ```json … ``` fence
        tail = tail.strip("`").strip()
        if tail[:4].lower() == "json":
            tail = tail[4:].strip()
    try:
        obj = json.loads(tail)
        quotes = {int(k): str(v) for k, v in obj.items()}
    except (ValueError, TypeError, AttributeError):
        log.warning("citation support block present but unparseable — skipping quote verification")
        return answer, {}, False
    return answer, quotes, True


def _build_citations(answer: str, hits: list[dict], allowed: set, quotes: dict) -> list[dict]:
    """Return one citation entry per unique kept [n], in first-use order.

    `allowed` is the set of markers that passed the range + supporting-quote
    checks; `quotes` maps a marker to the verbatim span the model cited for it
    (attached per entry so the UI can show/highlight the exact support).
    """
    used = [int(n) for n in re.findall(r"\[(\d+)\]", answer)]
    seen: set[int] = set()
    citations: list[dict] = []
    for n in used:
        if n in seen or n not in allowed:
            continue
        seen.add(n)
        h = hits[n - 1]
        citations.append({
            "n": n,
            "source": h["source"],
            "chunk_index": h["chunk_index"],
            # The CFR text reference shown in the UI (e.g. "14 CFR § 91.3"); the
            # section/part are kept as structured metadata. The exact retrieved
            # passage (below) is shown in-app so users can verify without leaving.
            "cfr_citation": h.get("cfr_citation"),
            "section": h.get("section"),
            "part": h.get("part"),
            "text": h.get("text", ""),
            # The verbatim span the model quoted as support for this marker,
            # already verified to be a substring of `text` (None if unverified).
            "quote": quotes.get(n),
        })
    return citations


if __name__ == "__main__":
    # threaded=True so a long-lived streaming response doesn't block other
    # requests (the dev server is single-threaded otherwise).
    app.run(port=5000, debug=True, threaded=True)
