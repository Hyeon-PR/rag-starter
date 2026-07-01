"""Context Management RAG starter — extended chat backend.

This is the Foundations chat backend with stubs for retrieval-augmented generation.

TODO:
  1. Update SYSTEM_PROMPT with citation rules.
  2. In /api/chat: retrieve top-K chunks for the user's question.
  3. Format chunks as a numbered context block.
  4. Build the user_content with CONTEXT + QUESTION.
  5. Parse citation numbers from the answer; return them to the frontend.
"""
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
from flask import Flask, jsonify, request
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
and no closing summary. Be as brief as the question allows while still citing every claim with [n]."""


# Relevance gate. Calibrated on the Gemini index (in-domain top scores ~0.73–0.79,
# out-of-domain ~0.49–0.61): if the best dense-cosine match is below MIN_TOP_SCORE
# we treat the question as out-of-scope and abstain WITHOUT calling the LLM — a
# cheap, hard guarantee against answering when nothing relevant was retrieved
# (also screens off-topic and injected prompts). Re-calibrate per embedding
# backend; env-tunable.
TOP_K = int(os.environ.get("RETRIEVAL_K", "8"))
MIN_TOP_SCORE = float(os.environ.get("RETRIEVAL_MIN_SCORE", "0.66"))
CHAT_MODEL = os.environ.get("CHAT_MODEL", "claude-sonnet-4-6")
ABSTAIN_REPLY = (
    "The provided 14 CFR sources don't contain anything relevant to that question, "
    "so I can't answer it from the corpus."
)


@app.route("/api/chat", methods=["POST"])
def chat():
    user_message = request.json["message"]
    log.info("chat q=%r", user_message)

    # Retrieve the top-K most relevant chunks. indexer's rag.retrieval logger
    # traces which channels ran (dense/BM25/router/rerank) and the exact chunks.
    t_start = time.perf_counter()
    hits = search(user_message, INDEX, k=TOP_K)
    retrieval_ms = (time.perf_counter() - t_start) * 1000  # incl. query-embedding round-trip

    # Relevance gate: if the best dense-cosine match is below the bar, abstain up
    # front — no LLM call, no chance to hallucinate an answer the corpus can't
    # support. Gate on dense_score (raw cosine), not the hybrid `score`, which is
    # rank-based and not comparable to the calibrated cosine threshold.
    top_dense = max((h["dense_score"] for h in hits), default=0.0)
    if not hits or top_dense < MIN_TOP_SCORE:
        log.info(
            "gate top_dense=%.3f < %.2f -> ABSTAIN (no LLM call) | retrieval=%.0fms in=0 out=0",
            top_dense, MIN_TOP_SCORE, retrieval_ms,
        )
        return jsonify({
            "reply": ABSTAIN_REPLY,
            "citations": [],
            "abstained": True,
            "meta": {
                "question": user_message,
                "retrieval_ms": round(retrieval_ms),
                "total_ms": round(retrieval_ms),
            },
        })
    log.info(
        "gate top_dense=%.3f >= %.2f -> ANSWER (%d chunks in context)",
        top_dense, MIN_TOP_SCORE, len(hits),
    )

    # Augment the prompt with a numbered context block so the model can ground
    # its answer and cite sources.
    context = "\n\n".join(f"[{i + 1}] {h['text']}" for i, h in enumerate(hits))
    user_content = f"CONTEXT:\n{context}\n\nQUESTION:\n{user_message}"

    # Single-turn: only this question + the retrieved context is sent — no prior
    # conversation is threaded in — so input_tokens ≈ system + context + question.
    t_llm = time.perf_counter()
    resp = client.messages.create(
        model=CHAT_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )
    llm_ms = (time.perf_counter() - t_llm) * 1000
    answer = resp.content[0].text

    # Token usage + latency (total = retrieval, incl. embedding round-trip, + LLM).
    usage = resp.usage
    total_ms = (time.perf_counter() - t_start) * 1000
    log.info(
        "llm model=%s in=%d out=%d | latency retrieval=%.0fms llm=%.0fms total=%.0fms",
        CHAT_MODEL, usage.input_tokens, usage.output_tokens,
        retrieval_ms, llm_ms, total_ms,
    )

    # ────────────────────────────────────────────────────────────
    # TODO — citation extraction
    #
    # Parse [n] markers from the answer. Drop invented ones.
    # For each valid citation, return its source filename and chunk index
    # so the frontend can display them.
    # ────────────────────────────────────────────────────────────

    citations = _build_citations(answer, hits)
    meta = {
        # Echo the exact question this answer was produced for. A downstream
        # eval/grader can assert row["question"] == meta["question"] to catch a
        # mis-zip (the question column drifting out of sync with the answer it's
        # graded against) instead of silently scoring the wrong pair.
        "question": user_message,
        "model": CHAT_MODEL,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "retrieval_ms": round(retrieval_ms),
        "llm_ms": round(llm_ms),
        "total_ms": round(total_ms),
    }

    # Log exactly which retrieved chunks the answer cited (with a text snippet),
    # and flag any [n] the model emitted that has no matching source. This is the
    # crux of debugging whether an answer is grounded in the right reference.
    used = [int(x) for x in re.findall(r"\[(\d+)\]", answer)]
    valid_ns = {c["n"] for c in citations}
    invalid = sorted({x for x in used if x not in valid_ns})
    if citations:
        log.info("answer cited %d of %d retrieved chunks:", len(citations), len(hits))
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
        # A non-abstained answer with no resolvable citation is ungrounded: the
        # relevance gate let it through on retrieval score, but the model cited
        # nothing. Surface it (grounded=False, and a UI notice) instead of
        # silently returning it as if it were a normal cited answer. A hard
        # re-ask/abstain needs the entailment pass (roadmap); this is the flag.
        log.warning("non-abstained answer carries no citations — ungrounded")
    # Neutralize any out-of-range/invented [n] in the returned answer so a
    # consumer of `reply` that doesn't run the frontend verifier can't mistake it
    # for a real, resolvable citation. Valid markers are untouched; invalid ones
    # become a literal [?], which the frontend renders as a flagged badge.
    safe_answer = answer
    if invalid:
        log.warning(
            "answer used citation(s) with no matching source: %s (neutralized to [?])", invalid
        )
        safe_answer = re.sub(
            r"\[(\d+)\]",
            lambda m: m.group(0) if int(m.group(1)) in valid_ns else "[?]",
            answer,
        )

    return jsonify({
        "reply": safe_answer,
        "citations": citations,
        "grounded": bool(citations),
        "invalid_citations": invalid,
        "meta": meta,
    })


def _ecfr_url(hit: dict) -> str:
    """Deep link to the official eCFR page for a hit's CFR location.

    Uses eCFR's section permalink (…/current/title-14/section-91.3) for a normal
    numbered section; falls back to the Part page for appendices / SFARs or
    anything without a clean section number. eCFR's exact path scheme lives only
    here, so if it ever changes this is the one place to fix.
    """
    title = hit.get("title", 14)
    section = (hit.get("section") or "").strip()
    part = hit.get("part")
    base = f"https://www.ecfr.gov/current/title-{title}"
    if re.match(r"^\d+\.\w+$", section):
        return f"{base}/section-{section}"
    if part:
        return f"{base}/part-{part}"
    return base


def _build_citations(answer: str, hits: list[dict]) -> list[dict]:
    """Return one citation entry per unique valid [n] used in the answer.

    Starter implementation: extract the numbers, drop out-of-range, return
    the matching hit's filename + chunk_index. Improve as you like.
    """
    used = [int(n) for n in re.findall(r"\[(\d+)\]", answer)]
    seen: set[int] = set()
    citations: list[dict] = []
    for n in used:
        if n in seen or n < 1 or n > len(hits):
            continue
        seen.add(n)
        h = hits[n - 1]
        citations.append({
            "n": n,
            "source": h["source"],
            "chunk_index": h["chunk_index"],
            # For the UI: jump straight to the source (eCFR deep link) and show
            # the exact retrieved passage in-app.
            "cfr_citation": h.get("cfr_citation"),
            "section": h.get("section"),
            "part": h.get("part"),
            "url": _ecfr_url(h),
            "text": h.get("text", ""),
        })
    return citations


if __name__ == "__main__":
    app.run(port=5000, debug=True)
