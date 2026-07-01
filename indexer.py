"""Indexer — embed the 14 CFR corpus, persist the index, and search it.

Reads the structure-aware, citation-tagged chunks produced by cfr_ingest.py
(data/corpus.jsonl), embeds each chunk with a selectable backend (Voyage AI or
Gemini — see EMBED_BACKEND), and writes index.pkl, which the chat backend loads
at startup.

Chunking lives in cfr_ingest.py (it parses eCFR XML into §-level chunks); this
file embeds, stores, and serves hybrid retrieval — dense cosine fused with BM25
(Reciprocal Rank Fusion) plus a CFR §/Part router, with an optional reranker.

    python cfr_ingest.py     # build data/corpus.jsonl
    python indexer.py        # embed -> index.pkl

Needs the active backend's credentials (see .env.example): VOYAGE_API_KEY for
voyage, or GEMINI_API_KEY / Vertex-AI ADC for gemini.
"""
import json
import logging
import math
import os
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

# Load .env so VOYAGE_API_KEY / GEMINI_API_KEY / EMBED_BACKEND can live there when
# running `python indexer.py` directly (the backend's app.py loads it too).
load_dotenv(Path(__file__).parent / ".env")

# Embedding backend, selectable via env so you can bake off providers with no
# code change:  EMBED_BACKEND=voyage (default) | gemini
#   voyage -> voyage-4-large       (needs VOYAGE_API_KEY)
#   gemini -> gemini-embedding-001 (needs GEMINI_API_KEY; usable on the Gemini
#             API free tier or a Google Cloud trial credit)
# input_type matters for both: a "document" is embedded differently from a
# "query", and omitting it costs retrieval quality.
EMBED_BACKEND = os.environ.get("EMBED_BACKEND", "voyage").lower()
_DEFAULTS = {  # backend: (model, batch_size, default_dim | None for model-native)
    "voyage": ("voyage-4-large", 128, None),
    "gemini": ("gemini-embedding-001", 100, 1536),
}
if EMBED_BACKEND not in _DEFAULTS:
    raise RuntimeError(f"EMBED_BACKEND must be one of {sorted(_DEFAULTS)}, got {EMBED_BACKEND!r}")
MODEL_NAME = os.environ.get("EMBED_MODEL", _DEFAULTS[EMBED_BACKEND][0])
EMBED_BATCH = int(os.environ.get("EMBED_BATCH", _DEFAULTS[EMBED_BACKEND][1]))
EMBED_DIM = int(os.environ["EMBED_DIM"]) if os.environ.get("EMBED_DIM") else _DEFAULTS[EMBED_BACKEND][2]

CORPUS_PATH = Path(__file__).parent / "data" / "corpus.jsonl"
INDEX_PATH = Path(__file__).parent / "index.pkl"


# ── Embedding (Voyage AI or Gemini) ──────────────────────────────────────────

_voyage_client = None
_gemini_client = None


def _voyage():
    """Lazily build the Voyage client (imported only when this backend is used)."""
    global _voyage_client
    if _voyage_client is None:
        import voyageai
        if not os.environ.get("VOYAGE_API_KEY"):
            raise RuntimeError("VOYAGE_API_KEY is not set — add it to .env (see .env.example).")
        _voyage_client = voyageai.Client()  # reads VOYAGE_API_KEY from the env
    return _voyage_client


def _gemini():
    """Lazily build the Gemini client (imported only when this backend is used).

    Two auth modes:
      - Gemini Developer API (default): set GEMINI_API_KEY (or GOOGLE_API_KEY).
      - Vertex AI via ADC: set GOOGLE_GENAI_USE_VERTEXAI=true and
        GOOGLE_CLOUD_PROJECT (+ optional GOOGLE_CLOUD_LOCATION), then authenticate
        with `gcloud auth application-default login`. No API key — this is the
        path that draws on a Google Cloud trial credit.
    """
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true", "yes")
        if use_vertex:
            project = os.environ.get("GOOGLE_CLOUD_PROJECT")
            if not project:
                raise RuntimeError(
                    "Vertex AI mode (GOOGLE_GENAI_USE_VERTEXAI=true) needs GOOGLE_CLOUD_PROJECT set."
                )
            location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
            # Credentials come from ADC (gcloud auth application-default login, or
            # GOOGLE_APPLICATION_CREDENTIALS) — no API key.
            _gemini_client = genai.Client(vertexai=True, project=project, location=location)
        else:
            if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
                raise RuntimeError(
                    "GEMINI_API_KEY is not set. Either set it (Gemini Developer API), or set "
                    "GOOGLE_GENAI_USE_VERTEXAI=true + GOOGLE_CLOUD_PROJECT to use Vertex AI via ADC."
                )
            _gemini_client = genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY
    return _gemini_client


def _embed_once(texts: list[str], input_type: str) -> list[list[float]]:
    """One embedding API call for the active backend."""
    if EMBED_BACKEND == "voyage":
        kw = {"output_dimension": EMBED_DIM} if EMBED_DIM else {}
        return _voyage().embed(texts, model=MODEL_NAME, input_type=input_type, **kw).embeddings
    # gemini — task_type is Gemini's analogue of Voyage's input_type
    from google.genai import types
    task = "RETRIEVAL_QUERY" if input_type == "query" else "RETRIEVAL_DOCUMENT"
    cfg = types.EmbedContentConfig(task_type=task)
    if EMBED_DIM:
        cfg.output_dimensionality = EMBED_DIM
    resp = _gemini().models.embed_content(model=MODEL_NAME, contents=texts, config=cfg)
    return [e.values for e in resp.embeddings]


def _embed_batch(texts: list[str], input_type: str) -> list[list[float]]:
    for attempt in range(5):
        try:
            return _embed_once(texts, input_type)
        except Exception:  # rate limit / transient API error -> backoff & retry
            if attempt == 4:
                raise
            time.sleep(min(30, 3 * (attempt + 1)))
    raise RuntimeError("unreachable")


def embed(texts: list[str], input_type: str) -> np.ndarray:
    """Embed texts with the active backend (Voyage or Gemini). input_type is
    'document' (indexing) or 'query' (search). Returns an (N, dim) float32 array,
    unit-normalized so cosine similarity is a plain dot product — which also
    fixes Gemini's truncated-dimension vectors, which arrive un-normalized."""
    assert input_type in ("document", "query")
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        vectors.extend(_embed_batch(texts[i:i + EMBED_BATCH], input_type))
        if len(texts) > EMBED_BATCH:
            print(f"  embedded {min(i + EMBED_BATCH, len(texts))}/{len(texts)}", file=sys.stderr)
    arr = np.asarray(vectors, dtype=np.float32)
    # Defensive unit-normalization: Voyage already returns normalized vectors,
    # but this guarantees the dot-product == cosine shortcut used in search().
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


# ── Build / save / load ──────────────────────────────────────────────────────

def load_corpus() -> list[dict]:
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"No corpus at {CORPUS_PATH}. Run `python cfr_ingest.py` first."
        )
    with CORPUS_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_index() -> list[dict]:
    """Embed every chunk in data/corpus.jsonl; return records with embeddings."""
    rows = load_corpus()
    dim = f" (dim={EMBED_DIM})" if EMBED_DIM else ""
    print(f"Embedding {len(rows)} chunks with {EMBED_BACKEND}:{MODEL_NAME}{dim} ...")
    matrix = embed([r["text"] for r in rows], input_type="document")
    records: list[dict] = []
    for row, vec in zip(rows, matrix):
        rec = dict(row)  # carry all metadata (cfr_citation, part, section, …)
        rec["embedding"] = vec.tolist()
        records.append(rec)
    return records


def save_index(records: list[dict]) -> None:
    with INDEX_PATH.open("wb") as f:
        pickle.dump(records, f)


def load_index() -> list[dict]:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"No index at {INDEX_PATH}. Run `python indexer.py` first "
            "(after `python cfr_ingest.py`)."
        )
    with INDEX_PATH.open("rb") as f:
        return pickle.load(f)


# ── Retrieval: hybrid dense + lexical (BM25) with a CFR §-number router ────────
#
# Dense cosine alone misses the exact-term and exact-citation lookups that legal
# text turns on ("the term 'extended overwater operation' means …", "what does
# 61.3 require"). So search() fuses three signals:
#   1. dense   — cosine over the embedding matrix          (semantic recall)
#   2. lexical — BM25 over the chunk text                  (exact term / phrase)
#   3. router  — a bonus for chunks whose CFR §/Part the query names explicitly
# Dense and lexical are combined with Reciprocal Rank Fusion (RRF), which is
# scale-free (no per-channel score normalization to tune); the router adds its
# bonus on top. Every hit keeps `dense_score` (raw cosine); search() also
# force-includes the corpus-wide top-cosine chunk in its results, so the
# backend's relevance gate always sees the true best match — fusion reorders the
# rest, it can't evict that signal.
#
# Env toggles: HYBRID=0 → pure dense; RERANK=1 → Voyage cross-encoder rerank of
# the fused pool (needs VOYAGE_API_KEY).
HYBRID = os.environ.get("HYBRID", "1").lower() in ("1", "true", "yes")
RRF_K = int(os.environ.get("RRF_K", "60"))                     # RRF damping constant
SECTION_BONUS = float(os.environ.get("SECTION_BONUS", "0.5"))  # exact §-number match
PART_BONUS = float(os.environ.get("PART_BONUS", "0.003"))      # same-Part tie-breaker
RERANK = os.environ.get("RERANK", "").lower() in ("1", "true", "yes")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "rerank-2.5")
RERANK_POOL = int(os.environ.get("RERANK_POOL", "30"))         # fused candidates to rerank

# Retrieval trace logger. Library code only *gets* a logger; the application
# (backend/app.py) configures level + handlers. On by default at INFO there, so
# you can see which channels ran and exactly which chunks were retrieved;
# quiet it with RAG_LOG_LEVEL=WARNING.
_log = logging.getLogger("rag.retrieval")

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)*")  # keeps dotted §-numbers (61.3) whole
_SEC_RE = re.compile(r"\b(\d{1,3}\.\d+[a-z]?)\b")     # 61.3, 121.439, 25.1309a
_PART_RE = re.compile(r"\bpart\s+(\d{1,3})\b", re.IGNORECASE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class _BM25:
    """Compact Okapi BM25 over an inverted index (no third-party dependency).

    Built once per `records` list and cached, the same way the dense matrix is.
    """

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b, self.N = k1, b, len(corpus_tokens)
        self.doc_len = np.fromiter((len(t) for t in corpus_tokens), dtype=np.float32, count=self.N)
        self.avgdl = float(self.doc_len.mean()) if self.N else 0.0
        postings: "dict[str, list[tuple[int, int]]]" = {}
        for i, toks in enumerate(corpus_tokens):
            freqs: "dict[str, int]" = {}
            for t in toks:
                freqs[t] = freqs.get(t, 0) + 1
            for t, f in freqs.items():
                postings.setdefault(t, []).append((i, f))
        self.postings = postings
        self.idf = {t: math.log(1 + (self.N - len(p) + 0.5) / (len(p) + 0.5)) for t, p in postings.items()}

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        s = np.zeros(self.N, dtype=np.float32)
        if not self.N or self.avgdl == 0.0:
            return s
        norm = self.k1 * (1.0 - self.b + self.b * self.doc_len / self.avgdl)
        for t in set(query_tokens):
            posting = self.postings.get(t)
            if not posting:
                continue
            idf = self.idf[t]
            for doc_id, f in posting:
                s[doc_id] += idf * (f * (self.k1 + 1.0)) / (f + norm[doc_id])
        return s


# Caches keyed by id(records) but verified by identity — each entry holds the
# records list itself, so a reused id from a GC'd list can never return a stale
# matrix / BM25 index for a different (same-length) corpus.
_matrix_cache: "dict[int, tuple[list, np.ndarray]]" = {}
_bm25_cache: "dict[int, tuple[list, _BM25]]" = {}


def _matrix(records: list[dict]) -> np.ndarray:
    """Stack record embeddings into a cached (N, dim) matrix for fast search.

    Built once per records list (the index the backend loads at startup), not on
    every query.
    """
    entry = _matrix_cache.get(id(records))
    if entry is None or entry[0] is not records:
        matrix = np.asarray([r["embedding"] for r in records], dtype=np.float32)
        _matrix_cache[id(records)] = (records, matrix)
        return matrix
    return entry[1]


def _bm25(records: list[dict]) -> "_BM25":
    """Lazily build + cache the BM25 index over chunk text (per records list)."""
    entry = _bm25_cache.get(id(records))
    if entry is None or entry[0] is not records:
        bm = _BM25([_tokenize(r["text"]) for r in records])
        _bm25_cache[id(records)] = (records, bm)
        return bm
    return entry[1]


def _query_refs(query: str) -> "tuple[set[str], set[int]]":
    """Pull explicit CFR references from a query: section numbers and Parts.

    A bare section like '61.3' also implies its Part (61). Returns (sections, parts).
    """
    sections = set(_SEC_RE.findall(query))
    parts = {int(p) for p in _PART_RE.findall(query)}
    for s in sections:
        parts.add(int(s.split(".")[0]))
    return sections, parts


def _ranks(scores: np.ndarray) -> np.ndarray:
    """0-based rank of each element by descending score (rank 0 == best)."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(scores.shape[0], dtype=np.int64)
    ranks[order] = np.arange(scores.shape[0])
    return ranks


def _strip_embedding(record: dict) -> dict:
    return {k: v for k, v in record.items() if k != "embedding"}


def _make_hit(record: dict, dense: float, lexical: float, score: float) -> dict:
    """Build a returnable hit: the record (minus embedding) + the three scores."""
    h = _strip_embedding(record)
    h["score"] = float(score)
    h["dense_score"] = float(dense)
    h["lexical_score"] = float(lexical)
    return h


def _rerank(query: str, cand_indices, records: list[dict]) -> "list[tuple[int, float]]":
    """Optional Voyage cross-encoder rerank of the candidate pool.

    Returns (record_index, relevance_score) pairs best-first over ALL candidates
    (the caller truncates), so the caller can still force the dense-best chunk to
    survive truncation.
    """
    docs = [records[i]["text"] for i in cand_indices]
    result = _voyage().rerank(query, docs, model=RERANK_MODEL)
    return [(int(cand_indices[r.index]), float(r.relevance_score)) for r in result.results]


def _log_hits(query, hits, *, hybrid, reranked, sections, parts, n_total):
    """Emit a retrieval trace: which channels ran + the ranked chunks returned.

    Logs the exact chunk identity (cfr_citation / source / chunk_index) and every
    score, so a wrong answer can be traced to what retrieval actually surfaced.
    Router-pinned chunks are tagged. Skipped entirely if INFO is disabled.
    """
    if not _log.isEnabledFor(logging.INFO):
        return
    mode = "hybrid(dense+bm25+router)" if hybrid else "dense"
    router = ""
    if hybrid and (sections or parts):
        router = f" router[sections={sorted(sections)} parts={sorted(parts)}]"
    _log.info(
        "retrieval q=%r | backend=%s mode=%s rerank=%s%s | %d of %d chunks",
        query, EMBED_BACKEND, mode, "on" if reranked else "off", router, len(hits), n_total,
    )
    for rank, h in enumerate(hits, 1):
        pinned = ""
        if h.get("section") in sections:
            pinned = " [ROUTER:section]"
        elif h.get("part") in parts:
            pinned = " [ROUTER:part]"
        rr = f" rerank={h['rerank_score']:.3f}" if "rerank_score" in h else ""
        _log.info(
            "  #%d %-14s score=%.4f dense=%.3f bm25=%.2f%s%s  src=%s chunk=%s",
            rank,
            h.get("cfr_citation") or h.get("section") or "?",
            h["score"], h["dense_score"], h["lexical_score"], rr, pinned,
            h.get("source"), h.get("chunk_index"),
        )


def search(query: str, records: list[dict], k: int = 5) -> list[dict]:
    """Return the top-k records most relevant to the query.

    Hybrid by default (dense ⊕ BM25 fused with RRF, plus a CFR §/Part router);
    set HYBRID=0 for pure dense. Each hit is a copy of the record (minus the
    bulky embedding) with:
      - `dense_score`   : cosine similarity in [-1, 1] — the relevance-gate signal
      - `lexical_score` : raw BM25 score (0.0 when HYBRID=0)
      - `score`         : the value hits are ordered by (fused in hybrid mode,
                          cosine in dense mode)
    """
    if not records:
        return []
    [qvec] = embed([query], input_type="query")
    sims = _matrix(records) @ qvec  # cosine, since every row is unit-normalized
    n = len(records)

    if not HYBRID:
        kk = min(k, n)
        top = np.argpartition(-sims, kk - 1)[:kk]
        top = top[np.argsort(-sims[top])]
        hits = [_make_hit(records[i], sims[i], 0.0, sims[i]) for i in top]
        _log_hits(query, hits, hybrid=False, reranked=False, sections=set(), parts=set(), n_total=n)
        return hits

    bm = _bm25(records).scores(_tokenize(query))
    sections, parts = _query_refs(query)
    bonus = np.zeros(n, dtype=np.float32)
    if sections or parts:
        for i, r in enumerate(records):
            if r.get("section") in sections:
                bonus[i] = SECTION_BONUS
            elif r.get("part") in parts:
                bonus[i] = PART_BONUS

    fused = 1.0 / (RRF_K + _ranks(sims) + 1) + 1.0 / (RRF_K + _ranks(bm) + 1) + bonus

    # Candidate pool by fused score, but ALWAYS force-include the corpus-wide best
    # dense match. The relevance gate reads max(dense_score) from the returned
    # hits, so evicting the top-cosine chunk here would mis-calibrate the gate
    # (false abstains on well-phrased, low-keyword-overlap questions) and hide the
    # strongest semantic chunk from the LLM.
    pool = min(max(k, RERANK_POOL) if RERANK else k, n)
    cand = np.argsort(-fused, kind="stable")[:pool]
    best_dense = int(np.argmax(sims))
    if best_dense not in cand.tolist():
        cand = np.append(cand[:pool - 1], best_dense)

    rerank_score: "dict[int, float]" = {}
    if RERANK and len(cand):
        ranked = _rerank(query, cand, records)        # all candidates, best-first
        rerank_score = dict(ranked)
        final = [i for i, _ in ranked][:k]
        if best_dense not in final:                   # keep the gate signal intact
            final = final[:k - 1] + [best_dense]
    else:
        final = [int(i) for i in cand[:k]]

    hits = []
    for i in final:
        h = _make_hit(records[i], sims[i], bm[i], fused[i])
        if i in rerank_score:
            h["rerank_score"] = rerank_score[i]
            h["score"] = rerank_score[i]  # reflect the final (rerank) ordering
        hits.append(h)
    _log_hits(
        query, hits, hybrid=True, reranked=bool(rerank_score),
        sections=sections, parts=parts, n_total=n,
    )
    return hits


def main() -> None:
    records = build_index()
    save_index(records)
    print(f"\n✓ Indexed {len(records)} chunks → {INDEX_PATH.name}")


if __name__ == "__main__":
    main()
