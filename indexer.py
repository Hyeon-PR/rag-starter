"""Indexer — embed the 14 CFR corpus and persist a searchable index.

Reads the structure-aware, citation-tagged chunks produced by cfr_ingest.py
(data/corpus.jsonl), embeds each chunk with Voyage AI (voyage-4-large), and
writes index.pkl, which the chat backend loads at startup.

Chunking now lives in cfr_ingest.py (it parses eCFR XML into §-level chunks),
so this file only embeds, stores, and searches.

Requires VOYAGE_API_KEY in the environment (see .env.example):
    python cfr_ingest.py     # build data/corpus.jsonl
    python indexer.py        # embed -> index.pkl
"""
import json
import os
import pickle
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


# ── Search ───────────────────────────────────────────────────────────────────

_matrix_cache: "dict[int, np.ndarray]" = {}


def _matrix(records: list[dict]) -> np.ndarray:
    """Stack record embeddings into a cached (N, dim) matrix for fast search.

    Cached by id(records) so the matrix is built once for the index the backend
    loads at startup, not on every query.
    """
    key = id(records)
    cached = _matrix_cache.get(key)
    if cached is None or cached.shape[0] != len(records):
        cached = np.asarray([r["embedding"] for r in records], dtype=np.float32)
        _matrix_cache[key] = cached
    return cached


def search(query: str, records: list[dict], k: int = 5) -> list[dict]:
    """Return the top-k records most similar to the query.

    Each hit is a copy of the record (minus the bulky embedding) with a `score`
    added — cosine similarity in [-1, 1], higher is better — so callers can apply
    a relevance gate / abstain path.
    """
    if not records:
        return []
    [qvec] = embed([query], input_type="query")
    sims = _matrix(records) @ qvec  # cosine, since every row is unit-normalized
    k = min(k, len(records))
    top = np.argpartition(-sims, k - 1)[:k]
    top = top[np.argsort(-sims[top])]
    hits: list[dict] = []
    for i in top:
        hit = {key: val for key, val in records[i].items() if key != "embedding"}
        hit["score"] = float(sims[i])
        hits.append(hit)
    return hits


def main() -> None:
    records = build_index()
    save_index(records)
    print(f"\n✓ Indexed {len(records)} chunks → {INDEX_PATH.name}")


if __name__ == "__main__":
    main()
