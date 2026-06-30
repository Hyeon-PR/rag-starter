# Documentation

Map of the docs in this repo.

| Doc | What it covers |
|---|---|
| [`../README.md`](../README.md) | Project overview, setup, and the lab assignment (build the index, wire retrieval + citations). **Start here.** |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Production-grade target architecture — an upgrade path from the Apollo starter toward the **14 CFR contest corpus**, organized against the 6-category evaluation rubric (retrieval, generation, memory, data pipeline, cost/latency, safety, validation). |
| [`embedding-model.md`](embedding-model.md) | Decision note: why the design uses **`voyage-4-large`** (bake-off vs `voyage-law-2`, `voyage-4-nano` as the local fallback) for 14 CFR. |
| [`../frontend/README.md`](../frontend/README.md) | Frontend (React + Vite) chat UI — features and setup. |

## Target corpus pivot

The starter ships an **Apollo** sample corpus (21 `.md`, English Wikipedia). The **contest grades on all of 14 CFR** (FAA aviation regulations — English legal/regulatory text, ~50k–150k chunks). That pivot drives the current design decisions:

- **Embedder:** `voyage-4-large` (Anthropic-recommended Voyage AI), not `bge-m3` — see [`embedding-model.md`](embedding-model.md). Language is no longer a constraint and a paid API is acceptable.
- **Search:** brute-force cosine is dead at Title-14 scale → real ANN (pgvector HNSW / Qdrant) + a `§`-number router + BM25 hybrid.
- **Citations:** CFR's canonical `14 CFR § …` references are externally verifiable — a direct win for the Citations rubric.

## Current state vs. this design

The code in `indexer.py` / `backend/app.py` / `frontend/` is the **working starter**: single-stage dense retrieval (fixed `k=5`), one Sonnet call, single-turn, Apollo corpus. [`ARCHITECTURE.md`](ARCHITECTURE.md) is the **forward-looking design** — every recommendation is tagged **KEEP / ADD / REPLACE**, so it doubles as a roadmap. Nothing in it is implemented yet unless the code says otherwise.

## Known immediate fixes (surfaced by the architecture review)

These are small, real, and independent of the larger design:

- **Ingest 14 CFR.** The contest corpus is not in `documents/` as indexable text yet — pull the FAA Title 14 source (prefer eCFR bulk XML over PDF for clean Part/§ structure) and reindex.
- **`index.pkl` is stale** — built with the old 1000/100 chunker, *before* the current 1500/200 `chunk_text`. It is also Apollo-only. Rebuild after the corpus + embedder swap.
- **`documents/06-changelog.md`** is a non-corpus file that still gets indexed and can pollute answers/citations — add it to an ingest denylist.
- **`search()` discards distances** — return `(score, record)` so a relevance gate / abstain path becomes possible.
- **Embedder swap** — replace `sentence-transformers` with the `voyageai` client; remember `input_type="document"` at index and `input_type="query"` at search (Voyage embeddings stay unit-normalized, so the `1 − dot` cosine math is unchanged).
