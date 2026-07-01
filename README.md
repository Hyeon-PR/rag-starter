# 14 CFR RAG — retrieval-augmented Q&A over FAA aviation regulations

A retrieval-augmented chat system over **Title 14 of the U.S. Code of Federal
Regulations** (FAA aviation regulations). Ask a question in plain language or by
section number; the system retrieves the relevant CFR chunks, answers **strictly
from them**, and cites the exact `14 CFR §` sources — or **abstains** when the
corpus doesn't cover the question.

Built for the 14 CFR contest. The design rationale and forward roadmap live in
[`docs/`](docs/).

## Pipeline

```
eCFR XML  ─►  cfr_ingest.py  ─►  data/corpus.jsonl  ─►  indexer.py  ─►  index.pkl  ─►  backend/app.py  ─►  frontend/
 (source)     structure-aware     §-level, cited         embed +         dense+BM25      Flask chat:        React chat
              chunking            chunks (~10.7k)         persist         hybrid index    gate→cite→answer    + Sources
```

1. **Ingest** — `cfr_ingest.py` pulls Title 14 from the eCFR versioner API and
   parses the regulation XML into structure-aware, citation-tagged chunks. The
   **§ (section) is the retrieval + citation unit**: short sections are kept
   whole, long ones split with overlap; appendices/SFARs are handled too.
   Stdlib only (no third-party parser). → `data/corpus.jsonl` (~10,744 chunks).
2. **Index** — `indexer.py` embeds each chunk with a **selectable backend**
   (Voyage AI or Gemini) and persists `index.pkl`.
3. **Retrieve** — `indexer.search()` is **hybrid**: dense cosine ⊕ BM25 fused by
   Reciprocal Rank Fusion, plus a **CFR §/Part router** that pins exact-citation
   queries ("what does 91.3 say") to the right section.
4. **Generate** — `backend/app.py` runs a **relevance gate**: if nothing
   retrieved clears the calibrated cosine bar it abstains up front (no LLM call,
   no chance to hallucinate). Otherwise Claude answers from a numbered context
   block and **every claim is cited `[n]`**, mapped back to its `14 CFR §` source
   for the frontend.

## Layout

```
rag-starter/
├── cfr_ingest.py            eCFR XML → structure-aware, cited §-chunks  → data/corpus.jsonl
├── indexer.py               embed (Voyage|Gemini) + hybrid search       → index.pkl
├── backend/
│   ├── app.py               Flask chat: retrieve → gate → cite → answer
│   └── requirements.txt
├── frontend/                React + Vite chat UI (citations + Sources)
├── docs/                    architecture, embedding-model decision, doc index
└── .env.example
```

## Setup

```bash
# from the repo root — use a Linux/WSL Python (the .venv here is a Windows venv)
python3 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt

cp .env.example .env         # then edit .env (see below)
```

`.env` needs `ANTHROPIC_API_KEY` (for generation) and one embedding backend:

| `EMBED_BACKEND` | Credentials | Notes |
|---|---|---|
| `voyage` (default) | `VOYAGE_API_KEY` | `voyage-4-large`, 1024-d. |
| `gemini` | `GEMINI_API_KEY` **or** Vertex AI via ADC | `gemini-embedding-001`, 1536-d. ADC (`gcloud auth application-default login`) uses a Google Cloud trial credit — no API key. |

See [`docs/embedding-model.md`](docs/embedding-model.md) for the why and the
Vertex/ADC setup.

## Build the corpus + index

```bash
python cfr_ingest.py         # eCFR XML → data/corpus.jsonl   (~10.7k chunks)
python indexer.py            # embed   → index.pkl
```

`cfr_ingest.py` flags: `--parts 1 73 91` (only these Parts), `--limit N` (first
N Parts), `--date YYYY-MM-DD` (eCFR issue date; default latest), `--refresh`
(re-fetch even if cached). Fetched Part XML is cached under `data/`, so re-runs
are cheap.

## Run

```bash
# Terminal 1 — backend
cd backend && python app.py          # http://localhost:5000

# Terminal 2 — frontend
cd frontend && npm install && npm run dev   # http://localhost:5173
```

Open <http://localhost:5173>.

**Try these:**

- *"What does 14 CFR 91.3 say about the authority of the pilot in command?"* —
  the §-router pins **§ 91.3** to the top.
- *"What are the requirements to be issued a private pilot certificate?"* —
  semantic, spans several sections of Part 61.
- *"What's a good recipe for chocolate-chip cookies?"* — **out-of-corpus**; the
  system abstains instead of guessing.

## Retrieval knobs (env)

| Var | Default | Effect |
|---|---|---|
| `EMBED_BACKEND` / `EMBED_MODEL` / `EMBED_DIM` | `voyage` / model default / backend default | embedding backend + dims |
| `RETRIEVAL_K` | `5` | chunks retrieved per query |
| `RETRIEVAL_MIN_SCORE` | `0.66` | abstain threshold (best dense cosine); recalibrate per backend |
| `HYBRID` | `1` | `0` → pure dense (no BM25 / router) |
| `RRF_K` / `SECTION_BONUS` / `PART_BONUS` | `60` / `0.5` / `0.003` | fusion + §/Part router tuning |
| `RERANK` | off | `1` → Voyage cross-encoder rerank of the fused pool (needs `VOYAGE_API_KEY`) |

## Debugging retrieval

The backend logs a full trace per request (default level `INFO`; quiet with
`RAG_LOG_LEVEL=WARNING`) so you can see *why* an answer came out the way it did —
essential for judging whether it's grounded in the right section:

- **which channels ran** — `mode=hybrid(dense+bm25+router)` vs `dense`,
  `rerank on/off`, and the `§`/Part the router extracted from the query;
- **the exact chunks retrieved** — rank, `cfr_citation`, the
  `score`/`dense`/`bm25`(/`rerank`) values, `[ROUTER:…]` tags, and
  `source` + `chunk_index`;
- **the gate decision** — `top_dense` vs the abstain threshold (`ANSWER` / `ABSTAIN`);
- **which chunks the answer actually cited** — each `[n]` mapped to its
  `cfr_citation` / `source` / `chunk_index` with a text snippet, plus a warning
  for any `[n]` the model emitted with no matching source;
- **cost + latency** — the model's `input` / `output` token counts and per-stage
  latency (`retrieval`, which includes the query-embedding round-trip; `llm`; and
  `total`).

Each request is **single-turn / stateless** — only the current question plus the
retrieved context is sent to the model (no prior conversation), so `input` tokens
≈ system prompt + context + question.

```
INFO rag.chat: gate top_dense=0.731 >= 0.66 -> ANSWER (5 chunks in context)
INFO rag.retrieval: retrieval q='what does 91.3 say...' | backend=gemini mode=hybrid(dense+bm25+router) rerank=off router[sections=['91.3'] parts=[91]] | 5 of 10744 chunks
INFO rag.retrieval:   #1 14 CFR 91.3    score=0.53 dense=0.73 bm25=1.36 [ROUTER:section]  src=part-91.xml chunk=0
INFO rag.chat: llm model=claude-sonnet-4-6 in=812 out=143 | latency retrieval=118ms llm=1274ms total=1392ms
INFO rag.chat: answer cited 1 of 5 retrieved chunks:
INFO rag.chat:   [1] 14 CFR 91.3 src=part-91.xml chunk=0 :: The pilot in command is directly responsible for, and is the final authority...
```

## Notes

- `.env` holds secrets and is gitignored. `index.pkl` and `data/` are
  regenerable artifacts (also gitignored) — rebuild them with the commands above.
- The abstain threshold is calibrated to the embedding backend. The shipped
  `0.66` was set on the Gemini index (in-domain top scores ~0.73–0.80,
  out-of-domain ~0.49–0.61); re-measure if you switch backends or dims.

## Docs

- [`docs/README.md`](docs/README.md) — documentation index + status.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — target architecture / roadmap,
  mapped to the evaluation rubric (KEEP / ADD / REPLACE).
- [`docs/embedding-model.md`](docs/embedding-model.md) — embedding-model decision
  note (Voyage vs Gemini, Vertex/ADC).
