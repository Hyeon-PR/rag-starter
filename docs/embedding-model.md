# Decision note — embedding model

**Status:** proposed · **Date:** 2026-07-01 · **Scope:** retrieval bi-encoder choice
**Decision:** Use **`voyage-4-large`** as the dense embedder for the contest's **14 CFR** corpus, decided by a head-to-head bake-off against the legal-domain **`voyage-law-2`** (with the open-weight **`voyage-4-nano`** as the free local baseline / fallback). Keep extra precision in a cross-encoder **reranker** and a lexical **BM25** channel — the embedder is one stage, not the whole answer.

> **This supersedes the earlier `bge-m3` recommendation.** That call was made for the Apollo Wikipedia starter corpus *and* a multilingual (Korean-query) requirement. Both premises changed: the contest grades on **14 CFR** (English legal/regulatory text), **language no longer matters** (optimize for raw retrieval performance), and a **paid embedding API is acceptable**. When the premises change, the decision changes.

## Context

Anthropic ships no embedding model of its own; its [official guidance](https://platform.claude.com/docs/en/build-with-claude/embeddings) recommends **Voyage AI**. The relevant Voyage models:

| Model | Context | Dims | Notes |
|---|---|---|---|
| **`voyage-4-large`** | 32,000 | 1024 default · 256/512/2048 (Matryoshka) | Latest gen (Jan 2026). Best general-purpose + multilingual retrieval. |
| `voyage-4` / `voyage-4-lite` | 32,000 | same | Balanced / cost-latency tiers. |
| `voyage-4-nano` | 32,000 | same | **Open-weight (Apache-2.0, on Hugging Face)** — locally deployable. |
| **`voyage-law-2`** | 16,000 | 1024 | Previous gen (2024). Tuned for **legal + long-context** retrieval. |

The baseline `paraphrase-multilingual-MiniLM-L12-v2` (384-d, **128-token window**) silently truncates the corpus's chunks to their first ~128 tokens before embedding — a poor *retrieval* model regardless of corpus. The real question is which strong model to replace it with.

## Why a premium embedder is now justified (it wasn't for Apollo)

On clean encyclopedic prose, first-stage recall is rarely the bottleneck. **14 CFR is the opposite case:** dense cross-references, defined terms, "shall/may/must" phrasing, and large volumes of near-duplicate boilerplate that weak embedders conflate. And the asymmetry is decisive — **if the bi-encoder misses the right § at stage 1, no reranker can recover it** (rerank only reorders what was retrieved). So first-stage embedder quality matters far more here.

## Why `voyage-4-large` over `voyage-law-2` (lean, then verify)

The crux is whether a **2024 legal-specialized** model still beats a **2026 general-SOTA** model on legal retrieval. The lean is `voyage-4-large`, for reasons beyond recency:

1. **Generational gap usually wins.** Two generations of base-model improvement typically erase an older model's domain edge.
2. **32k context vs 16k.** Some CFR sections are long; 32k lets you embed larger parent chunks without truncation — arguably more "long-context legal" headroom than `voyage-law-2` itself.
3. **Matryoshka dims + int8/binary quantization** (4×/32× smaller) keep an all-of-Title-14 index (~50k–150k vectors) compact and fast. `voyage-law-2` offers neither.

But "lean" is not "decide." **Bake off `voyage-4-large` vs `voyage-law-2`** (and `voyage-4-nano` as the free baseline) on a 14 CFR golden set; pick on recall@k + citation-correctness, not reputation. See [`ARCHITECTURE.md`](ARCHITECTURE.md) Phase 4.

## Cost is (almost) irrelevant to the score

The Cost rubric counts **LLM context tokens**, not embedding tokens. Embedding all of Title 14 is a one-time job (a few dollars even at premium pricing); per-query embedding is a handful of tokens = negligible. So choosing the *best* embedder costs essentially nothing in rubric-cost terms — don't down-tier on cost grounds. (Verify current pricing on Voyage's pricing page.)

## What going Voyage actually trades

- **External API dependency** — acceptable here (the contest permits it); wire `voyage-4-nano` locally as a fallback so a Voyage outage/throttle degrades gracefully.
- **Latency** — +network round-trip per query (~100–300 ms) vs ~30–60 ms local. Fine for UX.
- **Reproducibility** — a hosted model can shift; pin the model string and snapshot the index. `voyage-4-nano` is pinnable if that matters.
- **Lose the native sparse channel** — `bge-m3` emitted dense + sparse from one model; Voyage is dense-only, so **add BM25 separately** (the design wanted hybrid anyway). Voyage swaps only the dense bi-encoder; the hybrid + rerank scaffolding stands.

## Migration gotchas (small, but easy to get wrong)

- **`input_type` is mandatory.** Pass `input_type="document"` at index time and `input_type="query"` at search time — Voyage prepends different prompts; omitting it costs recall.
- **Embeddings are unit-normalized**, so the starter's `cosine = 1 − dot` shortcut stays valid — the swap is just the `embed()` implementation (`sentence-transformers` → `voyageai` client), not the distance math.

## When `voyage-law-2` *should* win the bake-off (so it's a fair test)

If the golden set is dominated by tightly-scoped, terminology-heavy clause lookups where domain vocabulary tuning beats general semantic strength, `voyage-law-2` may edge ahead. That is exactly what the measurement is for — let the 14 CFR numbers decide.
