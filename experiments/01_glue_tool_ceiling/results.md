# 01 — Glue-tool ceiling: results

**Date:** 2026-05-01
**Snapshot:** `raw-tools/v8` (200 tools)
**Pipeline:** beam=3, rerank_k=20, k=10 (unchanged from production)

## Headline numbers

| Metric | Production (LLM decomposer) | Oracle decomposition | Gap |
|---|---|---|---|
| COMPLEX set-recall@10 | 0.908 | **1.000** | +0.092 |
| COMPLEX full-cover | 0.800 | **1.000** | +0.200 |
| ULTRA set-recall@10 | (not run today) | **1.000** | — |
| ULTRA full-cover | (not run today) | **1.000** | — |

**Zero residual misses.** Every gold tool in every complex and ultra-complex case is retrievable when the decomposer names it. The retrieval+rerank stack downstream of decomposition is not the bottleneck.

## Verdict

Decomposition-side bottleneck **confirmed and isolated**, per the hypothesis decision rule:

> ≥ 0.99 set-recall, ≥ 0.96 full-cover → next spike: glue-tool prompt hints

100% on both means the entire 9.2-point complex set-recall gap and the entire 20-point full-cover gap are caused by the LLM decomposer failing to name implicit utility tools. No retrieval, rerank, beam-width, or clustering work is justified for the complex/ultra-complex regime until decomposition is solved.

## Implications for paper-driven work

This kills the priority of any paper that targets retrieval-after-decomposition for our current bottleneck:
- **DSP, IRCoT, RQ-RAG, ColBERT-style retrieval improvements** — would all push against an already-saturated ceiling.
- The papers worth deep-reading are now narrowed to the three that target *what to feed the retriever*: **ReAct, Self-Ask, Least-to-Most refinement**. All three converge on post-retrieval feedback to catch missing intents.

But before any paper-driven spike: **try the cheap things first** since the ceiling shows there's 9–20 points of free recall sitting on the table:

1. **Spike 02 — glue-tool prompt hints**: tell the decomposer "if query mentions a URL, include `url_parse` and `http_get`; if it mentions arithmetic/conversion/totalling, include `calculator`; if it mentions a time/date, include `get_current_datetime`." 30-minute prompt experiment. Likely captures most of the gap.

2. **Spike 03 — pattern-match injector** (if Spike 02 doesn't close most of the gap): regex on the original query for keywords (URL, JSON, %, mile/km, total, convert) and inject the matching tool's `intent_phrase` as an extra sub-query alongside the LLM output. Doesn't even need an LLM call.

3. **Only after those plateau** — promote one of ReAct / Self-Ask / Least-to-Most to a full spike with a worktree, decision rule, and ablation plan.

## Cost paid

44 embedding calls (one per gold tool intent_phrase, deduped by cache hit). ~13 seconds wall time. No LLM calls (oracle bypasses the decomposer). Cache-friendly for re-runs.

## What this does NOT show

- **Doesn't measure simple recall** — simple cases use direct embedding retrieval, not decomposition. Their 0.96 number is unrelated to this spike.
- **Doesn't validate the LLM decomposer's positive work** — the decomposer correctly handles 91% of cases. The ceiling shows the *upper bound*, not which queries the decomposer already solves.
- **Doesn't account for false positives** — oracle injects exactly the right intents. A real glue-tool injector that fires too eagerly could hurt precision/rerank quality. Spikes 02/03 must measure full set-recall, not just whether gold tools appear.
