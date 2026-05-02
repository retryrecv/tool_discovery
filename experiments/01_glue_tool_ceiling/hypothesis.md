# 01 — Glue-tool ceiling measurement

## Hypothesis

The complex misses (10 cases at recall 0.91, 0.80 full-cover) all share the same shape: the LLM decomposer fails to name an implicit utility tool (`calculator`, `get_current_datetime`, `http_get`, `url_parse`, `json_query`).

If true, then **upper-bounding decomposition quality** (replacing the LLM's sub-queries with an oracle list built from the gold tools' own intent phrases) should push complex set-recall@10 to ≈1.0 and full-cover to ≈1.0.

## What this measures

This is a **diagnostic ceiling**, not a deployable change:
- **If oracle recall ≈ 1.0** → the entire complex-recall gap is decomposition-side. Every future spike should target the decomposer (better prompts, schema injection, post-decomposition glue-tool sweep, ReAct-style refinement). No retrieval/rerank/clustering work is justified.
- **If oracle recall < 1.0** → there is residual retrieval/rerank failure even with perfect intent. The remaining gap quantifies how much paper-driven retrieval work could buy.

## Method

1. Load the same v8 snapshot that the production eval uses (`raw-tools/v8`).
2. For each complex test case, build sub-queries directly from the **gold tools' `intent_phrase` enrichments** (one sub-query per gold tool). This is the strongest possible decomposition — every required tool is mentioned by name.
3. Feed those sub-queries through the unchanged `retrieve_decomposed` path (same beam=3, rerank_k=20, k=10).
4. Measure simple recall, complex set-recall, complex full-cover. Compare to the production eval (0.960 / 0.908 / 0.800 per the 2026-05-01 run).
5. Apply the same oracle to ULTRA_COMPLEX cases for a second reading.

## Decision rule

| Oracle complex set-recall | Oracle full-cover | Verdict |
|---|---|---|
| ≥ 0.99 | ≥ 0.96 | Decomposition-side bottleneck confirmed. Next spike: glue-tool prompt hints (#02). |
| 0.95–0.98 | 0.85–0.95 | Mostly decomposition-side, minor retrieval residue. Same next spike, deprioritize retrieval work. |
| < 0.95 | < 0.85 | Retrieval bottleneck still exists even with oracle decomposition. Triage the still-missing tools manually before any further work. |

## Out of scope

- Doesn't change the production pipeline.
- Doesn't measure ULTRA_COMPLEX as a primary metric (the test set was extended after D4 shipped; we lack a baseline for it).
- Doesn't touch the simple cases — they're already at 0.96 with no decomposition involved.
