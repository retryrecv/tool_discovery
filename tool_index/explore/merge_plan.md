# Merge plan: Direction 3 + Direction 1

## Status

- D3 traverser change exists on `explore/multivector-nodes` (commit `2347e12`).
- D3+D1 stacked exists on `explore/multivector-plus-rerank` (D1 lives only as a script `eval_colbert_rerank.py`, not yet productized).
- Two proposed `tasks.json` entries drafted in `proposed_tasks.json` (priorities 56, 57).

## Sequenced merge to `phase1-router`

### Step 1 — land D3 (the proven, single-file win)

1. Cherry-pick `2347e12` (D3 traverser change) onto `phase1-router`.
2. Add unit tests called out in `explore-direction3-multivector-traverser.steps[3-4]` to `tests/unit/test_retrieval/`.
3. Append the D3 entry from `proposed_tasks.json` to `tasks.json` with `passes: true`.
4. Run full suite: `cd tool_index && uv run pytest`.
5. Commit: "Promote Direction 3: child-MaxSim inner-node scoring (recall 0.917 -> 0.978)".

### Step 2 — productize D1 (move the script logic into the library)

D1 currently lives only as `scripts/eval_colbert_rerank.py` on `explore/colbert-rerank`. To merge cleanly:

1. Create `src/tool_index/retrieval/rerank.py` with `rerank_tools(query_emb, candidate_ids, tool_vectors_by_id, k) -> list[str]`.
2. Add `precompute_tool_vectors(enrichments, embedder, cache) -> dict[str, list[list[float]]]` (one-time pre-embed of intent_phrase + example_queries per tool).
3. Extend `retrieve()` signature with optional `rerank_k`, `tool_vectors`; default `rerank_k=None` keeps existing behavior.
4. Extend `validation/recall_benchmark.py` to accept and thread `rerank_k`/`tool_vectors`.
5. Add the rerank knob to `Config` + `_pipeline_config.py` (default off so existing snapshots stay reproducible).
6. Add unit tests called out in `explore-direction1-colbert-leaf-rerank.steps[4]`.
7. Append D1 entry from `proposed_tasks.json` to `tasks.json` with `passes: true`.
8. Run full suite + `scripts/stage_validate.py --run raw-tools` with rerank on; assert recall >= 0.99.
9. Commit: "Promote Direction 1: ColBERT-style leaf reranker (recall 0.978 -> 0.994 stacked)".

### Step 3 — clean up

1. Move `relabel_low_discriminability.py` and `expand_queries.py` into `scripts/explore/` (they were spike-only and shouldn't pollute the main scripts dir).
2. Mark `explore/direction2_doc2query.json` with `"status": "archived"` and a `"result"` block citing the −0.150 regression.
3. Remove the four worktrees with `git worktree remove ../tds-{doc2query,multivector,colbert,stack}` once their branches are merged or explicitly archived.
4. Keep `explore/results.md` as the postmortem record on `main`.

## Risk / blast radius

- **D3 alone**: minimal — single file, defaults preserved, no schema or snapshot format changes. Old snapshots keep working unchanged.
- **D1 productization**: medium — touches `Config`, `recall_benchmark`, and adds a new module. Default `rerank_k=None` means recall numbers in old snapshots stay reproducible; opt-in only.
- **Latency cost of D1 in production**: top-2k traversal + 2k MaxSim ops per query at small N (~90 tools) is negligible; at 10k tools, depends on how many leaves the traverser surfaces — the rerank cost is `O(rerank_k * vectors_per_tool)` regardless of catalog size.
