# Explore: recall-improvement spikes

Self-contained exploration files, one per direction. Not part of `tasks.json` — these are spikes, not commitments.

| File | Direction | Paper | Cost | Expected gain |
|---|---|---|---|---|
| `direction1_colbert_rerank.json` | Late-interaction reranker over top-2k leaves | ColBERT (arXiv:2004.12832) | medium (extra embeddings) | +0.02–0.05 |
| `direction2_doc2query.json` | Expand each tool's text with N predicted queries before embedding | Doc2Query (1904.08375) / HyDE (2212.10496) | low (cached LLM calls) | +0.03–0.06 |
| `direction3_multivector_nodes.json` | M centroids per inner node, MaxSim scoring at traversal | ColBERT + HILL §3.2 soft-cluster | medium (rebuild stage 4) | +0.04–0.08 |

## Workflow per direction

1. Read the JSON file — it has hypothesis, baseline, design, measurement plan, decision rule.
2. Implement on a branch; do not edit `tasks.json`.
3. Run the ablation listed under `measurement.ablation`.
4. If `decision_rule` is met → promote to a real entry in `tasks.json` and merge. Otherwise → set `status: "archived"` with a `result` field and leave the file as a record.

## Recommended order

Direction 2 first (cheapest, no tree changes), then Direction 3 (highest ceiling, reuses existing `scale/soft_cluster.py`), then Direction 1 (reranker, stacks on whichever wins).
