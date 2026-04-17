# clustering

Embedding-space clustering used in stages 3 and 4.

- `hdbscan_runner.py` — HDBSCAN-style density clustering for leaves.
- `agglomerative.py` — agglomerative clustering for upward levels (L3 → L2 → L1).
- `neighbors.py` — k-NN graph construction over embeddings.
- `noise_handling.py` — policy for HDBSCAN "noise" points (reassign vs. keep as singletons).
- `rebalance.py` — split oversized clusters, merge undersized ones to hit target fanout.

## Conventions

- Clusterers take `np.ndarray` embeddings + params, return `list[list[int]]` (cluster → member indices).
- No LLM calls here — descriptions come later in `labeling/`.
- Target fanout lives in config, not hardcoded. Typical: 5–15 children per node.
