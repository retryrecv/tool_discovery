"""Recall@k benchmark — does the traverser actually find the right tool?

Runs the top-down traverser against every synthetic query and measures
whether the expected ``tool_id`` appears in the top-k results. This is the
single number that tells us whether the index is fit for purpose; the
other validators check prerequisites.
"""
from __future__ import annotations

from ..schema import Tree
from ..retrieval import retrieve


def run_retrieval_benchmark(tree: Tree, queries: list[dict], embedder, k: int) -> float:
    """Compute recall@k on the synthetic eval set.

    Args:
        tree: The assembled tree.
        queries: ``{tool_id, query}`` rows from
            ``generate_synthetic_queries``.
        embedder: `EmbeddingProvider` used to embed each query at eval
            time. Must match the model that produced node embeddings —
            mixing models makes cosine scores meaningless.
        k: Top-k depth. Smaller ``k`` is a stricter test.

    Returns:
        Fraction of queries whose gold ``tool_id`` appeared in the top-k.
        Empty query set returns ``1.0`` by convention (nothing failed).
    """
    if not queries:
        return 1.0
    hits = 0
    for row in queries:
        q_emb = embedder.embed(row["query"])
        candidates = retrieve(tree, q_emb, k=k)
        if row["tool_id"] in candidates:
            hits += 1
    return hits / len(queries)
