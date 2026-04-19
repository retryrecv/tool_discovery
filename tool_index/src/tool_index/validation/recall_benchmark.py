"""Recall@k benchmark — does the traverser actually find the right tool?

Runs the top-down traverser against every synthetic query and measures
whether the expected ``tool_id`` appears in the top-k results. This is the
single number that tells us whether the index is fit for purpose; the
other validators check prerequisites.
"""
from __future__ import annotations

from ..schema import Tree
from ..retrieval import retrieve, rerank_tools


def run_retrieval_benchmark(
    tree: Tree,
    queries: list[dict],
    embedder,
    k: int,
    beam: int = 2,
    rerank_k: int | None = None,
    tool_vectors: dict[str, list[list[float]]] | None = None,
) -> float:
    """Compute recall@k on the synthetic eval set.

    Args:
        tree: The assembled tree.
        queries: ``{tool_id, query}`` rows from
            ``generate_synthetic_queries``.
        embedder: `EmbeddingProvider` used to embed each query at eval
            time. Must match the model that produced node embeddings —
            mixing models makes cosine scores meaningless.
        k: Top-k depth. Smaller ``k`` is a stricter test.
        beam: Branching factor for the top-down traverser. Higher widens
            the search at each level — more recall, more compute.
        rerank_k: When set, traverse for ``rerank_k`` candidates then
            rerank to ``k`` via ``tool_vectors``. Must be >= ``k``.
            Default ``None`` keeps existing behavior.
        tool_vectors: ``{tool_id: [vec, ...]}`` from
            ``retrieval.precompute_tool_vectors``. Required iff
            ``rerank_k`` is set.

    Returns:
        Fraction of queries whose gold ``tool_id`` appeared in the top-k.
        Empty query set returns ``1.0`` by convention (nothing failed).
    """
    if not queries:
        return 1.0
    pull = rerank_k if rerank_k is not None else k
    hits = 0
    for row in queries:
        q_emb = embedder.embed(row["query"])
        candidates = retrieve(tree, q_emb, k=pull, beam=beam)
        if rerank_k is not None and tool_vectors is not None:
            candidates = rerank_tools(q_emb, candidates, tool_vectors, k)
        if row["tool_id"] in candidates[:k]:
            hits += 1
    return hits / len(queries)
