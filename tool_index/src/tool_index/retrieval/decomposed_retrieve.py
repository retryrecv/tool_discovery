"""Fan-out retrieval over pre-computed sub-queries.

Takes a list of sub-queries (produced upstream by ``decompose.decompose_query``
or any other splitter), runs the existing retrieve+rerank pipeline once per
sub-query, then unions the per-sub-query top-k pools by max-score-per-tool.

Strict parameter-passing: this module does NOT call the decomposer, embedder,
or LLM. The caller wires those together and passes the results in. Keeps
each stage independently testable.
"""
from __future__ import annotations

from ..schema import Tree
from .traverser import retrieve
from .rerank import rerank_tools


def retrieve_decomposed(
    tree: Tree,
    sub_query_embeddings: list[list[float]],
    tool_vectors: dict[str, list[list[float]]],
    *,
    k: int,
    rerank_k: int,
    beam: int,
) -> list[str]:
    """Union-rerank top-k tools across multiple sub-queries.

    Algorithm:
        1. For each sub-query embedding, run ``retrieve`` -> top ``rerank_k``
           candidates -> ``rerank_tools`` -> top ``k`` per sub-query.
        2. Pool all per-sub-query results, scoring each tool by its best
           rank across sub-queries (rank 1 = best score, rank k = worst).
        3. Tools that appear in multiple sub-queries are kept at their
           best rank — no double-counting.
        4. Return the global top ``k`` by aggregated score.

    A single-element ``sub_query_embeddings`` list reduces this to the
    ordinary retrieve+rerank path; behaviour is identical to the
    non-decomposed pipeline in that case.
    """
    if not sub_query_embeddings:
        return []

    # Per-sub-query scoring: rank position -> (k - rank) so higher = better.
    # A tool's final score is the max across all sub-queries.
    pool: dict[str, float] = {}
    for q_emb in sub_query_embeddings:
        candidates = retrieve(tree, q_emb, k=rerank_k, beam=beam)
        ranked = rerank_tools(q_emb, candidates, tool_vectors, k=k)
        for rank, tid in enumerate(ranked):
            score = float(k - rank)
            if score > pool.get(tid, 0.0):
                pool[tid] = score

    return [tid for tid, _ in sorted(pool.items(), key=lambda x: -x[1])[:k]]
