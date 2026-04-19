"""ColBERT-style leaf reranker.

After the top-down traverser surfaces a candidate pool of tools, rescore
each candidate by MaxSim between the query and a per-tool multi-vector
representation (intent_phrase + each example_query, embedded once).
Re-orders within the pool — never adds tools the traverser didn't surface.

Reference: Khattab & Zaharia 2020, "ColBERT: Efficient and Effective
Passage Search via Contextualized Late Interaction over BERT"
(arXiv:2004.12832).
"""
from __future__ import annotations
import numpy as np

from ..schema import Enrichment


def _cos(a, b) -> float:
    va, vb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    return float(va @ vb / (na * nb)) if na and nb else 0.0


def precompute_tool_vectors(
    enrichments: dict[str, Enrichment],
    embedder,
) -> dict[str, list[list[float]]]:
    """Embed each tool's intent_phrase + example_queries once.

    Returns ``{tool_id: [vec, vec, ...]}``. Reuses the embedder's cache
    via its provider — re-runs on the same texts are free.
    """
    out: dict[str, list[list[float]]] = {}
    for tid, enr in enrichments.items():
        texts = [enr.intent_phrase] + list(enr.example_queries)
        out[tid] = embedder.embed_batch(texts)
    return out


def rerank_tools(
    query_embedding: list[float],
    candidates: list[str],
    tool_vectors: dict[str, list[list[float]]],
    k: int,
) -> list[str]:
    """Reorder ``candidates`` by MaxSim against per-tool multi-vectors.

    Tools with no entry in ``tool_vectors`` get score 0.0 — they end up
    last but are not dropped from the pool.
    """
    scored: list[tuple[float, str]] = []
    for tid in candidates:
        vecs = tool_vectors.get(tid, [])
        score = max((_cos(query_embedding, v) for v in vecs), default=0.0)
        scored.append((score, tid))
    scored.sort(key=lambda x: -x[0])
    return [tid for _, tid in scored[:k]]
