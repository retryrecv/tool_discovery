"""Agglomerative (single-linkage) clustering on cosine distance.

Used by stage 4 (upward clustering) directly, and by stage 3's HDBSCAN
adapter as its underlying mechanism. Deterministic — no randomness, no
ordering-dependent tie-breaks beyond sort stability.
"""
from __future__ import annotations
import numpy as np


def _cosine_dist_matrix(X: np.ndarray) -> np.ndarray:
    """Dense n×n cosine-distance matrix.

    L2-normalizes rows first so the matrix is correct even if the caller
    passes non-normalized vectors (e.g. when the embedder doesn't
    normalize). Clipped to ``[-1, 1]`` to keep downstream ``arccos``-ish
    math safe against float drift.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms
    sim = Xn @ Xn.T
    sim = np.clip(sim, -1.0, 1.0)
    return 1.0 - sim


def agglomerative_cluster(
    embeddings: list[list[float]],
    distance_threshold: float,
    max_cluster_size: int | None = None,
) -> list[list[int]]:
    """Greedy agglomerative clustering with a cosine-distance cutoff.

    Single-linkage-ish: we accept any merge whose endpoint pair is within
    ``distance_threshold``, processed in ascending distance order via a
    disjoint-set union-find. Fine up to ~10k items — the O(n²) pair list
    becomes the bottleneck beyond that.

    Args:
        embeddings: Row-per-item embedding matrix (as nested lists).
        distance_threshold: Maximum cosine distance between any two items
            that can end up in the same cluster via the first merge step.
        max_cluster_size: If set, merges that would exceed this size are
            skipped — keeps clusters inside the configured fanout bound.

    Returns:
        List of clusters, each a list of input indices. Order of
        clusters and of members within each is stable w.r.t. input order.
    """
    n = len(embeddings)
    if n == 0:
        return []
    X = np.array(embeddings, dtype=np.float64)
    D = _cosine_dist_matrix(X)

    # Union-find representative array — each item starts as its own cluster.
    parent = list(range(n))

    def find(i: int) -> int:
        """Path-compressing union-find lookup."""
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # Upper-triangular candidate pairs under the threshold, sorted ascending.
    # Sorting by distance gives single-linkage-ish behavior: closest pairs
    # merge first, which defines each cluster's "core".
    iu = np.triu_indices(n, k=1)
    pairs = [(D[i, j], int(i), int(j)) for i, j in zip(*iu) if D[i, j] <= distance_threshold]
    pairs.sort()

    # Per-cluster size, keyed by the current root representative. Used to
    # enforce ``max_cluster_size`` without scanning every member on merge.
    size = {i: 1 for i in range(n)}

    for d, i, j in pairs:
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        new_size = size[ri] + size[rj]
        if max_cluster_size is not None and new_size > max_cluster_size:
            # Skipping leaves both endpoints in their current clusters —
            # they may still merge indirectly via other pairs, or stay
            # separate. Either way: deterministic.
            continue
        parent[ri] = rj
        size[rj] = new_size
        size.pop(ri, None)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        r = find(i)
        clusters.setdefault(r, []).append(i)
    return list(clusters.values())
