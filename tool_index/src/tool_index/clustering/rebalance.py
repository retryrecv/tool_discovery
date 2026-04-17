"""Rebalance clusters so they fit the configured per-level fanout window.

Raw clustering output almost never honors both ``min_size`` and
``max_size``. This module applies two deterministic post-processing passes:

1. **Split** any cluster larger than ``max_size`` by median distance to its
   centroid. Repeats until every cluster fits.
2. **Merge** each undersized cluster into its nearest neighbor that has
   room. Single sweep — if a cluster still can't be merged, we leave it
   alone (a rare case; the structural validator will flag it).

No randomness; order of operations is stable w.r.t. input list order.
"""
from __future__ import annotations
import numpy as np

from .agglomerative import _cosine_dist_matrix  # re-export for other modules if needed


def _centroid(X: np.ndarray, cluster: list[int]) -> np.ndarray:
    """Unweighted mean embedding of ``cluster``'s members."""
    return X[cluster].mean(axis=0)


def rebalance_clusters(
    clusters: list[list[int]],
    embeddings: list[list[float]],
    min_size: int,
    max_size: int,
) -> list[list[int]]:
    """Apply split-then-merge to bring clusters inside ``[min_size, max_size]``.

    Args:
        clusters: Clustering output (list of index lists).
        embeddings: Shared embedding matrix — clusters index into it.
        min_size: Target lower bound on cluster size. Clusters below this
            are merged into a neighbor if possible.
        max_size: Hard upper bound. Clusters above this are split.

    Returns:
        A new list of clusters. Input is not mutated beyond possible
        in-place ``append`` during merge.
    """
    if not clusters:
        return clusters
    X = np.array(embeddings, dtype=np.float64)

    # ---- Split oversized clusters by median distance to centroid.
    # Keep looping because a split half may itself still exceed max_size
    # (uncommon, but possible when max_size is tight).
    changed = True
    while changed:
        changed = False
        out: list[list[int]] = []
        for c in clusters:
            if len(c) > max_size:
                cen = _centroid(X, c)
                dists = [(float(np.linalg.norm(X[i] - cen)), i) for i in c]
                dists.sort()
                # Split at the median — the half closest to centroid goes
                # in one cluster, the far half in the other. Deterministic
                # because `sorted` is stable.
                half = len(dists) // 2
                a = [i for _, i in dists[:half]]
                b = [i for _, i in dists[half:]]
                out.append(a)
                out.append(b)
                changed = True
            else:
                out.append(c)
        clusters = out

    # ---- Merge undersized clusters into nearest neighbor that has room.
    while True:
        # Pick the first undersized cluster each iteration so merges don't
        # skip anyone. Only merge if there's more than one cluster left —
        # otherwise "undersized" is unavoidable.
        idx = next((k for k, c in enumerate(clusters) if len(c) < min_size and len(clusters) > 1), None)
        if idx is None:
            break
        c = clusters[idx]
        cen_c = _centroid(X, c)

        best_j = None
        best_d = float("inf")
        for j, other in enumerate(clusters):
            if j == idx:
                continue
            # Skip candidates that would overflow after the merge —
            # re-splitting them would undo this work.
            if len(other) + len(c) > max_size:
                continue
            cen_o = _centroid(X, other)
            d = float(np.linalg.norm(cen_c - cen_o))
            if d < best_d:
                best_d = d
                best_j = j
        if best_j is None:
            # No viable neighbor — accept the undersized cluster; the
            # structural validator will surface it as a warning.
            break
        clusters[best_j] = clusters[best_j] + c
        clusters.pop(idx)
    return clusters
