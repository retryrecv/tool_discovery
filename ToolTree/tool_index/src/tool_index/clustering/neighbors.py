"""Nearest-neighbor lookup between clusters for contrastive labeling.

The labeler (``labeling/describe.py``) produces better, more
discriminative descriptions when shown what a cluster is *not* — the few
nearest sibling clusters. This module computes those neighbors by centroid
Euclidean distance. Euclidean (rather than cosine) is fine here because
we're comparing centroids of already-normalized vectors.
"""
from __future__ import annotations
import numpy as np


def nearest_clusters(
    target: list[int],
    clusters: list[list[int]],
    embeddings: list[list[float]],
    k: int = 3,
) -> list[int]:
    """Return indices of the ``k`` nearest clusters to ``target`` (excluding itself).

    Args:
        target: The member-index list of the cluster we want neighbors for.
        clusters: Full list of clusters, including ``target``. Identity
            (``is``) comparison excludes the target from its own
            neighbor list — callers must pass the same list object they
            originally received.
        embeddings: The embedding matrix used to build ``clusters``.
        k: Maximum neighbors to return. Fewer if there are fewer clusters.

    Returns:
        Indices into ``clusters``, sorted nearest-first.
    """
    X = np.array(embeddings, dtype=np.float64)
    # Centroid = unweighted mean of member embeddings.
    cen_t = X[target].mean(axis=0)
    ranked: list[tuple[float, int]] = []
    for j, other in enumerate(clusters):
        # Identity check — we want to exclude the target itself, not any
        # other cluster that happens to have the same member list.
        if other is target:
            continue
        cen_o = X[other].mean(axis=0)
        d = float(np.linalg.norm(cen_t - cen_o))
        ranked.append((d, j))
    ranked.sort()
    return [j for _, j in ranked[:k]]
