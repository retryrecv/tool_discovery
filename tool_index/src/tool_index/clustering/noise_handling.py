"""Noise-point attachment for density clustering.

HDBSCAN treats isolated points as "noise" rather than forcing them into
a cluster. For our tree, every tool must live somewhere, so we reattach
noise points to whichever existing cluster's centroid is nearest.
"""
from __future__ import annotations
import numpy as np


def attach_noise(
    clusters: list[list[int]],
    embeddings: list[list[float]],
    noise_indices: list[int],
) -> list[list[int]]:
    """Append each noise point to the cluster with the nearest centroid.

    Args:
        clusters: Existing clusters, mutated in place — a noise index is
            appended to the chosen cluster's member list.
        embeddings: The embedding matrix. Indices in ``noise_indices`` and
            cluster member lists both point into this.
        noise_indices: Points not yet assigned to any cluster.

    Returns:
        The (same) ``clusters`` list, for chaining.
    """
    if not noise_indices:
        return clusters
    X = np.array(embeddings, dtype=np.float64)
    for i in noise_indices:
        # Linear scan — fine at catalog scale. For large k we'd index
        # centroids in a k-d tree.
        best, best_d = 0, float("inf")
        for k, c in enumerate(clusters):
            cen = X[c].mean(axis=0)
            d = float(np.linalg.norm(X[i] - cen))
            if d < best_d:
                best_d = d
                best = k
        clusters[best].append(i)
    return clusters
