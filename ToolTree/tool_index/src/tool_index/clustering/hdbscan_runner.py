"""HDBSCAN-style density clustering adapter for leaf tools.

Real HDBSCAN has a heavy numerical dependency; we approximate it with
agglomerative clustering at a fixed distance threshold, which is good
enough at tool-catalog scale (~10k items) and keeps the dependency
footprint tiny. Singletons below ``min_cluster_size`` are treated as
"noise" and reattached by `noise_handling.attach_noise`.
"""
from __future__ import annotations
from .agglomerative import agglomerative_cluster


def hdbscan_cluster(
    embeddings: list[list[float]],
    min_cluster_size: int = 2,
    max_cluster_size: int | None = None,
    distance_threshold: float = 0.55,
) -> list[list[int]]:
    """Cluster leaves by density proxy.

    Args:
        embeddings: Row-per-item embeddings.
        min_cluster_size: Minimum members for a "real" cluster. Smaller
            clusters are emitted as-is here; noise handling is the caller's
            responsibility (see ``noise_handling.attach_noise``).
        max_cluster_size: Optional ceiling enforced during merging.
        distance_threshold: Cosine-distance cutoff used by the underlying
            agglomerative run. Typical stage-3 value: 0.3.

    Returns:
        List of clusters as member-index lists. May contain singletons
        when the distance cutoff leaves items isolated.
    """
    return agglomerative_cluster(embeddings, distance_threshold, max_cluster_size)
