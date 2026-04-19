"""Residual upward clustering — HILL cross-layer residual learning.

Stage 4 clusters on ``embedding - parent_centroid`` so deeper levels
capture finer detail. Opt-in via ``ResidualClusterConfig``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class ResidualClusterConfig:
    enabled: bool = False


def _as_np(v: Sequence[float]) -> np.ndarray:
    return np.asarray(v, dtype=np.float64)


def cluster_upward_residual(
    nodes: list,
    parent_centroids: dict[str, list[float]],
    cluster_fn: Callable[[list[list[float]]], list[int]],
) -> list[int]:
    """Cluster ``nodes`` after subtracting each node's parent centroid.

    Args:
        nodes: Iterable with ``embedding`` and ``parent_id`` attributes.
        parent_centroids: Map from ``parent_id`` → centroid embedding.
        cluster_fn: Clustering function taking residual embeddings list
            and returning integer labels aligned with ``nodes``.

    Nodes whose parent is not in ``parent_centroids`` are clustered on
    their raw embedding (no residual subtraction) — this matches the
    ``cluster_upward(raw)`` baseline so that disabling residuals reduces
    to the existing behavior.
    """
    residuals: list[list[float]] = []
    for n in nodes:
        emb = _as_np(n.embedding)
        pid = getattr(n, "parent_id", None)
        if pid and pid in parent_centroids:
            emb = emb - _as_np(parent_centroids[pid])
        residuals.append(emb.tolist())
    return cluster_fn(residuals)


def reconstruction_residuals(
    leaf_embeddings: dict[str, list[float]],
    centroids_by_node: dict[str, list[float]],
    paths: dict[str, list[str]],
) -> dict[str, float]:
    """Return ||leaf - sum(path centroids)|| per leaf.

    Used as a sanity invariant: a well-built residual hierarchy should
    sum to something close to the leaf embedding.
    """
    out: dict[str, float] = {}
    for leaf_id, emb in leaf_embeddings.items():
        path = paths.get(leaf_id, [])
        s = np.zeros_like(_as_np(emb))
        for nid in path:
            c = centroids_by_node.get(nid)
            if c is None:
                continue
            s = s + _as_np(c)
        out[leaf_id] = float(np.linalg.norm(_as_np(emb) - s))
    return out
