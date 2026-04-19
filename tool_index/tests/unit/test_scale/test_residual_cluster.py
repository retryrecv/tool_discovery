from __future__ import annotations

import numpy as np

from tool_index.scale import cluster_upward_residual, reconstruction_residuals


class _Node:
    def __init__(self, nid: str, embedding: list[float], parent_id: str | None = None) -> None:
        self.id = nid
        self.embedding = embedding
        self.parent_id = parent_id


def _kmeans2(n_clusters: int):
    def fn(embeddings: list[list[float]]) -> list[int]:
        rng = np.random.default_rng(0)
        x = np.asarray(embeddings, dtype=np.float64)
        idx = rng.choice(x.shape[0], size=n_clusters, replace=False)
        c = x[idx].copy()
        for _ in range(10):
            d = ((x[:, None, :] - c[None, :, :]) ** 2).sum(-1)
            labels = d.argmin(axis=1)
            for k in range(n_clusters):
                mask = labels == k
                if mask.any():
                    c[k] = x[mask].mean(axis=0)
        d = ((x[:, None, :] - c[None, :, :]) ** 2).sum(-1)
        return d.argmin(axis=1).tolist()
    return fn


def test_no_parent_matches_raw_clustering() -> None:
    nodes = [
        _Node("a", [1.0, 0.0]),
        _Node("b", [0.9, 0.1]),
        _Node("c", [5.0, 5.0]),
        _Node("d", [4.9, 5.1]),
    ]
    fn = _kmeans2(2)
    residual_labels = cluster_upward_residual(nodes, {}, fn)
    raw_labels = fn([n.embedding for n in nodes])
    assert residual_labels == raw_labels


def test_residual_subtraction_centers_children() -> None:
    # Two parents; children are offsets from parent centroids.
    parents = {"p1": [10.0, 0.0], "p2": [0.0, 10.0]}
    nodes = [
        _Node("a", [11.0, 0.0], parent_id="p1"),
        _Node("b", [9.5, 0.2], parent_id="p1"),
        _Node("c", [0.2, 11.0], parent_id="p2"),
        _Node("d", [-0.1, 9.8], parent_id="p2"),
    ]
    captured: dict = {}

    def capture(embs):
        captured["embs"] = embs
        return [0] * len(embs)

    cluster_upward_residual(nodes, parents, capture)
    embs = np.array(captured["embs"])
    # After subtraction, all children should be near origin.
    assert np.all(np.linalg.norm(embs, axis=1) < 2.0)


def test_reconstruction_residuals_zero_for_perfect_path() -> None:
    leaf_emb = {"leaf": [3.0, 4.0]}
    centroids = {"n1": [1.0, 1.0], "n2": [2.0, 3.0]}
    paths = {"leaf": ["n1", "n2"]}
    out = reconstruction_residuals(leaf_emb, centroids, paths)
    assert out["leaf"] == 0.0


def test_reconstruction_residuals_nonzero_for_mismatch() -> None:
    leaf_emb = {"leaf": [10.0, 0.0]}
    centroids = {"n1": [1.0, 0.0]}
    paths = {"leaf": ["n1"]}
    out = reconstruction_residuals(leaf_emb, centroids, paths)
    assert out["leaf"] == 9.0
