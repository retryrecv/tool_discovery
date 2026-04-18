from __future__ import annotations

import numpy as np

from tool_index.scale import build_neighbor_graph, NeighborConfig


def test_numpy_backend_returns_k_neighbors() -> None:
    rng = np.random.default_rng(0)
    embs = rng.normal(size=(20, 8)).tolist()
    out = build_neighbor_graph(embs, NeighborConfig(k=5, backend="numpy"))
    assert len(out) == 20
    for row in out:
        assert len(row) == 5
        for j, dist in row:
            assert 0 <= j < 20
            assert dist >= 0.0


def test_self_excluded() -> None:
    embs = np.eye(5).tolist()
    out = build_neighbor_graph(embs, NeighborConfig(k=4, backend="numpy"))
    for i, row in enumerate(out):
        assert all(j != i for j, _ in row)


def test_distances_sorted_ascending() -> None:
    rng = np.random.default_rng(1)
    embs = rng.normal(size=(15, 4)).tolist()
    out = build_neighbor_graph(embs, NeighborConfig(k=6, backend="numpy"))
    for row in out:
        dists = [d for _, d in row]
        assert dists == sorted(dists)


def test_auto_uses_numpy_for_small_input() -> None:
    embs = np.eye(10).tolist()
    out = build_neighbor_graph(embs, NeighborConfig(k=3, backend="auto"))
    assert len(out) == 10
