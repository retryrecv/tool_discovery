from __future__ import annotations

import numpy as np

from tool_index.scale import EMRefineConfig, build_kmeans_centroids, em_refine


def _blobs(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=[5.0, 0.0], scale=0.1, size=(30, 2))
    b = rng.normal(loc=[-5.0, 0.0], scale=0.1, size=(30, 2))
    c = rng.normal(loc=[0.0, 5.0], scale=0.1, size=(30, 2))
    return np.vstack([a, b, c])


def test_kmeans_numpy_converges() -> None:
    x = _blobs()
    cfg = EMRefineConfig(n_clusters=3, n_iter=20, backend="numpy", seed=0)
    labels, centroids, wcss = build_kmeans_centroids(x, cfg)
    assert labels.shape == (90,)
    assert centroids.shape == (3, 2)
    assert wcss[-1] <= wcss[0]


def test_wcss_decreases_monotonically() -> None:
    x = _blobs(seed=1)
    cfg = EMRefineConfig(n_clusters=3, n_iter=15, backend="numpy", seed=0)
    _, _, wcss = build_kmeans_centroids(x, cfg)
    for i in range(1, len(wcss)):
        assert wcss[i] <= wcss[i - 1] + 1e-6


def test_em_refine_disabled_zero_iter() -> None:
    x = _blobs()
    init = np.array([[5.0, 0.0], [-5.0, 0.0], [0.0, 5.0]])
    labels_a, centroids_a, hist_a = em_refine(x, init, n_iter=0)
    assert len(hist_a) == 1
    np.testing.assert_allclose(centroids_a, init)


def test_em_refine_improves_wcss() -> None:
    x = _blobs()
    init = np.array([[4.0, 0.5], [-4.0, -0.5], [0.5, 4.0]])
    _, _, hist_0 = em_refine(x, init, n_iter=0)
    _, _, hist_k = em_refine(x, init, n_iter=10)
    assert hist_k[-1] < hist_0[0]


def test_auto_backend_prefers_numpy_for_small_input() -> None:
    x = _blobs()
    cfg = EMRefineConfig(n_clusters=3, n_iter=10, backend="auto", seed=0)
    labels, centroids, _ = build_kmeans_centroids(x, cfg)
    assert labels.shape == (90,)
    assert centroids.shape == (3, 2)
