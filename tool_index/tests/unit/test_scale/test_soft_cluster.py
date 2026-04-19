from __future__ import annotations

import numpy as np

from tool_index.scale import (
    SoftClusterConfig,
    alpha_schedule,
    assign_soft,
    assignment_entropy,
    soft_cluster,
)


def _blob_embeddings(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=[5.0, 0.0], scale=0.1, size=(20, 2))
    b = rng.normal(loc=[0.0, 5.0], scale=0.1, size=(20, 2))
    c = rng.normal(loc=[-5.0, -5.0], scale=0.1, size=(20, 2))
    return np.vstack([a, b, c])


def test_assign_soft_rows_sum_to_one() -> None:
    x = _blob_embeddings()
    c = np.array([[5.0, 0.0], [0.0, 5.0], [-5.0, -5.0]])
    a, pseudo = assign_soft(x, c, alpha=1.0)
    assert a.shape == (60, 3)
    assert np.allclose(a.sum(axis=1), 1.0)
    assert pseudo.shape == (60, 2)


def test_alpha_schedule_monotonic() -> None:
    cfg = SoftClusterConfig(n_clusters=3, max_iters=5, max_alpha=10.0, alpha_exp=2.0)
    vals = [alpha_schedule(i, cfg) for i in range(cfg.max_iters)]
    assert vals == sorted(vals)
    assert vals[-1] == cfg.max_alpha


def test_converges_to_hard_clusters_on_clean_blobs() -> None:
    x = _blob_embeddings()
    cfg = SoftClusterConfig(n_clusters=3, max_iters=30, max_alpha=50.0, balance_weight=0.0)
    labels, centroids = soft_cluster(x, cfg)
    # Each true blob should collapse to a single assigned cluster.
    for start in (0, 20, 40):
        assert len(set(labels[start:start + 20].tolist())) == 1


def test_low_alpha_high_entropy() -> None:
    x = _blob_embeddings()
    c = np.array([[5.0, 0.0], [0.0, 5.0], [-5.0, -5.0]])
    a_low, _ = assign_soft(x, c, alpha=0.0001)
    a_high, _ = assign_soft(x, c, alpha=50.0)
    assert assignment_entropy(a_low) > assignment_entropy(a_high)
    # Near-uniform: log(3) ~ 1.0986.
    assert assignment_entropy(a_low) > 1.0


def test_balance_regularizer_reduces_stddev() -> None:
    rng = np.random.default_rng(0)
    # Imbalanced init: 80 near one center, 10 at each of the other two.
    a = rng.normal(loc=[5.0, 0.0], scale=0.1, size=(80, 2))
    b = rng.normal(loc=[0.0, 5.0], scale=0.1, size=(10, 2))
    c = rng.normal(loc=[-5.0, -5.0], scale=0.1, size=(10, 2))
    x = np.vstack([a, b, c])

    cfg0 = SoftClusterConfig(n_clusters=3, max_iters=15, max_alpha=5.0, balance_weight=0.0)
    labels0, _ = soft_cluster(x, cfg0)
    std0 = float(np.std(np.bincount(labels0, minlength=3)))

    cfg1 = SoftClusterConfig(n_clusters=3, max_iters=15, max_alpha=5.0, balance_weight=1.0)
    labels1, _ = soft_cluster(x, cfg1)
    std1 = float(np.std(np.bincount(labels1, minlength=3)))

    assert std1 <= std0
