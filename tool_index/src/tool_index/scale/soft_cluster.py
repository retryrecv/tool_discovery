"""Soft-attention clustering — HILL-inspired stage-3 alternative.

Replaces hard threshold assignment with a softmax over centroids, annealed
by a temperature schedule. Opt-in via ``SoftClusterConfig``; default stage 3
is unaffected.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SoftClusterConfig:
    n_clusters: int
    max_iters: int = 20
    max_alpha: float = 10.0
    alpha_exp: float = 2.0
    balance_weight: float = 0.1
    tol: float = 1e-4
    seed: int = 0


def _pairwise_sqdist(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    # (N, K)
    xn = (x * x).sum(axis=1, keepdims=True)
    cn = (c * c).sum(axis=1, keepdims=True).T
    return xn + cn - 2.0 * x @ c.T


def alpha_schedule(step: int, cfg: SoftClusterConfig) -> float:
    if cfg.max_iters <= 1:
        return cfg.max_alpha
    frac = step / max(cfg.max_iters - 1, 1)
    return cfg.max_alpha * (frac ** cfg.alpha_exp)


def assign_soft(
    embeddings: list[list[float]] | np.ndarray,
    centroids: list[list[float]] | np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (assignments [N,K], pseudo_centroids [N,D]).

    ``assignments[i,k] = softmax(-alpha * ||x_i - c_k||^2)``.
    ``pseudo_centroids[i] = sum_k a_ik c_k``.
    """
    x = np.asarray(embeddings, dtype=np.float64)
    c = np.asarray(centroids, dtype=np.float64)
    if x.ndim != 2 or c.ndim != 2:
        raise ValueError("embeddings and centroids must be 2-D")
    d = _pairwise_sqdist(x, c)
    logits = -alpha * d
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    a = e / e.sum(axis=1, keepdims=True)
    return a, a @ c


def assignment_entropy(a: np.ndarray) -> float:
    p = np.clip(a, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum(axis=1).mean())


def _init_centroids(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = x.shape[0]
    idx = rng.choice(n, size=min(k, n), replace=False)
    c = x[idx].copy()
    if c.shape[0] < k:
        pad = rng.normal(size=(k - c.shape[0], x.shape[1]))
        c = np.vstack([c, pad])
    return c


def soft_cluster(
    embeddings: list[list[float]] | np.ndarray,
    cfg: SoftClusterConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Run annealed soft assignment with FLOPs-style balance regularizer.

    Returns ``(hard_labels [N], centroids [K,D])``. Hard labels come from
    argmax of the final soft assignment (caller can still use the soft
    distribution by calling ``assign_soft`` with the returned centroids).
    """
    x = np.asarray(embeddings, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("embeddings must be 2-D")
    n = x.shape[0]
    k = min(cfg.n_clusters, n)
    rng = np.random.default_rng(cfg.seed)
    c = _init_centroids(x, k, rng)

    prev_obj = float("inf")
    for step in range(cfg.max_iters):
        alpha = alpha_schedule(step, cfg)
        a, _ = assign_soft(x, c, alpha)
        mass = a.sum(axis=0) + 1e-12           # (K,)
        # FLOPs-style balance term: penalize mass concentration (sum_k p_k^2).
        p = mass / mass.sum()
        balance_pen = float((p * p).sum()) * cfg.balance_weight
        # M-step: weighted centroid update; rebalance weights toward under-used
        # clusters to fight collapse (FLOPs-style regularizer effect).
        inv_mass = (mass.mean() / mass)  # boost low-mass clusters
        w = a * inv_mass[None, :] ** cfg.balance_weight
        wsum = w.sum(axis=0, keepdims=True).T + 1e-12
        c_new = (w.T @ x) / wsum
        shift = float(np.linalg.norm(c_new - c))
        c = c_new
        obj = shift + balance_pen
        if abs(prev_obj - obj) < cfg.tol:
            break
        prev_obj = obj

    a_final, _ = assign_soft(x, c, cfg.max_alpha)
    labels = a_final.argmax(axis=1).astype(np.int64)
    return labels, c
