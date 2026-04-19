"""EM FAISS K-Means refinement — HILL Section 3.6.

Treats indices as Gaussian-mixture clusters; alternates
    E-step: reassign each point to nearest current centroid
    M-step: recompute centroid as mean of assigned points
Iterates to convergence or ``max_iter``.

FAISS is used when available; a NumPy fallback matches the interface for
small inputs and tests.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EMRefineConfig:
    n_clusters: int
    n_iter: int = 10
    backend: str = "auto"     # "auto" | "numpy" | "faiss"
    tol: float = 1e-6
    seed: int = 0


def _init_centroids(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.choice(x.shape[0], size=min(k, x.shape[0]), replace=False)
    c = x[idx].copy()
    if c.shape[0] < k:
        pad = rng.normal(size=(k - c.shape[0], x.shape[1]))
        c = np.vstack([c, pad])
    return c


def _assign(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    xn = (x * x).sum(axis=1, keepdims=True)
    cn = (c * c).sum(axis=1, keepdims=True).T
    d = xn + cn - 2.0 * x @ c.T
    return d.argmin(axis=1)


def _wcss(x: np.ndarray, c: np.ndarray, labels: np.ndarray) -> float:
    diffs = x - c[labels]
    return float((diffs * diffs).sum())


def _faiss_available() -> bool:
    try:
        import faiss  # noqa: F401
        return True
    except ImportError:
        return False


def _numpy_kmeans(
    x: np.ndarray, cfg: EMRefineConfig
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    rng = np.random.default_rng(cfg.seed)
    c = _init_centroids(x, cfg.n_clusters, rng)
    wcss_hist: list[float] = []
    labels = _assign(x, c)
    wcss_hist.append(_wcss(x, c, labels))
    for _ in range(cfg.n_iter):
        new_c = np.zeros_like(c)
        for k in range(c.shape[0]):
            mask = labels == k
            if mask.any():
                new_c[k] = x[mask].mean(axis=0)
            else:
                new_c[k] = c[k]
        shift = float(np.linalg.norm(new_c - c))
        c = new_c
        labels = _assign(x, c)
        wcss_hist.append(_wcss(x, c, labels))
        if shift < cfg.tol:
            break
    return labels, c, wcss_hist


def _faiss_kmeans(
    x: np.ndarray, cfg: EMRefineConfig
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    import faiss
    d = x.shape[1]
    km = faiss.Kmeans(d, cfg.n_clusters, niter=cfg.n_iter, seed=cfg.seed, verbose=False)
    km.train(x.astype("float32"))
    c = np.asarray(km.centroids, dtype=np.float64)
    labels = _assign(x, c)
    wcss_hist = [_wcss(x, c, labels)]
    return labels, c, wcss_hist


def build_kmeans_centroids(
    embeddings: list[list[float]] | np.ndarray,
    cfg: EMRefineConfig,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Returns ``(labels, centroids, wcss_history)``."""
    x = np.asarray(embeddings, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("embeddings must be 2-D")
    backend = cfg.backend
    if backend == "auto":
        backend = "faiss" if (x.shape[0] >= 1024 and _faiss_available()) else "numpy"
    if backend == "faiss":
        return _faiss_kmeans(x, cfg)
    return _numpy_kmeans(x, cfg)


def em_refine(
    embeddings: list[list[float]] | np.ndarray,
    init_centroids: list[list[float]] | np.ndarray,
    n_iter: int,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Refine ``init_centroids`` with ``n_iter`` EM steps.

    Disabled when ``n_iter <= 0`` — returns the initial assignment unchanged.
    """
    x = np.asarray(embeddings, dtype=np.float64)
    c = np.asarray(init_centroids, dtype=np.float64).copy()
    labels = _assign(x, c)
    hist = [_wcss(x, c, labels)]
    if n_iter <= 0:
        return labels, c, hist
    for _ in range(n_iter):
        new_c = np.zeros_like(c)
        for k in range(c.shape[0]):
            mask = labels == k
            if mask.any():
                new_c[k] = x[mask].mean(axis=0)
            else:
                new_c[k] = c[k]
        shift = float(np.linalg.norm(new_c - c))
        c = new_c
        labels = _assign(x, c)
        hist.append(_wcss(x, c, labels))
        if shift < tol:
            break
    return labels, c, hist
