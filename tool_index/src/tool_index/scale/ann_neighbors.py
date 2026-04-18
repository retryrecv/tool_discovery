"""ANN-backed neighbor graph for stage 3.

Pairwise cosine on 10k × 10k vectors is 100M ops per matmul plus the
sort. Using HNSW (via `faiss` if available, else `hnswlib`, else a
NumPy fallback that's still O(n²) but vectorized) we keep clustering
sub-second.

The `build_neighbor_graph` function returns a list of `(neighbor_idx,
distance)` lists per item, using **cosine distance** (1 - cos_sim) so
it plugs straight into the existing agglomerative clusterer that
expects distances.

Optional deps: install `faiss-cpu` for the fast path. We don't add it
to pyproject because it's a heavy build dep; document with `uv add
faiss-cpu` instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class NeighborConfig:
    k: int = 16
    backend: str = "auto"  # auto | faiss | hnswlib | numpy


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def _numpy_knn(mat: np.ndarray, k: int) -> list[list[tuple[int, float]]]:
    n = mat.shape[0]
    sims = mat @ mat.T  # cosine similarity (mat is unit-norm)
    np.fill_diagonal(sims, -np.inf)
    k_eff = min(k, n - 1)
    if k_eff <= 0:
        return [[] for _ in range(n)]
    idx = np.argpartition(-sims, kth=k_eff - 1, axis=1)[:, :k_eff]
    out: list[list[tuple[int, float]]] = []
    for i in range(n):
        cols = idx[i]
        ranked = sorted(((int(j), float(1.0 - sims[i, j])) for j in cols), key=lambda x: x[1])
        out.append(ranked)
    return out


def _faiss_knn(mat: np.ndarray, k: int):
    import faiss  # type: ignore

    d = mat.shape[1]
    n = mat.shape[0]
    index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 64
    index.hnsw.efSearch = 64
    index.add(mat.astype(np.float32))
    k_eff = min(k + 1, n)
    sims, idx = index.search(mat.astype(np.float32), k_eff)
    out: list[list[tuple[int, float]]] = []
    for i in range(n):
        row = []
        for j_pos in range(k_eff):
            j = int(idx[i, j_pos])
            if j == -1 or j == i:
                continue
            row.append((j, float(1.0 - sims[i, j_pos])))
            if len(row) >= k:
                break
        out.append(row)
    return out


def _hnswlib_knn(mat: np.ndarray, k: int):
    import hnswlib  # type: ignore

    n, d = mat.shape
    p = hnswlib.Index(space="cosine", dim=d)
    p.init_index(max_elements=n, ef_construction=64, M=32)
    p.add_items(mat, np.arange(n))
    p.set_ef(max(64, k + 1))
    labels, dists = p.knn_query(mat, k=min(k + 1, n))
    out: list[list[tuple[int, float]]] = []
    for i in range(n):
        row = []
        for j_pos in range(labels.shape[1]):
            j = int(labels[i, j_pos])
            if j == i:
                continue
            row.append((j, float(dists[i, j_pos])))
            if len(row) >= k:
                break
        out.append(row)
    return out


def build_neighbor_graph(
    embeddings: Sequence[Sequence[float]],
    config: NeighborConfig | None = None,
) -> list[list[tuple[int, float]]]:
    """Return per-item k-NN as `[(neighbor_idx, cosine_distance), ...]`.

    `backend="auto"` picks faiss → hnswlib → numpy in order of availability.
    Below ~2k items the numpy fallback is fast enough that ANN overhead
    isn't worth it; the auto path uses numpy for small inputs.
    """
    cfg = config or NeighborConfig()
    mat = _l2_normalize(np.asarray(embeddings, dtype=np.float64))

    if cfg.backend == "numpy":
        return _numpy_knn(mat, cfg.k)
    if cfg.backend == "faiss":
        return _faiss_knn(mat, cfg.k)
    if cfg.backend == "hnswlib":
        return _hnswlib_knn(mat, cfg.k)

    if mat.shape[0] < 2000:
        return _numpy_knn(mat, cfg.k)
    try:
        return _faiss_knn(mat, cfg.k)
    except ImportError:
        pass
    try:
        return _hnswlib_knn(mat, cfg.k)
    except ImportError:
        pass
    return _numpy_knn(mat, cfg.k)
