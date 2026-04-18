"""Sampled discriminability — cap O(siblings²) judge calls.

Stage 5's discriminability check asks the judge LLM to compare every
pair of sibling nodes under each parent. With 20 siblings that's 190
calls per parent; with 50 it's 1225. At 10k tools you cannot afford
this.

`sample_pairs` returns at most `max_pairs` deterministically-chosen
pairs per parent, prioritizing pairs whose embeddings are closest
(most likely to be confused, so most useful to validate).

Determinism is critical: a re-run on the same tree must score the same
pairs, otherwise `quality_score` drifts. We sort by (cosine distance,
id_a, id_b) before truncation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SampleStrategy:
    max_pairs: int = 30
    closest_first: bool = True


def _cos_dist(a, b) -> float:
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0 or nb == 0:
        return 1.0
    return float(1.0 - va @ vb / (na * nb))


def sample_pairs(
    siblings: list,
    *,
    strategy: SampleStrategy | None = None,
    embedding_attr: str = "embedding",
    id_attr: str = "id",
) -> list[tuple[object, object]]:
    """Return up to `max_pairs` (a, b) sibling pairs to send to the judge.

    `siblings` is any sequence of objects that have an embedding and an
    id attribute (Nodes, descriptors, dicts via duck typing won't work
    — use a wrapper).

    Behaviour:
        - Generates the full O(n²) pair list once, scores by cosine
          distance, sorts ascending (closest pairs first since they're
          the riskiest), then truncates.
        - If `closest_first=False`, samples evenly across the distance
          range (every k-th element after sort) — useful when you want
          coverage instead of focus.
    """
    s = strategy or SampleStrategy()
    n = len(siblings)
    if n < 2:
        return []

    pairs: list[tuple[float, str, str, object, object]] = []
    for i in range(n):
        ei = getattr(siblings[i], embedding_attr)
        ii = getattr(siblings[i], id_attr)
        for j in range(i + 1, n):
            ej = getattr(siblings[j], embedding_attr)
            ij = getattr(siblings[j], id_attr)
            pairs.append((_cos_dist(ei, ej), ii, ij, siblings[i], siblings[j]))

    pairs.sort(key=lambda p: (p[0], p[1], p[2]))
    if len(pairs) <= s.max_pairs:
        chosen = pairs
    elif s.closest_first:
        chosen = pairs[: s.max_pairs]
    else:
        step = max(1, len(pairs) // s.max_pairs)
        chosen = pairs[::step][: s.max_pairs]
    return [(p[3], p[4]) for p in chosen]
