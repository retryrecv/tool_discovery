"""Tree quality score — recall@k against a curated golden sample set.

Sample file format (one JSON object per line):
    {"query": "find files modified yesterday", "tool_id": "tool_xxx"}
    {"query": "...", "tool_ids": ["tool_a", "tool_b"]}   # multi-correct

`compute_quality_score` returns a single float in [0, 1] plus a
breakdown so the rollback gate can log *why* a snapshot regressed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..retrieval.traverser import retrieve
from ..schema import Tree


@dataclass
class QualityScore:
    score: float
    sample_count: int
    hits: int
    k: int
    misses: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "sample_count": self.sample_count,
            "hits": self.hits,
            "k": self.k,
            "misses": self.misses,
        }


def _load_samples(samples_path: str | Path) -> list[dict]:
    p = Path(samples_path)
    if not p.exists():
        return []
    out: list[dict] = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def compute_quality_score(
    tree: Tree,
    samples_path: str | Path,
    embedder,
    k: int = 10,
    beam: int = 2,
) -> QualityScore:
    """Run each sample query through the traverser, count hits in top-k.

    A sample hits if any of its expected tool IDs appears in the
    retrieved top-k. Empty sample file → score 1.0 (no evidence to fail
    on; first-time customers won't be blocked from promoting).
    """
    samples = _load_samples(samples_path)
    if not samples:
        return QualityScore(score=1.0, sample_count=0, hits=0, k=k)

    hits = 0
    misses: list[dict] = []
    for s in samples:
        expected = set(s.get("tool_ids") or [s["tool_id"]])
        q_emb = embedder.embed(s["query"])
        retrieved = retrieve(tree, q_emb, k=k, beam=beam)
        if expected & set(retrieved):
            hits += 1
        else:
            misses.append({
                "query": s["query"],
                "expected": sorted(expected),
                "retrieved_top_k": retrieved[:k],
            })

    return QualityScore(
        score=hits / len(samples),
        sample_count=len(samples),
        hits=hits,
        k=k,
        misses=misses,
    )
