"""Auto-tuner — small grid sweep over the knobs that move recall the most.

Searched dimensions (kept tiny on purpose; LLM cost is the bottleneck):
    thresholds["domain"]    in {0.05, 0.10, 0.15}
    thresholds["category"]  in {0.10, 0.15, 0.20}
    fanout["category"][1]   in {6, 10, 14}    (max categories per domain)

For each point we expect the caller to:
  1. produce a Tree (calling the build pipeline with the candidate config),
  2. score it against the held-out eval queries.

We don't run the pipeline ourselves — that lets the caller cache tree
artifacts and decide whether to skip points whose stages would
duplicate work that just ran. The caller passes a `score_fn` callback.

`positive_eval` rewards retrieving the labeled tool in top-k.
`negative_eval` penalizes retrieving the labeled tool in top-k.
The blended objective is `positive_hit_rate - lambda * negative_hit_rate`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .eval_adapter import EvalQuery


@dataclass(frozen=True)
class ConfigPoint:
    domain_threshold: float
    category_threshold: float
    category_max_fanout: int

    def label(self) -> str:
        return (
            f"dom={self.domain_threshold:.2f}"
            f"_cat={self.category_threshold:.2f}"
            f"_catmax={self.category_max_fanout}"
        )


@dataclass
class TuneResult:
    point: ConfigPoint
    objective: float
    positive_hit_rate: float
    negative_hit_rate: float
    sample_count: int
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "point": {
                "domain_threshold": self.point.domain_threshold,
                "category_threshold": self.point.category_threshold,
                "category_max_fanout": self.point.category_max_fanout,
            },
            "objective": self.objective,
            "positive_hit_rate": self.positive_hit_rate,
            "negative_hit_rate": self.negative_hit_rate,
            "sample_count": self.sample_count,
            "extra": self.extra,
        }


DEFAULT_GRID: list[ConfigPoint] = [
    ConfigPoint(d, c, m)
    for d in (0.05, 0.10, 0.15)
    for c in (0.10, 0.15, 0.20)
    for m in (6, 10, 14)
]


def score_eval(
    eval_set: list[EvalQuery],
    retrieve_fn: Callable[[str], list[str]],
) -> tuple[float, float, int]:
    """Run each query through `retrieve_fn` and compute blended hit rates.

    Returns (positive_hit_rate, negative_hit_rate, sample_count).
    """
    pos_total = neg_total = pos_hits = neg_hits = 0
    for q in eval_set:
        retrieved = set(retrieve_fn(q.query))
        if q.polarity == "positive":
            pos_total += 1
            if q.tool_id in retrieved:
                pos_hits += 1
        else:  # negative
            neg_total += 1
            if q.tool_id in retrieved:
                neg_hits += 1
    pos_rate = pos_hits / pos_total if pos_total else 1.0
    neg_rate = neg_hits / neg_total if neg_total else 0.0
    return pos_rate, neg_rate, pos_total + neg_total


def sweep_configs(
    grid: list[ConfigPoint],
    score_fn: Callable[[ConfigPoint], tuple[float, float, int]],
    *,
    negative_penalty: float = 0.5,
) -> list[TuneResult]:
    """Score every grid point via `score_fn` and return results sorted best→worst.

    `score_fn(point)` must return `(positive_hit_rate, negative_hit_rate, sample_count)`.
    Objective = positive_hit_rate - negative_penalty * negative_hit_rate.
    """
    results: list[TuneResult] = []
    for point in grid:
        pos, neg, n = score_fn(point)
        objective = pos - negative_penalty * neg
        results.append(TuneResult(
            point=point,
            objective=objective,
            positive_hit_rate=pos,
            negative_hit_rate=neg,
            sample_count=n,
        ))
    results.sort(key=lambda r: -r.objective)
    return results
