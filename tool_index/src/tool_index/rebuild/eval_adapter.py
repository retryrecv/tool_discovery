"""Convert feedback labels into evaluation queries for autotuning.

Source: `data/snapshots/<customer>/feedback/*.jsonl` from Phase 2.

Rules:
    - POSITIVE label, confidence ≥ min_confidence  →  expected tool is the routed one
    - NEGATIVE label                               →  expected tool is NOT the routed one
                                                      (we don't know what IS correct, so we
                                                      use it as an anti-example: any retrieval
                                                      that re-routes to the same tool is wrong)
    - UNKNOWN                                      →  dropped (too noisy to learn from)

We deduplicate by query (last write wins) so a query that flipped from
NEGATIVE → POSITIVE later in the window is treated as POSITIVE. Honest
to the user's most recent observed behavior.

The output format mirrors `samples.jsonl` so `router.quality.compute_quality_score`
can score it directly — but with a `polarity` field added so the
autotuner knows how to read each row.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..router.layout import CustomerLayout


@dataclass(frozen=True)
class EvalQuery:
    query: str
    tool_id: str
    polarity: str  # "positive" or "negative"
    confidence: float

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "tool_id": self.tool_id,
            "polarity": self.polarity,
            "confidence": self.confidence,
        }


def _iter_feedback(layout: CustomerLayout, days: list[str] | None) -> Iterable[dict]:
    fb_dir = layout.root / "feedback"
    if not fb_dir.exists():
        return
    if days is None:
        files = sorted(fb_dir.glob("*.jsonl"))
    else:
        files = [fb_dir / f"{d}.jsonl" for d in days]
    for f in files:
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_eval_set(
    snapshots_root: str | Path,
    customer_id: str,
    *,
    days: list[str] | None = None,
    min_confidence: float = 0.5,
) -> list[EvalQuery]:
    layout = CustomerLayout.for_customer(snapshots_root, customer_id)
    by_query: dict[str, EvalQuery] = {}
    for row in _iter_feedback(layout, days):
        label = row["label"]
        polarity = label["polarity"]
        if polarity == "unknown":
            continue
        if label["confidence"] < min_confidence:
            continue
        tool = row.get("routed_tool_id")
        if not tool:
            continue
        eq = EvalQuery(
            query=row["query"],
            tool_id=tool,
            polarity=polarity,
            confidence=label["confidence"],
        )
        by_query[eq.query] = eq
    return list(by_query.values())
