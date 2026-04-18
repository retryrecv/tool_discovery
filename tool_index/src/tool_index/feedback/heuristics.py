"""Heuristics turning a session into per-record feedback labels.

Three signals, all derived from how the user/agent behaved AFTER a route:

1. **Retry**: a near-duplicate query came back within `retry_seconds`.
   Strong negative for the original route. Similarity is character-set
   Jaccard on lowercased tokens — cheap, no embedder dependency.

2. **Follow-up tool switch**: the next route in the session picks a
   DIFFERENT tool for a similar query. Negative for the first route,
   positive for the second.

3. **Abandonment**: the routed call has no follow-up activity in the
   session. Weak negative — could mean satisfied OR gave up. We label
   with low confidence so Phase 3 can choose to filter it out.

If none fire, label is POSITIVE with low confidence (presumed working).
The build pipeline should weight by `confidence`, not treat every label
as ground truth.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from .labels import FeedbackLabel, FeedbackRecord, Polarity
from .sessionize import Session, _parse_ts


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(s: str) -> set[str]:
    return set(_TOKEN_RE.findall(s.lower()))


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


@dataclass(frozen=True)
class Heuristics:
    retry_seconds: float = 60.0
    retry_similarity: float = 0.6
    followup_similarity: float = 0.4
    abandonment_confidence: float = 0.2
    presumed_positive_confidence: float = 0.3
    negative_confidence: float = 0.7


def label_session(session: Session, h: Heuristics | None = None) -> list[FeedbackRecord]:
    h = h or Heuristics()
    out: list[FeedbackRecord] = []
    recs = session.records

    for i, r in enumerate(recs):
        label = _classify(i, recs, h)
        out.append(FeedbackRecord(
            customer_id=r["customer_id"],
            query=r["query"],
            routed_tool_id=r.get("routed_tool_id"),
            snapshot_version=r.get("snapshot_version", ""),
            label=label,
            request_id=r["request_id"],
            session_id=session.session_id,
            timestamp=r["timestamp"],
        ))
    return out


def _classify(i: int, recs: list[dict], h: Heuristics) -> FeedbackLabel:
    cur = recs[i]
    has_next = i + 1 < len(recs)

    if has_next:
        nxt = recs[i + 1]
        gap = (_parse_ts(nxt["timestamp"]) - _parse_ts(cur["timestamp"])).total_seconds()
        sim = _jaccard(cur["query"], nxt["query"])

        if gap <= h.retry_seconds and sim >= h.retry_similarity:
            return FeedbackLabel(
                polarity=Polarity.NEGATIVE,
                confidence=h.negative_confidence,
                reason=f"retry within {gap:.0f}s, similarity={sim:.2f}",
            )

        if (
            sim >= h.followup_similarity
            and cur.get("routed_tool_id")
            and nxt.get("routed_tool_id")
            and cur["routed_tool_id"] != nxt["routed_tool_id"]
        ):
            return FeedbackLabel(
                polarity=Polarity.NEGATIVE,
                confidence=h.negative_confidence,
                reason=f"follow-up switched tool (sim={sim:.2f})",
            )

        return FeedbackLabel(
            polarity=Polarity.POSITIVE,
            confidence=h.presumed_positive_confidence,
            reason="follow-up activity, no negative signal",
        )

    return FeedbackLabel(
        polarity=Polarity.UNKNOWN,
        confidence=h.abandonment_confidence,
        reason="no follow-up in session (abandonment or satisfied)",
    )
