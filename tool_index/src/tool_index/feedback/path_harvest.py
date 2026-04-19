"""Beam-path feedback harvesting — HILL Section 3.5.

When a leaf label is POSITIVE, emit one auxiliary ``FeedbackRecord`` per
(level, node) on the recorded route path, with a graduated confidence
floor per level (HILL's phi_DEP / phi_IR strictness ramp).

Additive: callers opt in by invoking ``harvest_path_labels`` after
``label_session``. Existing heuristics output is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

from .labels import FeedbackLabel, FeedbackRecord, Polarity


@dataclass(frozen=True)
class PathHarvestConfig:
    # Graduated confidence floors per level (HILL-style strictness ramp).
    level_floor: tuple[float, ...] = (0.4, 0.6, 0.8)
    enabled: bool = True


def harvest_path_labels(
    records: list[FeedbackRecord],
    request_paths: dict[str, list[str]],
    cfg: PathHarvestConfig | None = None,
) -> list[FeedbackRecord]:
    """Expand POSITIVE leaf labels into per-(level, node) records.

    Args:
        records: Output of ``label_session`` for a session.
        request_paths: Map ``request_id`` → path list (root-descended node ids).
        cfg: Level floors and enable flag.

    For a positive leaf at request_id ``r`` with path ``[L1, L2, L3]``,
    emit three records keyed by level index in ``extra``: ``level`` field
    = 1..N. Confidence = ``max(original_conf, level_floor[level - 1])``.
    Non-positive records are passed through unchanged.
    """
    cfg = cfg or PathHarvestConfig()
    if not cfg.enabled:
        return list(records)

    out: list[FeedbackRecord] = []
    for rec in records:
        out.append(rec)
        if rec.label.polarity is not Polarity.POSITIVE:
            continue
        path = request_paths.get(rec.request_id) or []
        for idx, node_id in enumerate(path):
            level = idx + 1
            floor = cfg.level_floor[min(level - 1, len(cfg.level_floor) - 1)]
            conf = max(rec.label.confidence, floor)
            out.append(FeedbackRecord(
                customer_id=rec.customer_id,
                query=rec.query,
                routed_tool_id=rec.routed_tool_id,
                snapshot_version=rec.snapshot_version,
                label=FeedbackLabel(
                    polarity=Polarity.POSITIVE,
                    confidence=conf,
                    reason=f"path-harvest L{level}",
                ),
                request_id=rec.request_id,
                session_id=rec.session_id,
                timestamp=rec.timestamp,
                extra={"level": level, "node_id": node_id, "harvested": True},
            ))
    return out
