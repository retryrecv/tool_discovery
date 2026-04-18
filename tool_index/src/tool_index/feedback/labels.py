"""Feedback dataclasses — what gets written to disk.

Polarity is the only thing the build pipeline (Phase 3) cares about.
The rest is provenance so a human can audit why a label was assigned.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

SCHEMA_VERSION = 1


class Polarity(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FeedbackLabel:
    polarity: Polarity
    confidence: float
    reason: str


@dataclass(frozen=True)
class FeedbackRecord:
    customer_id: str
    query: str
    routed_tool_id: str | None
    snapshot_version: str
    label: FeedbackLabel
    request_id: str
    session_id: str | None
    timestamp: str
    schema_version: int = SCHEMA_VERSION
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "customer_id": self.customer_id,
            "query": self.query,
            "routed_tool_id": self.routed_tool_id,
            "snapshot_version": self.snapshot_version,
            "label": {
                "polarity": self.label.polarity.value,
                "confidence": self.label.confidence,
                "reason": self.label.reason,
            },
            "request_id": self.request_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "schema_version": self.schema_version,
            "extra": self.extra,
        }
