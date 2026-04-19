"""Phase 2 — implicit feedback from router telemetry.

Reads the JSONL request logs written by `router.telemetry`, groups them
into sessions, applies heuristics to label each routed call as
positive / negative / unknown, and writes per-customer feedback files.

Modules:
    sessionize   — group route records into sessions
    heuristics   — retry / follow-up / abandonment label rules
    labels       — label dataclasses
    writer       — append-only JSONL writer for feedback
    pipeline     — read logs → sessionize → label → write (the script entry)

These labels are NOT consumed yet by the build pipeline; that's Phase 3.
For now we just collect signal so by the time auto-tuning lands there's
already weeks of data to learn from.
"""
from .labels import FeedbackLabel, FeedbackRecord, Polarity
from .sessionize import sessionize, Session
from .heuristics import label_session, Heuristics
from .writer import FeedbackWriter
from .pipeline import process_day
from .path_harvest import harvest_path_labels, PathHarvestConfig

__all__ = [
    "FeedbackLabel", "FeedbackRecord", "Polarity",
    "sessionize", "Session",
    "label_session", "Heuristics",
    "FeedbackWriter",
    "process_day",
    "harvest_path_labels", "PathHarvestConfig",
]
