"""End-to-end: read a day's request log, sessionize, label, write feedback.

Used by `scripts/process_feedback.py`. Importable so future stream
processors can drive the same path on micro-batches.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..router.layout import CustomerLayout
from .heuristics import Heuristics, label_session
from .labels import FeedbackRecord
from .sessionize import sessionize
from .writer import FeedbackWriter


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def process_day(
    snapshots_root: str | Path,
    customer_id: str,
    date_str: str,
    *,
    heuristics: Heuristics | None = None,
    gap_seconds: float = 300.0,
    overwrite: bool = True,
) -> dict:
    layout = CustomerLayout.for_customer(snapshots_root, customer_id)
    requests = _read_jsonl(layout.requests_path(date_str))
    if not requests:
        return {"customer_id": customer_id, "date": date_str, "requests": 0, "labels": 0, "files": {}}

    all_labels: list[FeedbackRecord] = []
    sessions = 0
    for sess in sessionize(requests, gap_seconds=gap_seconds):
        sessions += 1
        all_labels.extend(label_session(sess, heuristics))

    writer = FeedbackWriter(snapshots_root)
    files = writer.write(all_labels, overwrite=overwrite)
    return {
        "customer_id": customer_id,
        "date": date_str,
        "requests": len(requests),
        "sessions": sessions,
        "labels": len(all_labels),
        "files": files,
    }
