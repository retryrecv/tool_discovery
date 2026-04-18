"""Append-only feedback writer + per-customer layout helpers.

Layout (extends router.layout):
    data/snapshots/<customer>/feedback/<YYYY-MM-DD>.jsonl

Idempotency: callers should pass `process_day(..., overwrite=True)` for
deterministic batch reprocessing. The default is append, suitable for
streaming use later.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Iterable

from ..router.layout import CustomerLayout
from .labels import FeedbackRecord


def feedback_path(layout: CustomerLayout, date_str: str) -> Path:
    return layout.root / "feedback" / f"{date_str}.jsonl"


class FeedbackWriter:
    def __init__(self, snapshots_root: str | Path):
        self.snapshots_root = Path(snapshots_root)
        self._lock = threading.Lock()

    def write(self, records: Iterable[FeedbackRecord], *, overwrite: bool = False) -> dict[str, int]:
        counts: dict[str, int] = {}
        per_target: dict[Path, list[FeedbackRecord]] = {}
        for r in records:
            layout = CustomerLayout.for_customer(self.snapshots_root, r.customer_id)
            (layout.root / "feedback").mkdir(parents=True, exist_ok=True)
            date_str = r.timestamp[:10]
            per_target.setdefault(feedback_path(layout, date_str), []).append(r)

        with self._lock:
            for path, batch in per_target.items():
                mode = "w" if overwrite else "a"
                with path.open(mode) as f:
                    for r in batch:
                        f.write(json.dumps(r.to_dict()) + "\n")
                key = f"{path.parent.parent.name}/{path.name}"
                counts[key] = counts.get(key, 0) + len(batch)
        return counts
