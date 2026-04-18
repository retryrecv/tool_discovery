"""Session reconstruction from raw route records.

A session is a contiguous run of requests from the same customer with
the same `session_id`, OR — when `session_id` is missing — a run of
requests from the same customer where each consecutive pair is within
`gap_seconds` of the other (default 300s = 5 minutes).

Records are sorted by timestamp before grouping; the input file is
expected to be append-only and roughly time-ordered already, but we
don't trust it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Iterator


def _parse_ts(s: str) -> datetime:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


@dataclass
class Session:
    customer_id: str
    session_id: str | None
    records: list[dict] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.records)


def sessionize(
    records: Iterable[dict],
    *,
    gap_seconds: float = 300.0,
) -> Iterator[Session]:
    """Yield `Session` objects from a stream of route records.

    Records with explicit `session_id` are grouped strictly by that ID.
    Records without one are grouped per-customer by time gap.
    """
    by_customer: dict[str, list[dict]] = {}
    for r in records:
        by_customer.setdefault(r["customer_id"], []).append(r)

    for customer_id, recs in by_customer.items():
        recs.sort(key=lambda r: r["timestamp"])

        explicit: dict[str, list[dict]] = {}
        implicit: list[dict] = []
        for r in recs:
            sid = r.get("session_id")
            if sid:
                explicit.setdefault(sid, []).append(r)
            else:
                implicit.append(r)

        for sid, group in explicit.items():
            yield Session(customer_id=customer_id, session_id=sid, records=group)

        if not implicit:
            continue
        cur: list[dict] = [implicit[0]]
        for prev, this in zip(implicit, implicit[1:]):
            gap = (_parse_ts(this["timestamp"]) - _parse_ts(prev["timestamp"])).total_seconds()
            if gap <= gap_seconds:
                cur.append(this)
            else:
                yield Session(customer_id=customer_id, session_id=None, records=cur)
                cur = [this]
        yield Session(customer_id=customer_id, session_id=None, records=cur)
