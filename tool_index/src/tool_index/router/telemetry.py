"""Append-only request/route telemetry.

One JSONL row per `/route` call, partitioned by UTC date so daily
batch jobs (Phase 3) can sweep yesterday's file as input. Schema is
flat on purpose — easier to tail, grep, and load into duckdb later.

Don't add fields without bumping `SCHEMA_VERSION`. Phase 2's feedback
sessionizer reads these rows and assumes the contract.
"""
from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .layout import CustomerLayout

SCHEMA_VERSION = 1


@dataclass
class RouteRecord:
    request_id: str
    customer_id: str
    snapshot_version: str
    query: str
    routed_tool_id: str | None
    path: list[str]
    node_scores: list[float]
    top_k_tool_ids: list[str]
    latency_ms: float
    timestamp: str
    session_id: str | None = None
    schema_version: int = SCHEMA_VERSION
    extra: dict = field(default_factory=dict)

    @classmethod
    def new(
        cls,
        *,
        customer_id: str,
        snapshot_version: str,
        query: str,
        routed_tool_id: str | None,
        path: list[str],
        node_scores: list[float],
        top_k_tool_ids: list[str],
        latency_ms: float,
        session_id: str | None = None,
        extra: dict | None = None,
    ) -> "RouteRecord":
        return cls(
            request_id=str(uuid.uuid4()),
            customer_id=customer_id,
            snapshot_version=snapshot_version,
            query=query,
            routed_tool_id=routed_tool_id,
            path=path,
            node_scores=node_scores,
            top_k_tool_ids=top_k_tool_ids,
            latency_ms=latency_ms,
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            session_id=session_id,
            extra=extra or {},
        )

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "customer_id": self.customer_id,
            "snapshot_version": self.snapshot_version,
            "query": self.query,
            "routed_tool_id": self.routed_tool_id,
            "path": self.path,
            "node_scores": self.node_scores,
            "top_k_tool_ids": self.top_k_tool_ids,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "schema_version": self.schema_version,
            "extra": self.extra,
        }


class RequestLogger:
    """Per-customer JSONL writer, thread-safe, daily-partitioned.

    Holds one open file handle per (customer, date). Rotates lazily on
    first write of a new UTC day. Safe to share across requests inside
    one process; for multi-process deployments use one logger per
    worker (each writes its own line atomically — JSONL tolerates
    interleaved appends from independent writers on POSIX).
    """

    def __init__(self, snapshots_root: str | Path):
        self.snapshots_root = Path(snapshots_root)
        self._handles: dict[tuple[str, str], object] = {}
        self._lock = threading.Lock()

    def log(self, record: RouteRecord) -> None:
        date_str = record.timestamp[:10]
        layout = CustomerLayout.for_customer(self.snapshots_root, record.customer_id)
        layout.ensure()
        path = layout.requests_path(date_str)
        line = json.dumps(record.to_dict()) + "\n"
        with self._lock:
            with path.open("a") as f:
                f.write(line)

    def close(self) -> None:
        with self._lock:
            self._handles.clear()
