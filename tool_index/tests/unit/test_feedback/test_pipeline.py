from __future__ import annotations

import json
from pathlib import Path

from tool_index.feedback import process_day
from tool_index.router import CustomerLayout
from tool_index.router.telemetry import RequestLogger, RouteRecord


def _seed(snapshots: Path, customer: str) -> None:
    logger = RequestLogger(snapshots)
    base_ts = "2026-04-17T10:00:0"
    rows = [
        ("0Z", "list files",     "t1"),
        ("5Z", "list all files", "t2"),
        ("9Z", "convert pdf",    "t3"),
    ]
    for tail, q, tool in rows:
        rec = RouteRecord.new(
            customer_id=customer, snapshot_version="v0",
            query=q, routed_tool_id=tool, path=[],
            node_scores=[], top_k_tool_ids=[tool],
            latency_ms=1.0, session_id="s1",
        )
        object.__setattr__(rec, "timestamp", base_ts + tail)
        logger.log(rec)


def test_process_day_writes_feedback(tmp_path: Path) -> None:
    _seed(tmp_path, "acme")
    summary = process_day(tmp_path, "acme", "2026-04-17")
    assert summary["requests"] == 3
    assert summary["labels"] == 3

    layout = CustomerLayout.for_customer(tmp_path, "acme")
    fb_path = layout.root / "feedback" / "2026-04-17.jsonl"
    assert fb_path.exists()
    rows = [json.loads(l) for l in fb_path.read_text().splitlines() if l.strip()]
    assert len(rows) == 3
    polarities = [r["label"]["polarity"] for r in rows]
    assert "negative" in polarities


def test_process_day_overwrites(tmp_path: Path) -> None:
    _seed(tmp_path, "acme")
    process_day(tmp_path, "acme", "2026-04-17")
    process_day(tmp_path, "acme", "2026-04-17")
    layout = CustomerLayout.for_customer(tmp_path, "acme")
    fb_path = layout.root / "feedback" / "2026-04-17.jsonl"
    rows = [l for l in fb_path.read_text().splitlines() if l.strip()]
    assert len(rows) == 3


def test_no_requests_returns_empty_summary(tmp_path: Path) -> None:
    summary = process_day(tmp_path, "ghost", "2026-04-17")
    assert summary["requests"] == 0
    assert summary["labels"] == 0
