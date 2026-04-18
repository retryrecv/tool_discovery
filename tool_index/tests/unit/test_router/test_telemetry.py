from __future__ import annotations

import json
from pathlib import Path

from tool_index.router import CustomerLayout
from tool_index.router.telemetry import RequestLogger, RouteRecord


def test_record_round_trip() -> None:
    rec = RouteRecord.new(
        customer_id="acme",
        snapshot_version="v3",
        query="how do I list files",
        routed_tool_id="tool_ls",
        path=["dom_a", "cat_b", "grp_c"],
        node_scores=[0.9, 0.8, 0.7],
        top_k_tool_ids=["tool_ls", "tool_find"],
        latency_ms=12.5,
        session_id="s-1",
    )
    d = rec.to_dict()
    assert d["customer_id"] == "acme"
    assert d["routed_tool_id"] == "tool_ls"
    assert d["schema_version"] == 1
    assert d["request_id"]
    assert d["timestamp"].endswith("Z")


def test_logger_appends_jsonl(tmp_path: Path) -> None:
    logger = RequestLogger(tmp_path)
    rec = RouteRecord.new(
        customer_id="acme",
        snapshot_version="v0",
        query="q",
        routed_tool_id="t1",
        path=[],
        node_scores=[],
        top_k_tool_ids=["t1"],
        latency_ms=1.0,
    )
    logger.log(rec)
    logger.log(rec)
    layout = CustomerLayout.for_customer(tmp_path, "acme")
    date_str = rec.timestamp[:10]
    p = layout.requests_path(date_str)
    lines = p.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["customer_id"] == "acme"
