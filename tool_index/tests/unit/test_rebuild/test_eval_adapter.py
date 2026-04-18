from __future__ import annotations

import json
from pathlib import Path

from tool_index.rebuild import build_eval_set
from tool_index.router import CustomerLayout


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows))


def _row(query, polarity, conf=0.7, tool="t1"):
    return {
        "customer_id": "acme",
        "query": query,
        "routed_tool_id": tool,
        "snapshot_version": "v0",
        "request_id": query,
        "session_id": "s",
        "timestamp": "2026-04-17T10:00:00Z",
        "label": {"polarity": polarity, "confidence": conf, "reason": "x"},
    }


def test_filters_unknown_and_low_conf(tmp_path: Path) -> None:
    layout = CustomerLayout.for_customer(tmp_path, "acme")
    _write(layout.root / "feedback" / "2026-04-17.jsonl", [
        _row("q1", "positive", conf=0.9),
        _row("q2", "unknown", conf=0.9),
        _row("q3", "positive", conf=0.1),
        _row("q4", "negative", conf=0.8),
    ])
    out = build_eval_set(tmp_path, "acme", min_confidence=0.5)
    qs = {q.query: q.polarity for q in out}
    assert qs == {"q1": "positive", "q4": "negative"}


def test_dedupes_by_query_keeping_last(tmp_path: Path) -> None:
    layout = CustomerLayout.for_customer(tmp_path, "acme")
    _write(layout.root / "feedback" / "2026-04-17.jsonl", [
        _row("same", "negative"),
    ])
    _write(layout.root / "feedback" / "2026-04-18.jsonl", [
        _row("same", "positive"),
    ])
    out = build_eval_set(tmp_path, "acme")
    assert len(out) == 1
    assert out[0].polarity == "positive"
