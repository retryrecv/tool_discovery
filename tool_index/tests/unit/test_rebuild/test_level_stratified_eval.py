from __future__ import annotations

import json
from pathlib import Path

from tool_index.rebuild.eval_adapter import build_level_stratified_eval
from tool_index.router import CustomerLayout


def _row(query, polarity, conf=0.7, tool="t1", level=0):
    r = {
        "customer_id": "acme",
        "query": query,
        "routed_tool_id": tool,
        "snapshot_version": "v0",
        "request_id": query,
        "session_id": "s",
        "timestamp": "2026-04-17T10:00:00Z",
        "label": {"polarity": polarity, "confidence": conf, "reason": "x"},
    }
    if level:
        r["extra"] = {"level": level, "harvested": True}
    return r


def test_groups_by_level(tmp_path: Path) -> None:
    layout = CustomerLayout.for_customer(tmp_path, "acme")
    fb = layout.root / "feedback" / "2026-04-17.jsonl"
    fb.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        _row("q1", "positive", level=0),  # leaf-only label
        _row("q2", "positive", level=1),
        _row("q3", "positive", level=2),
        _row("q4", "positive", level=2),
        _row("q5", "positive", level=3),
    ]
    fb.write_text("\n".join(json.dumps(r) for r in rows))

    out = build_level_stratified_eval(tmp_path, "acme")
    assert set(out.keys()) == {0, 1, 2, 3}
    assert len(out[2]) == 2
    assert {q.query for q in out[1]} == {"q2"}


def test_low_conf_filtered(tmp_path: Path) -> None:
    layout = CustomerLayout.for_customer(tmp_path, "acme")
    fb = layout.root / "feedback" / "2026-04-17.jsonl"
    fb.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        _row("q1", "positive", conf=0.9, level=1),
        _row("q2", "positive", conf=0.2, level=1),  # below floor
    ]
    fb.write_text("\n".join(json.dumps(r) for r in rows))

    out = build_level_stratified_eval(tmp_path, "acme", min_confidence=0.5)
    assert {q.query for q in out.get(1, [])} == {"q1"}
