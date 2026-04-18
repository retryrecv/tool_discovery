from __future__ import annotations

import json
from pathlib import Path

from tool_index.rebuild import diff_snapshots


def _write(path: Path, version: str, nodes: dict, tools: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "tree.json").write_text(json.dumps({
        "version": version,
        "root_id": "root",
        "nodes": nodes,
        "tools": tools,
        "build_trace": {},
    }))


def test_added_removed_relabeled(tmp_path: Path) -> None:
    a = tmp_path / "v0"
    b = tmp_path / "v1"
    _write(
        a, "v0",
        nodes={"n1": {"description": "old"}, "n2": {"description": "stable"}},
        tools={"t1": {}, "t2": {}},
    )
    _write(
        b, "v1",
        nodes={
            "n1": {"description": "renamed"},
            "n2": {"description": "stable"},
            "n3": {"description": "new"},
        },
        tools={"t2": {}, "t3": {}},
    )
    d = diff_snapshots(a, b)
    assert d.added_nodes == ["n3"]
    assert d.removed_nodes == []
    assert len(d.relabeled_nodes) == 1
    assert d.relabeled_nodes[0]["id"] == "n1"
    assert d.added_tools == ["t3"]
    assert d.removed_tools == ["t1"]
