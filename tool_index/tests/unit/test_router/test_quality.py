from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from tool_index.router.quality import compute_quality_score
from tool_index.schema import Node, Tree


@dataclass
class _StubEmbedder:
    table: dict
    def embed(self, text: str):
        return self.table.get(text, [0.0, 0.0, 1.0])


def _two_tool_tree() -> Tree:
    g1 = Node(id="g1", level="group", description="files", embedding=[1.0, 0.0, 0.0],
              children=["tool_ls", "tool_find"])
    g2 = Node(id="g2", level="group", description="math", embedding=[0.0, 1.0, 0.0],
              children=["tool_add"])
    root = Node(id="root", level="root", description="all", embedding=[0.0, 0.0, 1.0],
                children=["g1", "g2"])
    t = Tree(root=root)
    for n in (root, g1, g2):
        t.register(n)
    return t


def test_empty_samples_returns_one(tmp_path: Path) -> None:
    tree = _two_tool_tree()
    samples = tmp_path / "samples.jsonl"
    samples.write_text("")
    emb = _StubEmbedder({})
    q = compute_quality_score(tree, samples, emb, k=2, beam=2)
    assert q.score == 1.0
    assert q.sample_count == 0


def test_hits_counted(tmp_path: Path) -> None:
    tree = _two_tool_tree()
    samples = tmp_path / "samples.jsonl"
    samples.write_text(
        json.dumps({"query": "list files", "tool_id": "tool_ls"}) + "\n"
        + json.dumps({"query": "do math", "tool_id": "tool_add"}) + "\n"
        + json.dumps({"query": "missing topic", "tool_id": "tool_unknown"}) + "\n"
    )
    emb = _StubEmbedder({
        "list files": [1.0, 0.0, 0.0],
        "do math":    [0.0, 1.0, 0.0],
        "missing topic": [1.0, 0.0, 0.0],
    })
    q = compute_quality_score(tree, samples, emb, k=2, beam=2)
    assert q.sample_count == 3
    assert q.hits == 2
    assert abs(q.score - 2 / 3) < 1e-9
    assert len(q.misses) == 1
    assert q.misses[0]["query"] == "missing topic"
