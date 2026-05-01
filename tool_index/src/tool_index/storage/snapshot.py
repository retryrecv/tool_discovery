"""Snapshot read/write — the persistence boundary.

A frozen snapshot is a directory with three files:

    tree.json            — full graph + tool descriptors (see `Tree.to_dict`)
    embeddings.json      — node_id → embedding vector (flat JSON)
    build_trace.json     — how the snapshot was produced (see `BuildTrace`)

The embeddings are split out of ``tree.json`` so the graph file stays
diff-friendly (embeddings change every run, structure changes rarely).
"""
from __future__ import annotations
import json
from pathlib import Path

from ..schema import Tree


def write_immutable_snapshot(tree: Tree, out_dir: str | Path) -> Path:
    """Write all snapshot files into ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "tree.json").write_text(json.dumps(tree.to_dict(), indent=2))

    emb = {nid: list(n.embedding) for nid, n in tree.nodes_by_id.items()}
    (out / "embeddings.json").write_text(json.dumps(emb))

    bt = tree.build_trace.to_dict()
    (out / "build_trace.json").write_text(json.dumps(bt, indent=2))
    return out


def load_snapshot(in_dir: str | Path) -> Tree:
    """Rehydrate a `Tree` from a snapshot directory."""
    p = Path(in_dir)
    data = json.loads((p / "tree.json").read_text())
    return Tree.from_dict(data)

