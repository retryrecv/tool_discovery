"""Snapshot read/write — the persistence boundary.

A frozen snapshot is a directory with four files:

    tree.json            — full graph + tool descriptors (see `Tree.to_dict`)
    embeddings.json      — node_id → embedding vector (flat JSON)
    build_trace.json     — how the snapshot was produced (see `BuildTrace`)
    seed_eval_set.jsonl  — one query per line for the recall benchmark

The embeddings are split out of ``tree.json`` so the graph file stays
diff-friendly (embeddings change every run, structure changes rarely).
"""
from __future__ import annotations
import json
from pathlib import Path

from ..schema import Tree


def write_immutable_snapshot(tree: Tree, out_dir: str | Path) -> Path:
    """Write all four snapshot files into ``out_dir``.

    Args:
        tree: Tree to serialize. Must have `build_trace` populated —
            stage 6 does this before calling us.
        out_dir: Target directory. Created if missing. Caller (stage 6)
            guarantees it's a fresh version slot, so overwrite safety
            isn't checked here.

    Returns:
        ``out_dir`` as a `Path`, for chaining.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Graph + tools — indented for human readability in diffs.
    (out / "tree.json").write_text(json.dumps(tree.to_dict(), indent=2))

    # Embeddings serialized as JSON array map. Keeps dep footprint tiny
    # (no parquet/numpy-on-disk); fine up to ~10k nodes × ~1024 dims.
    emb = {nid: list(n.embedding) for nid, n in tree.nodes_by_id.items()}
    (out / "embeddings.json").write_text(json.dumps(emb))

    bt = tree.build_trace.to_dict()
    (out / "build_trace.json").write_text(json.dumps(bt, indent=2))

    # Eval set as JSONL (one row per line) so readers can stream it
    # without holding the full list in memory.
    seed = bt.get("seed_eval_set", [])
    with (out / "seed_eval_set.jsonl").open("w") as f:
        for row in seed:
            f.write(json.dumps(row) + "\n")
    return out


def load_snapshot(in_dir: str | Path) -> Tree:
    """Rehydrate a `Tree` from a snapshot directory.

    Only reads ``tree.json`` — the other files are inspected by tooling
    but don't affect the graph itself. Embeddings are carried inside
    ``tree.json``'s node entries already.
    """
    p = Path(in_dir)
    data = json.loads((p / "tree.json").read_text())
    return Tree.from_dict(data)
