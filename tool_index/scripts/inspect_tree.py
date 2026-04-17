#!/usr/bin/env python
"""Pretty-print a frozen snapshot for debugging.

Usage:
    python scripts/inspect_tree.py <snapshot_dir>

Prints the tree structure (level, description, per-node inner/leaf
counts) so you can eyeball whether the shape and labels look right. Not
part of the package API — standalone script, run directly.
"""
from __future__ import annotations
import sys
from pathlib import Path

from tool_index.storage import load_snapshot


def main(argv: list[str]) -> int:
    """Entry point. Returns exit code (2 for usage error, 0 for success)."""
    if len(argv) < 2:
        print("usage: inspect_tree.py <snapshot_dir>", file=sys.stderr)
        return 2
    tree = load_snapshot(argv[1])
    print(f"version: {tree.version}")
    print(f"tools:   {len(tree.tools_by_id)}")
    print(f"depth:   {tree.depth()}")

    def walk(node, indent: int = 0):
        """Recursively print this node and descend into inner children.

        Leaf children (tool IDs) are counted but not printed individually —
        otherwise large trees produce unreadable output. Inspect tool
        leaves via the snapshot's ``tree.json`` directly if needed.
        """
        prefix = "  " * indent
        inner = [cid for cid in node.children if cid in tree.nodes_by_id]
        leaf_count = len([cid for cid in node.children if cid not in tree.nodes_by_id])
        tag = f"[{node.level}]"
        print(f"{prefix}{tag} {node.description}  (inner={len(inner)}, leaves={leaf_count})")
        for cid in inner:
            walk(tree.nodes_by_id[cid], indent + 1)

    walk(tree.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
