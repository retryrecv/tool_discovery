"""Diff two snapshot trees — what nodes appeared/disappeared/got relabeled.

Operates on the JSON returned by `Tree.to_dict()`. Identity is by node
ID; "relabeled" means the same ID has a different `description`.

Useful as a daily-rebuild summary post: a 5-node delta is a quiet day,
500 nodes mean the catalog or thresholds shifted hard.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class SnapshotDiff:
    prev_version: str
    next_version: str
    added_nodes: list[str] = field(default_factory=list)
    removed_nodes: list[str] = field(default_factory=list)
    relabeled_nodes: list[dict] = field(default_factory=list)
    added_tools: list[str] = field(default_factory=list)
    removed_tools: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "prev_version": self.prev_version,
            "next_version": self.next_version,
            "added_nodes": len(self.added_nodes),
            "removed_nodes": len(self.removed_nodes),
            "relabeled_nodes": len(self.relabeled_nodes),
            "added_tools": len(self.added_tools),
            "removed_tools": len(self.removed_tools),
        }

    def to_dict(self) -> dict:
        return {
            **self.summary(),
            "added_node_ids": self.added_nodes,
            "removed_node_ids": self.removed_nodes,
            "relabeled": self.relabeled_nodes,
            "added_tool_ids": self.added_tools,
            "removed_tool_ids": self.removed_tools,
        }


def _read_tree_json(version_dir: Path) -> dict:
    return json.loads((version_dir / "tree.json").read_text())


def diff_snapshots(prev_dir: str | Path, next_dir: str | Path) -> SnapshotDiff:
    prev = _read_tree_json(Path(prev_dir))
    nxt = _read_tree_json(Path(next_dir))

    prev_nodes = prev.get("nodes", {})
    next_nodes = nxt.get("nodes", {})
    prev_tools = prev.get("tools", {})
    next_tools = nxt.get("tools", {})

    added_nodes = sorted(set(next_nodes) - set(prev_nodes))
    removed_nodes = sorted(set(prev_nodes) - set(next_nodes))

    relabeled = []
    for nid in set(prev_nodes) & set(next_nodes):
        if prev_nodes[nid].get("description") != next_nodes[nid].get("description"):
            relabeled.append({
                "id": nid,
                "before": prev_nodes[nid].get("description"),
                "after": next_nodes[nid].get("description"),
            })

    added_tools = sorted(set(next_tools) - set(prev_tools))
    removed_tools = sorted(set(prev_tools) - set(next_tools))

    return SnapshotDiff(
        prev_version=prev.get("version", ""),
        next_version=nxt.get("version", ""),
        added_nodes=added_nodes,
        removed_nodes=removed_nodes,
        relabeled_nodes=relabeled,
        added_tools=added_tools,
        removed_tools=removed_tools,
    )
