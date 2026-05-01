"""`Tree`, `BuildTrace` — the top-level snapshot pair.

A frozen snapshot is the `Tree` graph plus its `BuildTrace` recipe,
serialized together with the tool embeddings produced during clustering.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterator

from .node import Node
from .descriptor import ToolDescriptor
from .constants import LEVEL_ROOT


@dataclass
class BuildTrace:
    """Recipe that produced the tree — enough to rebuild or audit it.

    Written once by the freeze stage. Not read on the hot path; present so
    two snapshots can be diffed to explain *why* their trees differ.
    """

    enricher_llm: str = ""
    labeler_llm: str = ""
    embedding_model: str = ""

    thresholds: dict = field(default_factory=dict)

    fanout: dict = field(default_factory=dict)

    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "enricher_llm": self.enricher_llm,
            "labeler_llm": self.labeler_llm,
            "embedding_model": self.embedding_model,
            "thresholds": self.thresholds,
            "fanout": self.fanout,
            "timestamp": self.timestamp,
        }


@dataclass
class Tree:
    """The full hierarchical index — nodes, tool leaves, and provenance.

    Construction goes: orchestrator builds nodes bottom-up, calls
    `register()` for each, then attaches `tools_by_id` and `build_trace`
    before handing off to the freeze stage.
    """

    root: Node

    nodes_by_id: dict[str, Node] = field(default_factory=dict)

    tools_by_id: dict[str, ToolDescriptor] = field(default_factory=dict)

    version: str = "v0-draft"

    build_trace: BuildTrace = field(default_factory=BuildTrace)

    def register(self, node: Node) -> None:
        self.nodes_by_id[node.id] = node

    def all_nodes(self) -> Iterator[Node]:
        return iter(self.nodes_by_id.values())

    def non_leaf_nodes(self) -> Iterator[Node]:
        for n in self.nodes_by_id.values():
            if n.children and any(c in self.nodes_by_id for c in n.children):
                yield n

    def depth(self) -> int:
        """Total levels including the root and the tool-leaf layer."""
        def _d(node: Node) -> int:
            inner = [self.nodes_by_id[c] for c in node.children if c in self.nodes_by_id]
            leaf_children = [c for c in node.children if c not in self.nodes_by_id]
            if inner:
                return 1 + max(_d(c) for c in inner)
            if leaf_children:
                return 2
            return 1
        return _d(self.root)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "root_id": self.root.id,
            "nodes": {nid: n.to_dict() for nid, n in self.nodes_by_id.items()},
            "tools": {tid: t.to_dict() for tid, t in self.tools_by_id.items()},
            "build_trace": self.build_trace.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Tree":
        nodes = {nid: Node.from_dict(nd) for nid, nd in d["nodes"].items()}
        tools = {tid: ToolDescriptor.from_dict(td) for tid, td in d["tools"].items()}
        root = nodes[d["root_id"]]
        bt = BuildTrace(**{k: v for k, v in d.get("build_trace", {}).items() if k in BuildTrace.__dataclass_fields__})
        return cls(root=root, nodes_by_id=nodes, tools_by_id=tools, version=d["version"], build_trace=bt)
