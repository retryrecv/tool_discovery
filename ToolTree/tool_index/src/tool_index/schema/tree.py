"""`Tree`, `BuildTrace`, `ValidationReport` — the top-level snapshot triple.

A frozen snapshot is exactly these three structures plus the tool embeddings
that stage 3 produced. `Tree` is the graph, `BuildTrace` is the recipe that
produced it, and `ValidationReport` is the evidence it's usable.

Keeping them together (but separately serializable) means a consumer can
load just the `Tree` for routing and ignore the trace / report unless
debugging.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Iterator

from .node import Node
from .descriptor import ToolDescriptor
from .constants import LEVEL_ROOT


@dataclass
class BuildTrace:
    """Recipe that produced the tree — enough to rebuild or audit it.

    Written once by stage 6. Not read on the hot path; present so two
    snapshots can be diffed to explain *why* their trees differ.
    """

    # Stable IDs of the providers used at each stage. Different IDs between
    # snapshots mean the models changed — a likely cause of tree drift.
    enricher_llm: str = ""
    labeler_llm: str = ""
    embedding_model: str = ""

    # Exact threshold dict from the config (near-dup, per-level cluster
    # distance, discriminability, min_recall). Float-for-float.
    thresholds: dict = field(default_factory=dict)

    # Per-level ``(min, max)`` fanout bounds from the config.
    fanout: dict = field(default_factory=dict)

    # Synthetic queries generated in stage 5. Duplicated into
    # `ValidationReport.seed_eval_set` for convenience; kept here so the
    # eval set travels with the trace even if the report is dropped.
    seed_eval_set: list = field(default_factory=list)

    # Summary block from `ValidationReport.summary()`. Stored here so a
    # snapshot is self-describing without a separate report file.
    validation: dict = field(default_factory=dict)

    # ISO-8601 timestamp of when stage 6 ran. Purely informational.
    timestamp: str = ""

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict."""
        return {
            "enricher_llm": self.enricher_llm,
            "labeler_llm": self.labeler_llm,
            "embedding_model": self.embedding_model,
            "thresholds": self.thresholds,
            "fanout": self.fanout,
            "seed_eval_set": self.seed_eval_set,
            "validation": self.validation,
            "timestamp": self.timestamp,
        }


@dataclass
class ValidationReport:
    """Outcome of stage 5 — structural, discriminability, and recall checks.

    ``passed`` is the single gate the orchestrator uses (when ``strict=True``)
    to decide whether to freeze. Everything else is diagnostic.
    """

    # False if any structural check or recall threshold failed. Flipped by
    # `fail()`; never set back to True after a failure.
    passed: bool = True

    # Hard-fail messages. Populated by `fail()`. Raised as
    # `pipeline.ValidationError` when strict mode is on.
    errors: list[str] = field(default_factory=list)

    # Soft issues worth noting but not fatal (e.g. a single cluster slightly
    # below the discriminability threshold). Never block freezing.
    warnings: list[str] = field(default_factory=list)

    # Measured recall@k on the synthetic eval set. Compared against
    # ``thresholds["min_recall"]``; surfaced in build logs.
    recall_at_k: float = 0.0

    # The synthetic queries used to measure recall, each tagged with its
    # expected tool ID. Persisted so the benchmark is reproducible.
    seed_eval_set: list[dict] = field(default_factory=list)

    # Per-validator breakdown — structure differs by validator. Read for
    # debugging; no production code should branch on its contents.
    details: dict[str, Any] = field(default_factory=dict)

    def fail(self, msg: str) -> None:
        """Mark the report as failed and record a human-readable reason."""
        self.passed = False
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        """Append a non-fatal observation."""
        self.warnings.append(msg)

    def summary(self) -> dict:
        """Compact view suitable for embedding in `BuildTrace.validation`."""
        return {
            "passed": self.passed,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "recall_at_k": self.recall_at_k,
            "details": self.details,
        }


@dataclass
class Tree:
    """The full hierarchical index — nodes, tool leaves, and provenance.

    Construction goes: orchestrator builds nodes bottom-up, calls
    `register()` for each, then attaches `tools_by_id` and `build_trace`
    before handing off to stage 6 for freezing.
    """

    # The L0 root. Always present, even for trivially small trees.
    root: Node

    # Every inner `Node` in the graph, keyed by `Node.id`. Populated via
    # `register()`. The root is also in this map.
    nodes_by_id: dict[str, Node] = field(default_factory=dict)

    # Every tool leaf, keyed by `ToolDescriptor.id`. An L3 node's
    # ``children`` list contains IDs into this map; the traverser
    # distinguishes leaves from inner nodes by lookup failure in
    # `nodes_by_id`.
    tools_by_id: dict[str, ToolDescriptor] = field(default_factory=dict)

    # Snapshot version string. ``"v0-draft"`` during construction; stage 6
    # replaces it with the frozen version (e.g. ``"v0"``).
    version: str = "v0-draft"

    # Provenance / recipe — filled in by stage 6.
    build_trace: BuildTrace = field(default_factory=BuildTrace)

    def register(self, node: Node) -> None:
        """Add (or replace) a node in the by-id map. Parent wiring is the
        caller's responsibility — this just makes the node addressable."""
        self.nodes_by_id[node.id] = node

    def all_nodes(self) -> Iterator[Node]:
        """Iterate every registered node (root + inner)."""
        return iter(self.nodes_by_id.values())

    def non_leaf_nodes(self) -> Iterator[Node]:
        """Yield nodes that have at least one child which is itself a node.

        Useful for validators that only care about inner structure (e.g.
        sibling discriminability at non-leaf levels).
        """
        for n in self.nodes_by_id.values():
            if n.children and any(c in self.nodes_by_id for c in n.children):
                yield n

    def depth(self) -> int:
        """Total levels including the root and the tool-leaf layer.

        A pure root with tool children has depth 2. A root → group → tool
        tree has depth 3. The design's target 4-level tree (root → domain
        → category → group → tool) has depth 5.
        """
        def _d(node: Node) -> int:
            inner = [self.nodes_by_id[c] for c in node.children if c in self.nodes_by_id]
            leaf_children = [c for c in node.children if c not in self.nodes_by_id]
            if inner:
                return 1 + max(_d(c) for c in inner)
            if leaf_children:
                return 2  # this node + the leaf layer beneath it
            return 1
        return _d(self.root)

    def to_dict(self) -> dict:
        """Serialize the whole tree for stage 6's ``tree.json``."""
        return {
            "version": self.version,
            "root_id": self.root.id,
            "nodes": {nid: n.to_dict() for nid, n in self.nodes_by_id.items()},
            "tools": {tid: t.to_dict() for tid, t in self.tools_by_id.items()},
            "build_trace": self.build_trace.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Tree":
        """Rehydrate a tree written by `to_dict` / stage 6.

        Unknown keys in ``build_trace`` are dropped so older snapshots stay
        readable after we add new trace fields.
        """
        nodes = {nid: Node.from_dict(nd) for nid, nd in d["nodes"].items()}
        tools = {tid: ToolDescriptor.from_dict(td) for tid, td in d["tools"].items()}
        root = nodes[d["root_id"]]
        bt = BuildTrace(**{k: v for k, v in d.get("build_trace", {}).items() if k in BuildTrace.__dataclass_fields__})
        return cls(root=root, nodes_by_id=nodes, tools_by_id=tools, version=d["version"], build_trace=bt)
