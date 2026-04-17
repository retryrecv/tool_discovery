"""`Node` — a single vertex in the tree (any of root / domain / category / group).

Nodes describe the *inner* structure of the index; tool leaves are stored
separately in `Tree.tools_by_id` and referenced by ID from an L3 node's
``children`` list. This split keeps tool metadata (signatures, docs) out of
the clustering / labeling hot path, and means a `Node` can be embedded and
compared without pulling in any tool's payload.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Node:
    """An inner node of the hierarchical index.

    A node's identity is its ``id``; equality-by-value is never needed.
    The ``level`` constrains what ``children`` may contain — see comment on
    that field below.
    """

    # Content-derived stable ID (see `utils/ids.py`). Persisted in snapshots
    # and referenced from parent nodes' ``children`` lists, so regenerating
    # from the same inputs yields the same graph.
    id: str

    # One of `schema.constants.LEVEL_*` (``"L0"`` … ``"L3"``). Tool leaves
    # are not `Node` instances — they live in `Tree.tools_by_id`. So a node
    # with ``level == "L4"`` should not exist in practice.
    level: str

    # Human- (and LLM-) readable summary of what this subtree covers. For
    # non-root nodes this is produced by `labeling/` and must be
    # discriminable from siblings (validated in stage 5).
    description: str

    # Vector embedding of ``description`` in the configured embedding space.
    # Used by the retrieval traverser to pick which child to descend into.
    # Length must equal ``EmbeddingProvider.dim`` for the model in use.
    embedding: list[float] = field(default_factory=list)

    # IDs of direct children. At levels L0-L2 these are other `Node` IDs;
    # at L3 they are `ToolDescriptor` IDs (leaves). The traverser
    # distinguishes the two by looking each ID up in `Tree.nodes_by_id` vs
    # `Tree.tools_by_id`.
    children: list[str] = field(default_factory=list)

    # ID of the parent `Node`. ``None`` for the root. Wired up by the
    # orchestrator after clustering, not by the clusterers themselves.
    parent_id: str | None = None

    # Free-form bookkeeping: cluster size, distance threshold used,
    # rebalance history, etc. Persisted with the snapshot for debugging
    # and post-hoc analysis. Never read by production code paths.
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict."""
        return {
            "id": self.id,
            "level": self.level,
            "description": self.description,
            "embedding": list(self.embedding),
            "children": list(self.children),
            "parent_id": self.parent_id,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Node":
        """Rehydrate from a snapshot dict, filling in sensible defaults."""
        return cls(
            id=d["id"],
            level=d["level"],
            description=d["description"],
            embedding=list(d.get("embedding", [])),
            children=list(d.get("children", [])),
            parent_id=d.get("parent_id"),
            provenance=d.get("provenance", {}),
        )
