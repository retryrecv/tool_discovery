"""Incremental rebuild planning.

Given a previous frozen snapshot and a fresh `raw_tools` list, compute
which tools changed and which ancestors must be re-clustered as a
result. The expensive stages can then run only over the affected
subset.

Heuristic for "must rebuild":
  - tools added or removed → re-enrich added, drop removed; re-cluster
    every parent group whose membership changed.
  - tools whose `(name, signature, doc, examples)` hash changed →
    re-enrich; re-cluster their group only (description likely shifted).
  - all parents up to root see their description re-labeled if any
    descendant changed (cheap label-only LLM call).

This module just produces the plan. Actually executing it is the
caller's job — a future `pipeline.incremental_orchestrator` would
consume the plan and dispatch stage-2/3/4 work for the changed subset
only.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Iterable

from ..schema import RawTool, Tree


def tool_content_hash(t: RawTool) -> str:
    canonical = {
        "name": t["name"],
        "signature": t.get("signature", ""),
        "doc": t.get("doc", ""),
        "examples": sorted(
            json.dumps(e, sort_keys=True) for e in t.get("examples", []) or []
        ),
    }
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True).encode("utf-8")
    ).hexdigest()


@dataclass
class DescriptorDiff:
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.added or self.removed or self.changed)


def diff_descriptors(
    prev_hashes: dict[str, str],
    new_tools: Iterable[RawTool],
) -> DescriptorDiff:
    """Compare new tools against `{tool_name: hash}` from the prior build."""
    diff = DescriptorDiff()
    new_map: dict[str, str] = {}
    for t in new_tools:
        h = tool_content_hash(t)
        new_map[t["name"]] = h
        if t["name"] not in prev_hashes:
            diff.added.append(t["name"])
        elif prev_hashes[t["name"]] != h:
            diff.changed.append(t["name"])
        else:
            diff.unchanged.append(t["name"])
    for name in prev_hashes:
        if name not in new_map:
            diff.removed.append(name)
    return diff


@dataclass
class IncrementalPlan:
    diff: DescriptorDiff
    tools_to_reenrich: list[str] = field(default_factory=list)
    groups_to_recluster: list[str] = field(default_factory=list)
    parents_to_relabel: list[str] = field(default_factory=list)
    requires_full_rebuild: bool = False
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "added": len(self.diff.added),
            "removed": len(self.diff.removed),
            "changed": len(self.diff.changed),
            "unchanged": len(self.diff.unchanged),
            "tools_to_reenrich": self.tools_to_reenrich,
            "groups_to_recluster": self.groups_to_recluster,
            "parents_to_relabel": self.parents_to_relabel,
            "requires_full_rebuild": self.requires_full_rebuild,
            "reason": self.reason,
        }


def plan_incremental_rebuild(
    prev_tree: Tree | None,
    prev_hashes: dict[str, str],
    new_tools: list[RawTool],
    *,
    full_rebuild_threshold: float = 0.30,
) -> IncrementalPlan:
    """Decide which subtree to rebuild.

    If the changed-fraction exceeds `full_rebuild_threshold` the plan
    flags a full rebuild — at that point incremental work isn't worth
    the bookkeeping vs. just re-running everything.
    """
    diff = diff_descriptors(prev_hashes, new_tools)

    if prev_tree is None:
        return IncrementalPlan(
            diff=diff,
            requires_full_rebuild=True,
            reason="no previous snapshot",
        )

    total = len(new_tools) + len(diff.removed)
    changed_total = len(diff.added) + len(diff.removed) + len(diff.changed)
    if total == 0 or changed_total / total >= full_rebuild_threshold:
        return IncrementalPlan(
            diff=diff,
            requires_full_rebuild=True,
            reason=f"changed fraction {changed_total}/{total} ≥ {full_rebuild_threshold}",
        )

    name_to_tool_id: dict[str, str] = {}
    for tid, td in prev_tree.tools_by_id.items():
        name_to_tool_id[td.name] = tid

    affected_tool_ids: set[str] = set()
    for name in diff.changed + diff.removed:
        if name in name_to_tool_id:
            affected_tool_ids.add(name_to_tool_id[name])

    affected_groups: set[str] = set()
    for node in prev_tree.nodes_by_id.values():
        if any(c in affected_tool_ids for c in node.children):
            affected_groups.add(node.id)

    parents_to_relabel: set[str] = set(affected_groups)
    cur = set(affected_groups)
    while cur:
        next_set: set[str] = set()
        for nid in cur:
            node = prev_tree.nodes_by_id.get(nid)
            if node and node.parent_id and node.parent_id in prev_tree.nodes_by_id:
                next_set.add(node.parent_id)
        parents_to_relabel |= next_set
        cur = next_set

    return IncrementalPlan(
        diff=diff,
        tools_to_reenrich=sorted(set(diff.added) | set(diff.changed)),
        groups_to_recluster=sorted(affected_groups),
        parents_to_relabel=sorted(parents_to_relabel),
        requires_full_rebuild=False,
        reason=f"{changed_total} of {total} tools changed",
    )
