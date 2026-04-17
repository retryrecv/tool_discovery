"""Structural validator — tree shape invariants.

Catches the kinds of bugs that make a tree *unusable* regardless of
description quality: orphan nodes, fanout bound violations, unexpected
depth. Runs first in stage 5 because the other validators assume a
structurally sound tree.

Failure modes:
    • ``fail()`` — orphan, single-child chain (above L3), fanout above max,
      or unexpected depth. Will trip ``strict=True`` builds.
    • ``warn()`` — fanout below min (surviving only because merge had no
      viable candidate).
"""
from __future__ import annotations
from ..schema import Tree, Node, ValidationReport


def check_structural(tree: Tree, fanout: dict, expected_depth: int, report: ValidationReport) -> None:
    """Validate tree shape and populate the report with findings.

    Args:
        tree: The assembled tree to check.
        fanout: Per-level ``{name: (min, max)}`` bounds from config.
        expected_depth: Depth the tree should have. The orchestrator passes
            ``tree.depth()`` so collapsed 3-level trees don't fail here.
        report: Mutated with ``fail()``, ``warn()``, and ``details``.
    """
    seen_ids = set(tree.nodes_by_id)

    # Orphan check — every non-root node must have a parent registered in
    # the tree. Finding one here means the orchestrator's wiring missed a
    # node, which is always a bug.
    for n in tree.all_nodes():
        if n.id == tree.root.id:
            continue
        if n.parent_id is None or n.parent_id not in seen_ids:
            report.fail(f"orphan node {n.id}")

    # Map a node's level to the fanout key describing *its children*. Root
    # (L0) has domain-level children; L1 has categories; L2 has groups;
    # L3 has tool leaves.
    for n in tree.all_nodes():
        if not n.children:
            continue
        if n.level == "L0":
            key = "domain"
        elif n.level == "L1":
            key = "category"
        elif n.level == "L2":
            key = "group"
        elif n.level == "L3":
            key = "tool"
        else:
            continue
        lo, hi = fanout.get(key, (1, 10**6))
        count = len(n.children)
        # Single-child non-leaf chains collapse routing — the retrieval
        # step becomes useless if every descent has only one option.
        # Exempt L3 because a single-tool group is legitimate (e.g. a
        # very niche capability).
        if count == 1 and n.level != "L3":
            report.fail(f"single-child chain at {n.id} (level {n.level})")
        # Below-min is a soft signal — rebalance already tried to fix it.
        if count < lo and count >= 1:
            report.warn(f"fanout below min at {n.id}: {count} < {lo}")
        # Above-max is hard-fail: rebalance should have split this.
        if count > hi:
            report.fail(f"fanout above max at {n.id}: {count} > {hi}")

    depth = tree.depth()
    report.details["depth"] = depth
    if depth != expected_depth:
        report.fail(f"tree depth {depth} != expected {expected_depth}")
