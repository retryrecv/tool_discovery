from __future__ import annotations

from tool_index.scale import (
    DescriptorDiff,
    diff_descriptors,
    plan_incremental_rebuild,
    tool_content_hash,
)
from tool_index.schema import Node, Tree
from tool_index.schema.descriptor import ToolDescriptor


def test_hash_stable_across_example_order() -> None:
    a = {"name": "t", "signature": "s", "doc": "d", "examples": [{"args": {"x": 1}}, {"args": {"y": 2}}]}
    b = {"name": "t", "signature": "s", "doc": "d", "examples": [{"args": {"y": 2}}, {"args": {"x": 1}}]}
    assert tool_content_hash(a) == tool_content_hash(b)


def test_diff_added_removed_changed() -> None:
    prev = {"a": "h_a", "b": "h_b", "c": "h_c"}
    new = [
        {"name": "a", "signature": "", "doc": ""},
        {"name": "b", "signature": "", "doc": "changed"},
        {"name": "d", "signature": "", "doc": ""},
    ]
    d = diff_descriptors(prev, new)
    assert "d" in d.added
    assert "c" in d.removed
    assert "b" in d.changed


def test_no_prev_tree_forces_full_rebuild() -> None:
    plan = plan_incremental_rebuild(None, {}, [{"name": "x", "signature": "", "doc": ""}])
    assert plan.requires_full_rebuild
    assert "no previous snapshot" in plan.reason


def test_high_change_fraction_forces_full_rebuild() -> None:
    tools = [{"name": f"t{i}", "signature": "", "doc": ""} for i in range(10)]
    prev_hashes = {f"t{i}": "old" for i in range(10)}
    tree = Tree(root=Node(id="root", level="root", description="", embedding=[0.0]))
    plan = plan_incremental_rebuild(tree, prev_hashes, tools, full_rebuild_threshold=0.30)
    assert plan.requires_full_rebuild


def test_small_change_keeps_incremental() -> None:
    tools = [{"name": f"t{i}", "signature": "", "doc": ""} for i in range(10)]
    prev_hashes = {f"t{i}": tool_content_hash(tools[i]) for i in range(10)}
    prev_hashes["t1"] = "old"

    root = Node(id="root", level="root", description="", embedding=[0.0], children=["g0", "g1"])
    g0 = Node(id="g0", level="group", description="g0", embedding=[0.0], children=["tool_t0", "tool_t1"], parent_id="root")
    g1 = Node(id="g1", level="group", description="g1", embedding=[0.0], children=["tool_t2"], parent_id="root")
    tree = Tree(root=root)
    for n in (root, g0, g1):
        tree.register(n)
    tree.tools_by_id = {
        "tool_t0": ToolDescriptor(id="tool_t0", name="t0", signature="", original_doc=""),
        "tool_t1": ToolDescriptor(id="tool_t1", name="t1", signature="", original_doc=""),
        "tool_t2": ToolDescriptor(id="tool_t2", name="t2", signature="", original_doc=""),
    }

    plan = plan_incremental_rebuild(tree, prev_hashes, tools, full_rebuild_threshold=0.50)
    assert not plan.requires_full_rebuild
    assert "t1" in plan.tools_to_reenrich
    assert "g0" in plan.groups_to_recluster
    assert "root" in plan.parents_to_relabel
