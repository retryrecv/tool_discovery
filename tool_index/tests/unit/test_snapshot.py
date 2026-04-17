from pathlib import Path

from tool_index.schema import Tree, Node, ToolDescriptor, BuildTrace
from tool_index.storage import write_immutable_snapshot, load_snapshot


def test_snapshot_roundtrip(tmp_path: Path):
    root = Node(id="r", level="L0", description="all", children=["d"])
    d = Node(id="d", level="L1", description="dom", children=[], parent_id="r")
    tree = Tree(root=root, version="v0")
    tree.register(root)
    tree.register(d)
    tree.tools_by_id = {"t1": ToolDescriptor(id="t1", name="x", signature="", original_doc="")}
    tree.build_trace = BuildTrace(embedding_model="fake")

    out = write_immutable_snapshot(tree, tmp_path / "v0")
    assert (out / "tree.json").exists()
    assert (out / "build_trace.json").exists()

    loaded = load_snapshot(out)
    assert loaded.version == "v0"
    assert "t1" in loaded.tools_by_id
    assert loaded.depth() == 2
