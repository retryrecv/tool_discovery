from tool_index.schema import ToolDescriptor, Enrichment, Node, Tree


def test_descriptor_roundtrip():
    d = ToolDescriptor(id="t1", name="foo", signature="foo()", original_doc="hi")
    assert ToolDescriptor.from_dict(d.to_dict()) == d


def test_enrichment_compose_text():
    e = Enrichment("do x", "arg", "result", ["x", "y"], ["q1", "q2"])
    t = e.compose_leaf_text()
    assert "do x" in t and "q1" in t


def test_tree_depth_and_roundtrip():
    root = Node(id="r", level="L0", description="all", children=["d"])
    d = Node(id="d", level="L1", description="dom", children=["c"], parent_id="r")
    c = Node(id="c", level="L2", description="cat", children=["t1"], parent_id="d")
    tree = Tree(root=root)
    for n in (root, d, c):
        tree.register(n)
    tree.tools_by_id = {"t1": ToolDescriptor(id="t1", name="tool", signature="", original_doc="")}
    assert tree.depth() == 4
    restored = Tree.from_dict(tree.to_dict())
    assert restored.depth() == 4
    assert "t1" in restored.tools_by_id
