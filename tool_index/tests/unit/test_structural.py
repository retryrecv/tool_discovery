from tool_index.schema import Node, ToolDescriptor, Tree, ValidationReport
from tool_index.validation.structural import check_structural


FANOUT = {
    "domain": (1, 10),
    "category": (1, 10),
    "group": (1, 10),
    "tool": (1, 10),
}


def _register(tree: Tree, *nodes: Node) -> Tree:
    for node in nodes:
        tree.register(node)
    return tree


def _full_depth_tree() -> Tree:
    root = Node(id="r", level="L0", description="all", children=["d1", "d2"])
    d1 = Node(id="d1", level="L1", description="domain 1", children=["c1", "c2"], parent_id="r")
    d2 = Node(id="d2", level="L1", description="domain 2", children=["c3", "c4"], parent_id="r")
    c1 = Node(id="c1", level="L2", description="category 1", children=["g1", "g2"], parent_id="d1")
    c2 = Node(id="c2", level="L2", description="category 2", children=["g3", "g4"], parent_id="d1")
    c3 = Node(id="c3", level="L2", description="category 3", children=["g5", "g6"], parent_id="d2")
    c4 = Node(id="c4", level="L2", description="category 4", children=["g7", "g8"], parent_id="d2")
    groups = [
        Node(id="g1", level="L3", description="group 1", children=["t1"], parent_id="c1"),
        Node(id="g2", level="L3", description="group 2", children=["t2"], parent_id="c1"),
        Node(id="g3", level="L3", description="group 3", children=["t3"], parent_id="c2"),
        Node(id="g4", level="L3", description="group 4", children=["t4"], parent_id="c2"),
        Node(id="g5", level="L3", description="group 5", children=["t5"], parent_id="c3"),
        Node(id="g6", level="L3", description="group 6", children=["t6"], parent_id="c3"),
        Node(id="g7", level="L3", description="group 7", children=["t7"], parent_id="c4"),
        Node(id="g8", level="L3", description="group 8", children=["t8"], parent_id="c4"),
    ]
    tree = Tree(root=root)
    _register(tree, root, d1, d2, c1, c2, c3, c4, *groups)
    tree.tools_by_id = {
        f"t{i}": ToolDescriptor(id=f"t{i}", name=f"tool_{i}", signature="", original_doc=f"doc {i}")
        for i in range(1, 9)
    }
    return tree


def _collapsed_depth_tree() -> Tree:
    root = Node(id="r", level="L0", description="all", children=["d1", "d2"])
    d1 = Node(id="d1", level="L1", description="domain 1", children=["g1", "g2"], parent_id="r")
    d2 = Node(id="d2", level="L1", description="domain 2", children=["g3", "g4"], parent_id="r")
    g1 = Node(id="g1", level="L3", description="group 1", children=["t1"], parent_id="d1")
    g2 = Node(id="g2", level="L3", description="group 2", children=["t2"], parent_id="d1")
    g3 = Node(id="g3", level="L3", description="group 3", children=["t3"], parent_id="d2")
    g4 = Node(id="g4", level="L3", description="group 4", children=["t4"], parent_id="d2")
    tree = Tree(root=root)
    _register(tree, root, d1, d2, g1, g2, g3, g4)
    tree.tools_by_id = {
        f"t{i}": ToolDescriptor(id=f"t{i}", name=f"tool_{i}", signature="", original_doc=f"doc {i}")
        for i in range(1, 5)
    }
    return tree


def test_structural_accepts_full_depth_tree():
    report = ValidationReport()
    check_structural(_full_depth_tree(), FANOUT, expected_depth=5, report=report)
    assert report.passed is True
    assert report.details["depth"] == 5
    assert report.details["expected_depth"] == 5


def test_structural_accepts_collapsed_depth_tree():
    report = ValidationReport()
    check_structural(_collapsed_depth_tree(), FANOUT, expected_depth=4, report=report)
    assert report.passed is True
    assert report.details["depth"] == 4
    assert report.details["expected_depth"] == 4


def test_structural_rejects_full_depth_tree_when_expecting_collapsed_depth():
    report = ValidationReport()
    check_structural(_full_depth_tree(), FANOUT, expected_depth=4, report=report)
    assert report.passed is False
    assert "tree depth 5 != expected 4" in report.errors
    assert report.details["depth"] == 5
    assert report.details["expected_depth"] == 4


def test_structural_rejects_collapsed_depth_tree_when_expecting_full_depth():
    report = ValidationReport()
    check_structural(_collapsed_depth_tree(), FANOUT, expected_depth=5, report=report)
    assert report.passed is False
    assert "tree depth 4 != expected 5" in report.errors
    assert report.details["depth"] == 4
    assert report.details["expected_depth"] == 5
