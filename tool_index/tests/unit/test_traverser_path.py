from __future__ import annotations

from tool_index.retrieval.traverser import retrieve, retrieve_with_path
from tool_index.schema import Node, Tree


def _three_level_tree() -> Tree:
    root = Node(id="root", level="root", description="all",
                embedding=[0.0, 0.0, 1.0],
                children=["d_files", "d_math"])
    d_files = Node(id="d_files", level="domain", description="files",
                   embedding=[1.0, 0.0, 0.0], children=["c_readwrite"],
                   parent_id="root")
    c_readwrite = Node(id="c_readwrite", level="category", description="read/write",
                       embedding=[1.0, 0.1, 0.0], children=["g_read"],
                       parent_id="d_files")
    g_read = Node(id="g_read", level="group", description="read",
                  embedding=[1.0, 0.2, 0.0], children=["tool_ls", "tool_cat"],
                  parent_id="c_readwrite")
    d_math = Node(id="d_math", level="domain", description="math",
                  embedding=[0.0, 1.0, 0.0], children=["c_arith"],
                  parent_id="root")
    c_arith = Node(id="c_arith", level="category", description="arith",
                   embedding=[0.0, 1.0, 0.1], children=["g_add"],
                   parent_id="d_math")
    g_add = Node(id="g_add", level="group", description="add",
                 embedding=[0.0, 1.0, 0.2], children=["tool_add"],
                 parent_id="c_arith")
    t = Tree(root=root)
    for n in (root, d_files, c_readwrite, g_read, d_math, c_arith, g_add):
        t.register(n)
    return t


def test_retrieve_with_path_returns_same_tools_as_retrieve() -> None:
    t = _three_level_tree()
    q = [1.0, 0.0, 0.0]
    tools_a = retrieve(t, q, k=5, beam=1)
    tools_b, path = retrieve_with_path(t, q, k=5, beam=1)
    assert tools_a == tools_b


def test_path_descends_through_three_levels() -> None:
    t = _three_level_tree()
    q = [1.0, 0.0, 0.0]
    _, path = retrieve_with_path(t, q, k=5, beam=1)
    # Path should descend domain -> category -> group.
    assert path == ["d_files", "c_readwrite", "g_read"]


def test_path_routes_to_math_side_for_math_query() -> None:
    t = _three_level_tree()
    q = [0.0, 1.0, 0.0]
    _, path = retrieve_with_path(t, q, k=5, beam=1)
    assert path[0] == "d_math"
    assert path[-1] == "g_add"


def test_multivector_score_lets_child_rescue_misleading_parent() -> None:
    """Inner node with a misleading description but a matching child wins frontier."""
    root = Node(id="root", level="root", description="all",
                embedding=[0.0, 0.0, 1.0], children=["d_good", "d_bad"])
    # d_good's own description is far from query; child g_good is close.
    d_good = Node(id="d_good", level="domain", description="misleading",
                  embedding=[0.0, 0.0, 1.0], children=["g_good"], parent_id="root")
    g_good = Node(id="g_good", level="group", description="good",
                  embedding=[1.0, 0.0, 0.0], children=["tool_target"], parent_id="d_good")
    # d_bad's own description is closer to query than d_good's, but no child helps.
    d_bad = Node(id="d_bad", level="domain", description="distractor",
                 embedding=[0.5, 0.0, 0.5], children=["g_bad"], parent_id="root")
    g_bad = Node(id="g_bad", level="group", description="bad",
                 embedding=[0.4, 0.0, 0.4], children=["tool_other"], parent_id="d_bad")
    t = Tree(root=root)
    for n in (root, d_good, g_good, d_bad, g_bad):
        t.register(n)

    q = [1.0, 0.0, 0.0]
    tools = retrieve(t, q, k=5, beam=1)
    assert tools[0] == "tool_target"


def test_multivector_score_falls_back_to_node_embedding_when_no_inner_children() -> None:
    """Leaf-parent groups (whose children are tool IDs, not nodes) keep working."""
    root = Node(id="root", level="root", description="all",
                embedding=[0.0, 0.0, 1.0], children=["g_only"])
    g_only = Node(id="g_only", level="group", description="only",
                  embedding=[1.0, 0.0, 0.0], children=["tool_a", "tool_b"], parent_id="root")
    t = Tree(root=root)
    t.register(root)
    t.register(g_only)

    q = [1.0, 0.0, 0.0]
    tools = retrieve(t, q, k=5, beam=1)
    assert set(tools) == {"tool_a", "tool_b"}
