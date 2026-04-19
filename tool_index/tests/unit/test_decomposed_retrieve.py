"""Unit tests for the per-sub-query union retriever."""
from __future__ import annotations

from tool_index.retrieval.decomposed_retrieve import retrieve_decomposed
from tool_index.schema import Node, Tree


def _make_tree() -> Tree:
    """Two-leaf-group tree: group_a holds [tool_x, tool_y], group_b holds [tool_z]."""
    root = Node(id="root", level="L0", description="root", embedding=[0.0, 0.0], children=["group_a", "group_b"])
    group_a = Node(id="group_a", level="L3", description="alpha tools", embedding=[1.0, 0.0], children=["tool_x", "tool_y"])
    group_b = Node(id="group_b", level="L3", description="beta tools", embedding=[0.0, 1.0], children=["tool_z"])
    return Tree(
        root=root,
        nodes_by_id={"root": root, "group_a": group_a, "group_b": group_b},
    )


def _vec_pair() -> dict[str, list[list[float]]]:
    return {
        "tool_x": [[1.0, 0.0]],
        "tool_y": [[0.9, 0.1]],
        "tool_z": [[0.0, 1.0]],
    }


def test_decomposed_retrieve_unions_two_sub_queries():
    tree = _make_tree()
    vectors = _vec_pair()
    out = retrieve_decomposed(
        tree,
        sub_query_embeddings=[[1.0, 0.0], [0.0, 1.0]],
        tool_vectors=vectors,
        k=3,
        rerank_k=10,
        beam=2,
    )
    assert set(out) == {"tool_x", "tool_y", "tool_z"}


def test_decomposed_retrieve_single_sub_query_matches_baseline_shape():
    tree = _make_tree()
    out = retrieve_decomposed(
        tree,
        sub_query_embeddings=[[1.0, 0.0]],
        tool_vectors=_vec_pair(),
        k=2,
        rerank_k=10,
        beam=2,
    )
    assert out[0] == "tool_x"


def test_decomposed_retrieve_handles_empty_sub_queries():
    out = retrieve_decomposed(
        _make_tree(),
        sub_query_embeddings=[],
        tool_vectors=_vec_pair(),
        k=5,
        rerank_k=10,
        beam=2,
    )
    assert out == []


def test_decomposed_retrieve_respects_k():
    tree = _make_tree()
    out = retrieve_decomposed(
        tree,
        sub_query_embeddings=[[1.0, 0.0], [0.0, 1.0]],
        tool_vectors=_vec_pair(),
        k=2,
        rerank_k=10,
        beam=2,
    )
    assert len(out) == 2
