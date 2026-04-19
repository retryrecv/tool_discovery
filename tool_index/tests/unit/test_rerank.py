from __future__ import annotations

from tool_index.retrieval.rerank import rerank_tools


def test_rerank_promotes_tool_with_close_query_vector_even_when_intent_is_far() -> None:
    """A tool whose intent is far from query but has a matching example_query
    should be promoted by MaxSim."""
    q = [1.0, 0.0, 0.0]
    candidates = ["t_intent_only", "t_query_match", "t_far"]
    tool_vectors = {
        # intent close-ish, no helpful queries.
        "t_intent_only": [[0.5, 0.5, 0.0]],
        # intent far, but one example query lands on the query vector exactly.
        "t_query_match": [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        # everything far from query.
        "t_far": [[0.0, 1.0, 0.0]],
    }
    out = rerank_tools(q, candidates, tool_vectors, k=2)
    assert out[0] == "t_query_match"


def test_rerank_keeps_unknown_tools_at_the_back() -> None:
    """Tools missing from tool_vectors get score 0.0 and end up last,
    but are not dropped."""
    q = [1.0, 0.0, 0.0]
    candidates = ["t_known", "t_unknown"]
    tool_vectors = {"t_known": [[1.0, 0.0, 0.0]]}
    out = rerank_tools(q, candidates, tool_vectors, k=2)
    assert out == ["t_known", "t_unknown"]


def test_rerank_truncates_to_k() -> None:
    q = [1.0, 0.0, 0.0]
    candidates = ["a", "b", "c", "d"]
    tool_vectors = {
        "a": [[1.0, 0.0, 0.0]],
        "b": [[0.9, 0.1, 0.0]],
        "c": [[0.8, 0.2, 0.0]],
        "d": [[0.7, 0.3, 0.0]],
    }
    out = rerank_tools(q, candidates, tool_vectors, k=2)
    assert out == ["a", "b"]
