"""Unit tests for retrieval-aware recursive decomposition."""
from __future__ import annotations

from tool_index.retrieval.recursive_decompose import (
    build_refine_prompt,
    decompose_query_with_dependency_hints,
    build_plan_prompt,
    plan_query_steps,
    retrieve_dependency_hinted_decomposed,
    retrieve_recursive_decomposed,
    retrieve_refined_decomposed,
    score_tools,
)
from tool_index.schema import Node, ToolDescriptor, Tree


class _ScriptedLLM:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.prompts: list[str] = []

    def call(self, prompt: str, *, schema: str = "") -> str:
        self.prompts.append(prompt)
        if not self.responses:
            return "[]"
        return self.responses.pop(0)


class _FakeEmbedder:
    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors

    def embed(self, text: str) -> list[float]:
        return self.vectors[text]


def _tree() -> Tree:
    root = Node(id="root", level="L0", description="root", embedding=[0.0, 0.0], children=["group_a", "group_b"])
    group_a = Node(id="group_a", level="L3", description="alpha", embedding=[1.0, 0.0], children=["tool_a"])
    group_b = Node(id="group_b", level="L3", description="beta", embedding=[0.0, 1.0], children=["tool_b"])
    return Tree(root=root, nodes_by_id={"root": root, "group_a": group_a, "group_b": group_b})


def _descriptors() -> dict[str, ToolDescriptor]:
    return {
        "tool_a": ToolDescriptor(id="tool_a", name="parse_url", signature="parse_url(url)", original_doc="Parse URL parts."),
        "tool_b": ToolDescriptor(id="tool_b", name="http_get", signature="http_get(url)", original_doc="Fetch a URL."),
        "tool_c": ToolDescriptor(id="tool_c", name="format_json", signature="format_json(text)", original_doc="Format JSON."),
    }


def test_score_tools_returns_maxsim_scores():
    ranked = score_tools(
        [1.0, 0.0],
        ["tool_a", "tool_b"],
        {"tool_a": [[0.8, 0.2], [1.0, 0.0]], "tool_b": [[0.0, 1.0]]},
    )
    assert [item.tool_id for item in ranked] == ["tool_a", "tool_b"]
    assert ranked[0].score == 1.0


def test_recursive_decompose_accepts_confident_initial_step():
    llm = _ScriptedLLM(['[{"text": "parse the URL", "expected_tools": 1}]'])
    embedder = _FakeEmbedder({"parse the URL": [1.0, 0.0]})
    result = retrieve_recursive_decomposed(
        _tree(),
        "parse the URL",
        llm,
        embedder,
        {"tool_a": [[1.0, 0.0]], "tool_b": [[0.5, 0.5]], "tool_c": [[0.0, 1.0]]},
        ["data tools"],
        k=2,
        rerank_k=3,
        beam=2,
        accept_score=0.8,
        accept_margin=0.2,
    )
    assert result.tool_ids[0] == "tool_a"
    assert result.traces[0].decision == "resolved"
    assert result.llm_calls == 1


def test_recursive_decompose_refines_weak_step():
    llm = _ScriptedLLM([
        '[{"text": "fetch the page at the URL", "expected_tools": 2}]',
        '[{"text": "parse the URL", "expected_tools": 1}, {"text": "fetch the URL", "expected_tools": 1}]',
    ])
    embedder = _FakeEmbedder(
        {
            "fetch the page at the URL": [0.7, 0.7],
            "parse the URL": [1.0, 0.0],
            "fetch the URL": [0.0, 1.0],
        }
    )
    result = retrieve_recursive_decomposed(
        _tree(),
        "take this URL, fetch it, and show the response",
        llm,
        embedder,
        {"tool_a": [[1.0, 0.0]], "tool_b": [[0.0, 1.0]], "tool_c": [[0.2, 0.8]]},
        ["network and parsing"],
        k=2,
        rerank_k=3,
        beam=2,
        max_circles=2,
        accept_score=0.8,
        accept_margin=0.1,
    )
    assert set(result.tool_ids) >= {"tool_a", "tool_b"}
    assert result.traces[0].decision == "refine"
    assert result.traces[0].reason == "expected_multi_tool_step"
    assert result.traces[0].refined_into == ["parse the URL", "fetch the URL"]
    assert result.traces[0].refined_expected_tools == [1, 1]
    assert result.llm_calls == 2


def test_recursive_decompose_refines_confident_multi_intent_step():
    llm = _ScriptedLLM([
        '[{"text": "parse the URL and fetch the URL", "expected_tools": 1}]',
        '[{"text": "parse the URL", "expected_tools": 1}, {"text": "fetch the URL", "expected_tools": 1}]',
    ])
    embedder = _FakeEmbedder(
        {
            "parse the URL and fetch the URL": [1.0, 0.0],
            "parse the URL": [1.0, 0.0],
            "fetch the URL": [0.0, 1.0],
        }
    )
    result = retrieve_recursive_decomposed(
        _tree(),
        "parse and fetch",
        llm,
        embedder,
        {"tool_a": [[1.0, 0.0]], "tool_b": [[0.0, 1.0]]},
        [],
        k=2,
        rerank_k=2,
        beam=2,
        max_circles=2,
        accept_score=0.8,
        accept_margin=0.1,
    )
    assert result.traces[0].reason == "multi_intent_text"
    assert set(result.tool_ids) == {"tool_a", "tool_b"}


def test_recursive_decompose_marks_unresolved_at_max_circles():
    llm = _ScriptedLLM(['[{"text": "ambiguous workflow", "expected_tools": 1}]'])
    embedder = _FakeEmbedder({"ambiguous workflow": [0.7, 0.7]})
    result = retrieve_recursive_decomposed(
        _tree(),
        "ambiguous workflow",
        llm,
        embedder,
        {"tool_a": [[0.7, 0.7]], "tool_b": [[0.7, 0.7]]},
        [],
        k=2,
        rerank_k=2,
        beam=2,
        max_circles=1,
        accept_score=0.8,
        accept_margin=0.2,
    )
    assert result.tool_ids == []
    assert result.unresolved_steps[0].reason == "low_margin_max_circles"


def test_recursive_decompose_respects_refinement_budget():
    llm = _ScriptedLLM([
        '[{"text": "ambiguous one", "expected_tools": 1}, {"text": "ambiguous two", "expected_tools": 1}]',
        '[{"text": "parse the URL", "expected_tools": 1}]',
    ])
    embedder = _FakeEmbedder(
        {
            "ambiguous one": [0.7, 0.7],
            "ambiguous two": [0.7, 0.7],
            "parse the URL": [1.0, 0.0],
        }
    )
    result = retrieve_recursive_decomposed(
        _tree(),
        "two ambiguous operations",
        llm,
        embedder,
        {"tool_a": [[0.7, 0.7]], "tool_b": [[0.7, 0.7]]},
        [],
        k=2,
        rerank_k=2,
        beam=2,
        max_circles=2,
        max_refinements=1,
        accept_score=0.8,
        accept_margin=0.2,
    )
    assert result.llm_calls == 2
    assert any(trace.reason == "low_margin_max_refinements" for trace in result.unresolved_steps)


def test_recursive_decompose_rejects_refinement_that_loses_coverage():
    llm = _ScriptedLLM([
        '[{"text": "parse and fetch the URL", "expected_tools": 2}]',
        '[{"text": "parse the URL", "expected_tools": 1}]',
    ])
    embedder = _FakeEmbedder({"parse and fetch the URL": [0.7, 0.7]})
    result = retrieve_recursive_decomposed(
        _tree(),
        "parse and fetch the URL",
        llm,
        embedder,
        {"tool_a": [[0.7, 0.7]], "tool_b": [[0.7, 0.7]]},
        [],
        k=2,
        rerank_k=2,
        beam=2,
        max_circles=2,
        accept_score=0.8,
        accept_margin=0.2,
    )
    assert result.tool_ids == []
    assert result.unresolved_steps[0].reason == "refine_empty"
    assert result.traces[0].reason == "refine_empty"
    assert result.traces[0].refined_into == []


def test_refined_decompose_keeps_original_sub_query_pool_and_adds_refinement():
    llm = _ScriptedLLM([
        '["parse the URL and fetch the URL"]',
        '[{"text": "parse the URL", "expected_tools": 1}, {"text": "fetch the URL", "expected_tools": 1}]',
    ])
    embedder = _FakeEmbedder(
        {
            "parse the URL and fetch the URL": [1.0, 0.0],
            "parse the URL": [1.0, 0.0],
            "fetch the URL": [0.0, 1.0],
        }
    )
    result = retrieve_refined_decomposed(
        _tree(),
        "parse and fetch",
        llm,
        embedder,
        {"tool_a": [[1.0, 0.0]], "tool_b": [[0.0, 1.0]]},
        [],
        k=2,
        rerank_k=2,
        beam=2,
        max_circles=2,
        accept_score=0.8,
        accept_margin=0.1,
    )
    assert set(result.tool_ids) == {"tool_a", "tool_b"}
    assert result.traces[0].reason == "multi_intent_text"
    assert result.traces[0].refined_into == ["parse the URL", "fetch the URL"]
    assert result.llm_calls == 2


def test_dependency_hinted_decompose_parses_string_steps():
    llm = _ScriptedLLM(['["get the current date and time", "return weekday for current date"]'])
    steps = decompose_query_with_dependency_hints("what weekday is today?", llm, ["time tools"])
    assert steps == ["get the current date and time", "return weekday for current date"]
    assert "Implicit dependency rules" in llm.prompts[0]


def test_dependency_hinted_retrieval_unions_dependency_steps():
    llm = _ScriptedLLM(['["parse the URL", "fetch the URL"]'])
    embedder = _FakeEmbedder({"parse the URL": [1.0, 0.0], "fetch the URL": [0.0, 1.0]})
    result = retrieve_dependency_hinted_decomposed(
        _tree(),
        "parse and fetch",
        llm,
        embedder,
        {"tool_a": [[1.0, 0.0]], "tool_b": [[0.0, 1.0]]},
        [],
        k=2,
        rerank_k=2,
        beam=2,
    )
    assert set(result.tool_ids) == {"tool_a", "tool_b"}
    assert [trace.text for trace in result.traces] == ["parse the URL", "fetch the URL"]
    assert result.llm_calls == 1


def test_plan_query_steps_accepts_legacy_string_list():
    llm = _ScriptedLLM(['["parse the URL"]'])
    steps = plan_query_steps("parse", llm, [])
    assert [(step.text, step.expected_tools) for step in steps] == [("parse the URL", 1)]


def test_build_plan_prompt_includes_hard_few_shot_examples():
    prompt = build_plan_prompt("fetch and hash", ["network tools"])
    assert "Fetch this API URL" in prompt
    assert "parse the provided URL" in prompt
    assert "validate the ID scan image" in prompt
    assert "network tools" in prompt


def test_build_refine_prompt_includes_failure_and_candidates():
    prompt = build_refine_prompt(
        "fetch page",
        2,
        "low_margin",
        score_tools([1.0, 0.0], ["tool_a"], {"tool_a": [[1.0, 0.0]]}),
        ["network tools"],
        _descriptors(),
    )
    assert "low_margin" in prompt
    assert "Expected tool calls" in prompt
    assert "Fetch this API URL" in prompt
    assert "parse_url" in prompt
    assert "network tools" in prompt
