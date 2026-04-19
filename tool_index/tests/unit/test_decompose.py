"""Unit tests for the LLM query decomposer."""
from __future__ import annotations

import pytest

from tool_index.retrieval import decompose_query
from tool_index.retrieval.decompose import build_prompt


class _ScriptedLLM:
    """Returns canned responses in order; records prompts for inspection."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.prompts: list[str] = []

    def call(self, prompt: str, *, schema: str = "") -> str:
        self.prompts.append(prompt)
        if not self._responses:
            raise RuntimeError("No more scripted responses")
        return self._responses.pop(0)


def test_decompose_splits_multi_step_query():
    llm = _ScriptedLLM(['["fetch the API", "format the JSON", "hash the price"]'])
    out = decompose_query(
        "Fetch the API, format the JSON, then hash the price",
        llm,
        ["data transformation", "hashing"],
    )
    assert out == ["fetch the API", "format the JSON", "hash the price"]


def test_decompose_passes_atomic_query_through():
    llm = _ScriptedLLM(['["pretty-print this JSON"]'])
    out = decompose_query("pretty-print this JSON", llm, ["data transformation"])
    assert out == ["pretty-print this JSON"]


def test_decompose_strips_markdown_fence():
    llm = _ScriptedLLM(['```json\n["one", "two"]\n```'])
    out = decompose_query("a then b", llm, [])
    assert out == ["one", "two"]


def test_decompose_falls_back_on_invalid_json():
    llm = _ScriptedLLM(["not json at all"])
    q = "some query"
    out = decompose_query(q, llm, [])
    assert out == [q]


def test_decompose_falls_back_on_llm_failure():
    class _BoomLLM:
        def call(self, prompt: str, *, schema: str = "") -> str:
            raise RuntimeError("provider down")

    q = "some query"
    out = decompose_query(q, _BoomLLM(), [])
    assert out == [q]


def test_decompose_caps_max_sub_queries():
    llm = _ScriptedLLM(['["a", "b", "c", "d", "e"]'])
    out = decompose_query("q", llm, [], max_sub_queries=3)
    assert out == ["a", "b", "c"]


def test_build_prompt_includes_schema_lines():
    p = build_prompt("do x and y", ["domain alpha", "domain beta"])
    assert "domain alpha" in p
    assert "domain beta" in p
    assert "do x and y" in p


def test_build_prompt_handles_empty_schema():
    p = build_prompt("q", [])
    assert "no schema provided" in p
