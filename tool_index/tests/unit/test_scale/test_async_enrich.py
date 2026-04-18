from __future__ import annotations

import asyncio
import json

import pytest

from tool_index.scale import enrich_all_async, AsyncEnrichConfig
from tool_index.schema import ToolDescriptor


class _FakeAsyncLLM:
    id = "fake-llm"

    def __init__(self, responses: dict[str, str], *, fail_first_n: int = 0):
        self.responses = responses
        self.calls = 0
        self.fail_first_n = fail_first_n

    async def acall(self, prompt: str, schema: str | None = None) -> str:
        self.calls += 1
        if self.calls <= self.fail_first_n:
            raise RuntimeError("transient")
        for name, body in self.responses.items():
            if name in prompt:
                return body
        return json.dumps({
            "intent_phrase": "x", "input_kind": "i", "output_kind": "o",
            "synonyms": [], "example_queries": ["x"],
        })


def _td(name: str) -> ToolDescriptor:
    return ToolDescriptor(
        id=f"tool_{name}", name=name, signature=f"{name}() -> str",
        original_doc=f"does {name}",
    )


def test_concurrent_enrichment_returns_one_per_tool() -> None:
    descriptors = [_td(f"t{i}") for i in range(8)]
    llm = _FakeAsyncLLM({})
    out = asyncio.run(enrich_all_async(descriptors, llm, config=AsyncEnrichConfig(concurrency=4)))
    assert len(out) == 8
    assert llm.calls == 8


def test_retries_then_succeeds() -> None:
    descriptors = [_td("alpha")]
    llm = _FakeAsyncLLM({}, fail_first_n=1)
    out = asyncio.run(enrich_all_async(descriptors, llm, config=AsyncEnrichConfig(concurrency=1, retries=2, backoff_base=0.0)))
    assert len(out) == 1
    assert llm.calls == 2


def test_fence_wrapped_response_parses() -> None:
    body = "```json\n" + json.dumps({
        "intent_phrase": "wrapped", "input_kind": "i", "output_kind": "o",
        "synonyms": [], "example_queries": ["q"],
    }) + "\n```"
    descriptors = [_td("alpha")]
    llm = _FakeAsyncLLM({"alpha": body})
    out = asyncio.run(enrich_all_async(descriptors, llm, config=AsyncEnrichConfig(concurrency=1)))
    assert out["tool_alpha"].intent_phrase == "wrapped"
