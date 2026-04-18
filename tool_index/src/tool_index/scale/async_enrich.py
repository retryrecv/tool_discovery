"""Concurrent stage 2 — semaphore-bounded asyncio over enrichment calls.

Synchronous `enrich_all` issues one LLM call per tool serially. At 10k
tools and 1s per call that's ~3 hours. Async with concurrency=32 gets
us to ~5 minutes — provider rate limit becomes the bottleneck instead
of network round-trips.

We don't import the existing `pipeline.stage2_enrich` to avoid the
no-augmentation constraint. Cache contract is the same so existing
disk caches are reused.

Provider must expose `acall(prompt, schema=...) -> str`. If your
provider is sync-only, use `wrap_sync_provider` to run it in a thread
pool.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Awaitable, Callable, Protocol

from ..providers import DiskCache
from ..schema import Enrichment, ToolDescriptor
from .. import prompts as prompt_pkg

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL | re.IGNORECASE)


class AsyncLLM(Protocol):
    id: str
    async def acall(self, prompt: str, schema: str | None = None) -> str: ...


@dataclass(frozen=True)
class AsyncEnrichConfig:
    concurrency: int = 32
    retries: int = 2
    backoff_base: float = 0.5


def wrap_sync_provider(sync_provider) -> AsyncLLM:
    """Adapt a sync `LLMProvider` to `AsyncLLM` via a thread pool."""

    class _Wrapped:
        id = sync_provider.id

        async def acall(self, prompt: str, schema: str | None = None) -> str:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, lambda: sync_provider.call(prompt, schema=schema)
            )

    return _Wrapped()


def _fallback(d: ToolDescriptor) -> dict:
    return {
        "intent_phrase": d.name,
        "input_kind": "input",
        "output_kind": "output",
        "synonyms": [],
        "example_queries": [d.name],
    }


async def _one(
    d: ToolDescriptor,
    template: str,
    llm: AsyncLLM,
    cache: DiskCache | None,
    cfg: AsyncEnrichConfig,
    sem: asyncio.Semaphore,
) -> tuple[str, Enrichment]:
    prompt = template.format(name=d.name, signature=d.signature, doc=d.original_doc)
    cached = cache.get(llm.id, prompt) if cache is not None else None
    if cached is not None:
        raw = cached
    else:
        raw = await _call_with_retry(llm, prompt, cfg, sem)
        if cache is not None:
            cache.put(llm.id, prompt, raw)
    try:
        m = _FENCE_RE.match(raw)
        data = json.loads(m.group(1) if m else raw)
    except json.JSONDecodeError:
        data = _fallback(d)
    return d.id, Enrichment.from_dict(data)


async def _call_with_retry(
    llm: AsyncLLM,
    prompt: str,
    cfg: AsyncEnrichConfig,
    sem: asyncio.Semaphore,
) -> str:
    last_exc: Exception | None = None
    for attempt in range(cfg.retries + 1):
        try:
            async with sem:
                return await llm.acall(prompt, schema="enrich_tool")
        except Exception as e:  # provider-specific; we treat all as retryable
            last_exc = e
            if attempt == cfg.retries:
                break
            await asyncio.sleep(cfg.backoff_base * (2 ** attempt))
    raise RuntimeError(f"enrichment failed after {cfg.retries + 1} attempts: {last_exc}")


async def enrich_all_async(
    descriptors: list[ToolDescriptor],
    llm: AsyncLLM,
    *,
    cache: DiskCache | None = None,
    config: AsyncEnrichConfig | None = None,
) -> dict[str, Enrichment]:
    cfg = config or AsyncEnrichConfig()
    template = prompt_pkg.load("enrich_tool.txt")
    sem = asyncio.Semaphore(cfg.concurrency)
    coros = [_one(d, template, llm, cache, cfg, sem) for d in descriptors]
    pairs = await asyncio.gather(*coros)
    return dict(pairs)
