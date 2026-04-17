"""Stage 2 — LLM-enrich every `ToolDescriptor` with retrieval hints.

For each tool we ask the enricher LLM to produce an `Enrichment`
(intent phrase, I/O kinds, synonyms, example queries). Results are cached
on disk so re-runs on the same catalog skip the LLM.

This is the most LLM-expensive stage (one call per tool). Failures fall
back to a degenerate but usable enrichment so the pipeline never hard-stops
on a single malformed response.
"""
from __future__ import annotations
import json

from ..schema import ToolDescriptor, Enrichment
from ..utils.batching import chunks
from ..providers import LLMProvider, DiskCache

from .. import prompts as prompt_pkg


def enrich_all(
    descriptors: list[ToolDescriptor],
    llm: LLMProvider,
    cache: DiskCache | None = None,
    batch_size: int = 20,
) -> dict[str, Enrichment]:
    """Produce one `Enrichment` per descriptor, keyed by tool ID.

    Args:
        descriptors: Output of stage 1.
        llm: `LLMProvider` to call for each tool. Cache keys include
            ``llm.id`` so switching models invalidates cached entries.
        cache: Optional `DiskCache`; pass ``None`` to force fresh calls.
        batch_size: Size of the outer iteration chunk. Does *not* batch
            LLM calls — currently one request per tool, but the chunk
            boundary is where we'd add concurrency later.

    Returns:
        ``{tool_id: Enrichment}`` — one entry per descriptor, always.
    """
    template = prompt_pkg.load("enrich_tool.txt")
    out: dict[str, Enrichment] = {}
    for batch in chunks(descriptors, batch_size):
        for d in batch:
            prompt = template.format(name=d.name, signature=d.signature, doc=d.original_doc)
            # Cache hit avoids an LLM round-trip entirely.
            cached = cache.get(llm.id, prompt) if cache is not None else None
            if cached is not None:
                raw = cached
            else:
                raw = llm.call(prompt, schema="enrich_tool")
                if cache is not None:
                    cache.put(llm.id, prompt, raw)
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                # Degraded fallback: the tool still gets *an* enrichment so
                # clustering isn't blocked. Quality suffers, but the
                # pipeline completes and the issue shows up in validation.
                data = {
                    "intent_phrase": d.name,
                    "input_kind": "input",
                    "output_kind": "output",
                    "synonyms": [],
                    "example_queries": [d.name],
                }
            out[d.id] = Enrichment.from_dict(data)
    return out
