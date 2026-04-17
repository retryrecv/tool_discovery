"""Synthetic eval query generation.

For each tool, asks the labeler LLM to produce ``per_tool`` realistic
natural-language queries that should route to it. These become the seed
eval set used by the recall benchmark. Caching the prompt inputs (via the
provider cache) keeps repeated runs cheap.

Fallback: if the LLM returns invalid JSON, we reuse the enrichment's
``example_queries`` so the benchmark still has something to work with.
"""
from __future__ import annotations
import json

from ..schema import Enrichment
from .. import prompts as prompt_pkg


def generate_synthetic_queries(
    enrichments: dict[str, Enrichment],
    llm,
    per_tool: int,
) -> list[dict]:
    """Generate ``per_tool`` eval queries per tool.

    Args:
        enrichments: Stage 2 output, keyed by tool ID.
        llm: Labeler `LLMProvider` that produces the queries.
        per_tool: Requested query count per tool. LLM may return fewer —
            we truncate but don't pad.

    Returns:
        Flat list of ``{"tool_id", "query"}`` rows.
    """
    template = prompt_pkg.load("synthesize_queries.txt")
    out: list[dict] = []
    for tid, enr in enrichments.items():
        prompt = template.format(intent=enr.intent_phrase, n=per_tool)
        resp = llm.call(prompt, schema="synthesize_queries").strip()
        try:
            queries = json.loads(resp)
        except json.JSONDecodeError:
            # Degraded fallback — the enrichment's example_queries were
            # generated from the same tool and are usually fine as eval
            # stand-ins. Benchmark recall will still be meaningful.
            queries = enr.example_queries[:per_tool]
        for q in queries[:per_tool]:
            out.append({"tool_id": tid, "query": q})
    return out
