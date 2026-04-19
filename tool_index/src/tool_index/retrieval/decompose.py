"""LLM-based query decomposition for multi-step user queries.

A pure parameter-passing step: caller provides the query, an LLM provider,
and a list of schema descriptions (e.g. L2 category one-liners). Returns
a list of atomic sub-queries. No global state, no I/O outside the LLM call.

Pair with ``retrieve_decomposed`` (in ``traverser.py`` style) to fan-out
per sub-query and union the candidate pools.

Reference: Zhou et al. 2022, "Least-to-Most Prompting Enables Complex
Reasoning in Large Language Models" (arXiv:2205.10625).
"""
from __future__ import annotations

import json
import re

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL | re.IGNORECASE)


def _strip_fence(text: str) -> str:
    m = _FENCE_RE.match(text)
    return m.group(1) if m else text


def build_prompt(query: str, schema_lines: list[str]) -> str:
    """Build the decomposition prompt.

    ``schema_lines`` is a list of one-line domain/category descriptions
    (e.g. ``"data transformation, parsing, JSON/CSV/URL handling"``).
    Caller controls which level of the tree to inject — keeps the
    decomposer schema-agnostic.
    """
    schema_block = "\n".join(f"- {s}" for s in schema_lines) if schema_lines else "(no schema provided)"
    return (
        "You break a user query into atomic sub-queries, one per tool call.\n"
        "The tool catalog covers these capability areas:\n"
        f"{schema_block}\n\n"
        "Rules:\n"
        "- One sub-query per atomic operation the user wants.\n"
        "- Phrase each sub-query as a short standalone request.\n"
        "- If the user query is already atomic (one operation), return it unchanged as a single-element list.\n"
        "- Do not invent operations the user did not request.\n"
        "- Output JSON only: a list of strings, e.g. [\"fetch the API\", \"hash the price\"].\n\n"
        f"User query: {query}\n"
        "JSON:"
    )


def decompose_query(
    query: str,
    llm,
    schema_lines: list[str],
    *,
    max_sub_queries: int = 8,
) -> list[str]:
    """Return a list of atomic sub-queries for ``query``.

    Single-intent queries return ``[query]`` unchanged. Falls back to
    ``[query]`` on any LLM/JSON parse failure — decomposition must never
    block retrieval.

    Args:
        query: The user query to decompose.
        llm: LLM provider with a ``call(prompt) -> str`` interface.
        schema_lines: One-line descriptions of the catalog's capability
            areas (caller picks the level: L1 titles, L2 descriptions, etc.).
        max_sub_queries: Cap on returned sub-queries; protects against
            runaway decomposition.

    Returns:
        List of sub-query strings, length in [1, max_sub_queries].
    """
    prompt = build_prompt(query, schema_lines)
    try:
        raw = llm.call(prompt)
    except Exception:
        return [query]

    text = _strip_fence(raw).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [query]

    if not isinstance(parsed, list) or not parsed:
        return [query]

    cleaned = [str(s).strip() for s in parsed if isinstance(s, (str, int, float)) and str(s).strip()]
    if not cleaned:
        return [query]

    return cleaned[:max_sub_queries]
