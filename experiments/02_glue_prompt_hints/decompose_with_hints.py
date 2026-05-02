"""Forked decomposer prompt with glue-tool hints.

Same `decompose_query` interface as `tool_index.retrieval.decompose` —
import this module's `decompose_query` and inject it via monkey-patch
in `run_eval.py` to compare against production.

The only change is `build_prompt`: appends a "Glue-tool checklist" block
that gives the LLM explicit if-then rules for naming implicit utility
tools (`calculator`, `get_current_datetime`, `url_parse`, `http_get`,
`json_query`, `json_format`).

Source rules derived from the 10 complex misses in the 2026-05-01 eval —
see ../01_glue_tool_ceiling/results.md for the diagnosis.
"""
from __future__ import annotations

import json
import re

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL | re.IGNORECASE)


def _strip_fence(text: str) -> str:
    m = _FENCE_RE.match(text)
    return m.group(1) if m else text


_GLUE_CHECKLIST = """
Glue-tool checklist — users often skip naming the helper tools they
implicitly need. AFTER you list every operation the user explicitly
requested (paraphrased), APPEND extra sub-queries for the helpers
below when they apply. The original operations MUST stay in the
output — never replace them with a helper.

For example, "What day of the week is today?" must produce TWO
sub-queries: ["what day of the week is it", "get the current date and
time"] — not just ["get the current date and time"].

Helpers to append (each gets its own sub-query string):
- Arithmetic / totalling / unit conversion / numeric comparison
  (signals: "convert", "how many", "total", "average", "in weeks",
  "at X km/h", "percent", "share") → append: "calculate the result"
- "Now", "today", "current time/date", "what day", "right now",
  relative dates ("until New Year's", "in 3 days") → append:
  "get the current date and time"
- A URL is mentioned and the user wants its contents → append BOTH:
  "parse the URL into components" AND "fetch the URL contents over HTTP"
  (Exception: if the user only asks WHAT/WHERE the URL points to with no
  fetch verb, append only "parse the URL into components".)
- JSON manipulation — extracting a field, querying a value, picking
  a key, "pull the X field" → append: "query a value out of JSON"
- JSON output that should be human-readable ("pretty-print", "show me",
  "make readable", "format") → append: "format JSON for display"
- A list of records that needs summing / averaging / ordering
  → append "calculate the result" or "sort the array" as appropriate
"""


def build_prompt(query: str, schema_lines: list[str]) -> str:
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
        f"{_GLUE_CHECKLIST}"
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
