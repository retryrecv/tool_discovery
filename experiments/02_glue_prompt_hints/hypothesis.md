# 02 — Glue-tool prompt hints

## Hypothesis

Spike 01 proved the entire 9.2/20-point complex recall gap is decomposition-side. The miss pattern is uniform: the LLM decomposer fails to enumerate **implicit utility tools** that the user assumes but doesn't name.

Looking at the 10 complex misses from the 2026-05-01 production eval:

| Pattern | Missing tool | Count |
|---|---|---|
| User asks math/conversion/totalling without saying "calculate" | `calculator` | 4 |
| User says "what day/time" without saying "current datetime" | `get_current_datetime` | 2 |
| User mentions a URL without saying "parse" | `url_parse` | 1 |
| User asks JSON manipulation without naming `json_query` | `json_query` | 1 |
| User says "fetch the page" but next step needs HTTP | `http_get` | 1 |
| Other (SpaceX ordering ambiguity) | — | 1 |

If we add **explicit if-then rules** to the decomposer prompt for the top ~5 glue-tool patterns, complex set-recall should jump from 0.908 toward the 1.000 ceiling.

## What changes

Single edit: append a "Glue-tool checklist" block to `build_prompt` in `tool_index/src/tool_index/retrieval/decompose.py`. Pseudocode:

```
Glue-tool checklist (always include the matching helper as its own sub-query):
- If a sub-query involves arithmetic, totalling, conversion between units, or numeric comparison → also include `calculator`.
- If the query refers to "now", "today", "current time", "what day", or relative dates → also include `get current datetime`.
- If a URL is mentioned and the user wants the contents → also include `parse the URL` AND `fetch the URL via HTTP`.
- If the query manipulates JSON (extract a field, pretty-print, query a value) → include both `format the JSON` and `query a JSON field` as separate sub-queries.
- If a sub-query mentions a list of records that needs to be summed/averaged/ordered → also include `calculator` or `sort an array` as appropriate.
```

The actual phrasing will go through 1-2 iterations against the 10-miss set before measuring full eval.

## Method

1. Fork the prompt: keep current generic version, add the glue-tool block.
2. Run the modified prompt through the full eval (`eval_real_cases.py --decompose`) on the same `raw-tools/v8` snapshot.
3. **Measure both directions**:
   - Recall delta on the 10 complex misses (target: catch ≥ 7 of them)
   - Recall delta on the other 90 cases (guard: must not drop more than 0.005)
4. Inspect false-positive injections: did the glue-tool checklist add `calculator` to "send me a kitten picture"? If yes, prune the rule.

## Decision rule

| Outcome | Verdict |
|---|---|
| Complex set-recall ≥ 0.97 AND simple recall unchanged (±0.01) | **PROMOTE**: replace prompt in production decompose.py, commit. |
| Complex set-recall 0.93–0.96 | **ITERATE**: tune wording / add a rule, re-measure once. Cap at 2 iterations. |
| Complex set-recall < 0.93 OR simple recall drops > 0.01 | **ARCHIVE** as failed. The if-then format isn't enough — escalate to Spike 03 (regex pattern injector) or a paper-driven approach (Self-Ask refinement). |

## Implementation note

Don't edit `decompose.py` in place during the spike. Make a copy under `experiments/02_glue_prompt_hints/decompose_with_hints.py` that exports the same `decompose_query` symbol, and a `run_eval.py` that monkey-patches `tool_index.retrieval.decompose_query` at import time. This keeps `tool_index/` clean until the decision rule says PROMOTE.

## Cost estimate

- ~100 LLM decomposer calls per eval run
- Cache-friendly (same query → same prompt → cached)
- ~30 sec per re-run after first
- 2-3 prompt iterations max

## Out of scope

- The 1 SpaceX miss ("first endpoint and the second one for comparison") — that's a numbering / ordinal-handling problem, separate from glue tools. If the hints incidentally fix it, great; if not, leave for a later spike.
- ULTRA_COMPLEX cases — these are decomposition-stress tests with longer chains. Track the metric but don't optimize for it; if the complex hints help, they help.
- Changing `schema_lines` injection (the L1/L2 hint already in the prompt). That's a separate axis.
