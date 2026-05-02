# 03 — Pattern-match glue-tool injector

## Hypothesis

Spike 02 (prompt hints) demonstrated that modifying the LLM decomposer
prompt is a **global perturbation**: the longer prompt shifted baseline
LLM output even on unchanged single-intent queries, dropping simple
recall by 0.040 in exchange for +0.080 complex full-cover. ARCHIVED.

The miss diagnosis from Spike 01 still holds: the bottleneck is naming
implicit utility tools (`calculator`, `get_current_datetime`,
`url_parse`, `http_get`, `json_query`, etc.). What we need is a way to
inject those without touching the LLM call.

**Proposal:** keep the production decomposer prompt **unchanged**, then
run a **deterministic regex pass** over the **original query** (not the
LLM output). For each pattern that fires, append the matching tool's
`intent_phrase` to the sub-query list before union retrieval.

Because the LLM call is byte-identical to production, queries that don't
match any regex run with identical sub-queries → identical embeddings →
identical retrieval → **zero risk of simple-recall regression by
construction** (modulo embedding cache stability for the appended
phrases).

## Pattern → tool mapping

Sourced from the 16 glue-tool `intent_phrase` strings in `raw-tools/v8`
plus the miss patterns from Spike 01:

| Regex (case-insensitive, on original query) | Append intent_phrase | Tool |
|---|---|---|
| `\b(calculate|compute|how many|total|sum|average|mean|percent|%|share of)\b` | `evaluate arithmetic expressions` | `calculator` |
| `\b(convert|in (km|miles|kg|lbs|pounds|°[CF]|celsius|fahrenheit|kilometers|kilograms))\b`, `\bkm/?h\b`, `\bmph\b` | `convert values between common measurement units` | `unit_converter` |
| `\b(now|today|current (date|time)|right now|what (day|time)|tonight|tomorrow|yesterday)\b` | `get the current UTC date and time` | `get_current_datetime` |
| `\bday of (the )?week\b`, `\bwhat day\b` | `get weekday from a date` | `day_of_week` |
| `\b(time ?zone|in (PST|EST|UTC|GMT|JST|CET))\b`, `\bconvert .* to .* time\b` | `convert a datetime between timezones` | `timezone_convert` |
| `\b(days? until|days? since|how long until|between .* and .*)\b` (date context) | `compute time difference between two dates` | `date_diff` |
| `\bhttps?://`, `\bwww\.`, `\b(URL|url)\b` | `parse a URL into components` | `url_parse` |
| query also has fetch/get/download verb + URL match | `fetch content from a URL` | `http_get` |
| `\b(JSON|json)\b` AND (`extract`, `field`, `value`, `key`, `pull`, `pluck`, `path`) | `extract values from JSON by path` | `json_query` |
| `\b(JSON|json)\b` AND (`pretty[- ]?print`, `format`, `readable`, `show`, `display`) | `format JSON` | `json_format` |
| `\bsort\b`, `\b(ascending|descending)\b` AND list context | `sort an array` | `array_sort` |
| `\b(median|stdev|standard deviation|variance|min(imum)?|max(imum)?|mean)\b` | `compute basic descriptive statistics` | `statistics` |
| `\b(percent|percentage|%)\b` AND (`of`, `change`, `increase`, `decrease`) | `calculate percentages` | `percentage_calc` |
| `\b(base64|hex|url[- ]?encode|decode|encode)\b` | `encode or decode text strings` | `encode_decode` |
| `\b(uuid|guid|unique id|unique identifier)\b` | `generate a random UUID` | `generate_uuid` |
| `\b(sha256|hash|fingerprint|checksum)\b` | `compute a SHA-256 hash of text` | `hash_text` |

## Method

1. Implement `pattern_injector.py` exporting
   `inject(query: str, sub_queries: list[str]) -> list[str]`.
2. `run_eval.py` mirrors Spike 02's runner but imports the **stock**
   `tool_index.retrieval.decompose_query` and calls
   `inject(query, sub_queries)` after decomposition, before embedding.
3. Append-only: never replace LLM-produced sub-queries. Dedupe by
   string equality (case-folded) so the same intent isn't appended twice
   if the LLM already named it.

## Decision rule

| Outcome | Verdict |
|---|---|
| SIMPLE recall@10 unchanged (±0.005) AND COMPLEX full-cover ≥ 0.88 | **PROMOTE**: wire `inject` into production retrieval entry point. |
| SIMPLE unchanged AND COMPLEX full-cover 0.84–0.87 | **ITERATE**: tune patterns once (refine false-positive cases), re-measure. |
| SIMPLE drops > 0.005 OR COMPLEX < 0.84 | **ARCHIVE**: deterministic injection alone is insufficient. |

## False-positive guard

The two Spike 01 misses we're least confident about:
- "Miles Davis songs" → would `\bmile|km\b` fire `unit_converter`? Yes.
  Mitigation: require a digit in proximity for unit-converter
  (`\b\d+\s?(km|miles|kg)\b`) or a verb like "convert".
- "Hash my crypto wallet address" vs `\bhash\b` → fires `hash_text`,
  which is probably correct here. Leave.

We will measure false positives by inspecting the diff between
production sub-queries and post-injection sub-queries on the SIMPLE set.

## Out of scope

- The SpaceX ordinal miss (it's not a glue-tool problem).
- ULTRA cases (track but don't optimize).
- Score-the-injection-against-catalog mitigation (mentioned in Spike 02
  results) — only add if false-positive rate looks bad after iter 1.

## Cost estimate

- Same number of LLM calls as production (decomposer unchanged).
- Adds 0–N embedding calls per query for appended sub-queries; embedder
  cache makes repeats free.
- Eval run: ~30 sec (same as Spike 02).
