# 02 — Glue-tool prompt hints: results

**Date:** 2026-05-01
**Snapshot:** `raw-tools/v8` (200 tools)
**Pipeline:** beam=3, rerank_k=20, k=10 (unchanged)
**Verdict:** **ARCHIVE** — partial complex improvement, but unacceptable simple-recall regression.

## Headline numbers

| Metric | Production (no hints) | Iter 1 hints | Iter 2 hints | Decision rule |
|---|---|---|---|---|
| SIMPLE recall@10 | 0.960 | 0.960 | **0.920** | drop > 0.01 → ARCHIVE |
| COMPLEX set-recall@10 | 0.908 | 0.908 | 0.925 | <0.97 → would iterate |
| COMPLEX full-cover | 0.800 | 0.860 | **0.880** | meaningful gain |
| ULTRA set-recall@10 | — | 0.837 | 0.823 | (no baseline to gate) |
| ULTRA full-cover | — | 0.660 | 0.600 | (no baseline to gate) |

## What worked

- **Complex full-cover +8 points** (0.800 → 0.880) — 4 of 10 original complex misses recovered. Most of the calculator/datetime hints landed on the right cases.
- The "explicit example" phrasing in iter 2 ("must produce TWO sub-queries") fixed iter 1's substitution bug for `day_of_week`.

## What broke

**Simple recall regression of 0.040** is the killer. The new simple misses are not the same as production's:

| Query | Production | Iter 2 |
|---|---|---|
| "I copied this URL — what site does it actually point to?" → `url_parse` | hit | **miss** |
| "Make me a unique id for a new user record" → `generate_uuid` | miss | miss (unchanged) |
| "I need details about my last gift card redemption" → `reward__...get_order_information` | miss | miss (unchanged) |
| "Get me the SpaceX data for that first endpoint we use" → `reward__spacex_api__229...` | hit | **miss** |

Two new simple regressions. Both look like **prompt-perturbation noise** rather than a hint-rule firing wrong: the URL hint mentions "if user wants contents, append `parse the URL` AND `fetch...`", and the inspection shows the LLM correctly returned the unchanged single-element list — yet `url_parse` no longer ranks in top-10. The longer prompt and the `_GLUE_CHECKLIST` text appears to slightly shift LLM output even when no rule applies, which then propagates to slightly different sub-query phrasing in the unchanged case → slightly different embedding → loses by a tiebreaker against unrelated tools at rank 10.

The SpaceX miss is the same ordinal-handling problem as before, now amplified.

**ULTRA cases regressed too** between iter 1 and iter 2 — the more aggressive "MUST" wording prompted over-decomposition (avg 2.43 sub-queries vs production's 1.69), and the union-retrieval pool got noisier, pushing real gold tools below rank 10.

## Diagnosis

This approach has a fundamental problem: **modifying the decomposer prompt is a global change**, but the misses are local to specific query patterns. Even with carefully scoped if-then rules, the larger prompt changes baseline LLM output enough to hurt the 90% of queries that didn't need the hints.

The ceiling result (Spike 01) said "decomposition is the bottleneck" — but it didn't say "fix it inside the LLM prompt." The LLM is fragile to prompt edits in ways that propagate beyond the targeted rules.

## What to try next (Spike 03)

**Pattern-match injector** is now the priority approach. Instead of changing the LLM prompt:

1. Run the production decomposer **unchanged** to get the LLM's baseline sub-queries.
2. Run a separate, deterministic regex/keyword pass over the **original query** (not the LLM output) to detect glue-tool signals: `r"\bmile|km|kg|lb|°[CF]|percent|%"`, `\bnow|today|current|right now\b`, `\burl\b|http`, etc.
3. For each match, append the matching tool's `intent_phrase` (or its enrichment-derived short query) to the sub-query list.
4. Pass the union to `retrieve_decomposed`.

Advantages over prompt hints:
- **No regression on cases that don't match** — if no keyword fires, the pipeline runs identically to production.
- **Fully deterministic** — no LLM nondeterminism amplifying the change.
- **Cheap to tune** — adding/removing a regex pattern is one line.
- **Easy to ablate** — turn it off per-query for measurement.

Risk: false-positive matches (e.g. "miles" in "Miles Davis songs" injects `unit_converter`). Mitigation: score the injected sub-query against the catalog before adding — if the top match's similarity is below threshold, drop the injection.

## Files

- `hypothesis.md` — the design and decision rule
- `decompose_with_hints.py` — the forked decomposer (kept for reference; not promoted)
- `run_eval.py` — self-contained eval runner that imports the hinted decomposer
- `decompositions.jsonl` — per-query LLM output from the iter 2 run, useful for diagnosing the regression patterns
