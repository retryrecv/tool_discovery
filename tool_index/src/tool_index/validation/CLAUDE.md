# validation

Post-build checks. Stage 5 runs these; fatal failures raise `ValidationError`.

- `structural.py` — tree invariants: depth, fanout bounds, every leaf has a tool, no orphans.
- `discriminability.py` — sibling descriptions must be distinguishable (pairwise embedding distance above threshold).
- `synthetic_queries.py` — LLM-generated eval queries per tool (the "seed eval set").
- `recall_benchmark.py` — run traverser on synthetic queries, measure recall@k.
- `report.py` — aggregate results into a human-readable report + `build_trace.json` entry.

## Conventions

- Each validator returns a `ValidationResult` (pass/fail + details) — orchestrator decides fatality.
- Thresholds come from config, not hardcoded.
- Synthetic queries are cached — changing the prompt or model invalidates the cache.
