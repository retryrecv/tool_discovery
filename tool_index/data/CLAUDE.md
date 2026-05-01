# data

Pipeline inputs and outputs. Most of this is gitignored.

- `rawTools/` — TypeScript source catalogs (`*.tools.ts`).
- `generateTools/` — generated `tools.py` (the `raw_tools` list consumed by stage 1) and `test_cases.py` (natural-language query fixtures used by `scripts/eval_real_cases.py`).
- `cache/` — provider cache (LLM + embedding responses), keyed by `(provider, model, input_hash)`.
- `snapshots/` — frozen tree outputs, one subdir per version (`v0/`, `v1/`, ...).

## Conventions

- Never commit cache or snapshot contents — they're reproducible from code + config + raw inputs.
- Deleting `cache/` is always safe; it will be rebuilt on next run (at the cost of API calls).
