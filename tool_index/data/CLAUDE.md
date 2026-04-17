# data

Pipeline inputs and outputs. Most of this is gitignored.

- `raw/` — raw tool catalogs (inputs to stage 1).
- `cache/` — provider cache (LLM + embedding responses), keyed by `(provider, model, input_hash)`.
- `snapshots/` — frozen tree outputs, one subdir per version (`v0/`, `v1/`, ...).

## Conventions

- Never commit cache or snapshot contents — they're reproducible from code + config + raw inputs.
- `raw/` may contain small committed fixtures; large catalogs stay out of git.
- Deleting `cache/` is always safe; it will be rebuilt on next run (at the cost of API calls).
