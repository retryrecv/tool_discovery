# prompts

Prompt templates for all LLM calls (enrichment, labeling, synthetic queries).

## Conventions

- One template per file, named after its use (`enrich_tool.md`, `describe_cluster.md`, etc.).
- Templates are plain text with `{placeholders}` — rendered via `str.format` or equivalent.
- Never inline prompt strings in `labeling/` or `pipeline/` — always load from here. Keeps prompts diffable and cache keys stable.
- Changing a template invalidates provider cache entries that used it.
