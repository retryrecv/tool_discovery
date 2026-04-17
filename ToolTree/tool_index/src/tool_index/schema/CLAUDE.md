# schema

Data contracts used everywhere else. Keep this module **pure data** — no I/O, no LLM calls, no clustering logic.

- `descriptor.py` — `ToolDescriptor`: normalized tool definition (id, name, description, params).
- `enrichment.py` — `Enrichment`: LLM-generated fields (intent_phrase, io_kinds, synonyms, example_queries).
- `node.py` — `Node`: a tree node (L1/L2/L3/leaf) with children, description, embedding.
- `tree.py` — `Tree`: root + node index + metadata.
- `constants.py` — enums / magic strings (level names, IO kinds, etc.).

## Conventions

- Prefer frozen dataclasses or pydantic `BaseModel` with `frozen=True` for hashability.
- Breaking changes here ripple through snapshots — bump the snapshot schema version in `storage/versioning.py` when fields change.
