# src/tool_index

Python package root. Modules:

- `schema/` — dataclasses / pydantic contracts (ToolDescriptor, Enrichment, Node, Tree). Pure data, no logic.
- `providers/` — LLM + embedding adapters; fakes for tests.
- `pipeline/` — the six build stages + orchestrator.
- `clustering/` — agglomerative + HDBSCAN-emulating clusterers, neighbor graphs, rebalancing.
- `labeling/` — contrastive cluster description via LLM.
- `validation/` — structural / discriminability / recall checks.
- `retrieval/` — top-down traverser (used by validation + as reference).
- `storage/` — snapshot read/write, format, versioning.
- `prompts/` — prompt templates.
- `config/` — YAML config loader.
- `utils/` — batching, hashing, ID generation, logging.
- `cli.py` / `__main__.py` — CLI entrypoint (`python -m tool_index ...`).

## Import rules

- `schema` depends on nothing.
- `providers`, `utils`, `config` depend only on `schema`.
- `pipeline` is the only module that orchestrates across others.
- Never import from `pipeline` in leaf modules — keeps the DAG clean.
