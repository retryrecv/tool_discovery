# tool_index

Hierarchical Tool Index construction pipeline. Six stages:

1. **Normalize** raw tool defs → `ToolDescriptor`.
2. **Enrich** each tool (intent phrase, IO kinds, synonyms, examples).
3. **Cluster leaves** — tools → L3 groups (HDBSCAN-style).
4. **Cluster upward** — L3 → L2 → L1 on child descriptions.
5. **Validate** — structural, sibling discriminability, synthetic recall@k.
6. **Freeze** — immutable versioned snapshot.

## Commands

```bash
pip install -e .                                  # install
pytest tests/ -x                                  # run tests
python -m tool_index build \
  --config configs/default.yaml \
  --input tests/fixtures/mini_tools.json \
  --output data/snapshots                         # build a tree
```

## Layout

- `src/tool_index/` — package source (see per-module CLAUDE.md).
- `tests/` — unit, integration, golden.
- `configs/` — YAML configs (default/dev/prod).
- `data/` — raw inputs, cache, snapshots (gitignored outputs).
- `scripts/` — one-off utilities.

## Conventions

- Orchestrator entrypoint is `pipeline.orchestrator.build_tree_index`.
- All LLM/embedding access goes through `providers/` — never import SDKs directly elsewhere.
- Determinism matters: use seeded RNGs, stable hashing (`utils.hashing`), stable IDs (`utils.ids`).
