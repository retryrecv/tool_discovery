# tool_index

Hierarchical Tool Index construction pipeline for scaling LLM tool use to
large catalogs (~10k tools). Implements the 6-stage build described in
[the shared design conversation](https://claude.ai/share/d8283d5c-9504-4ef7-a076-2ad9b765c722):

1. **Normalize** raw tool definitions into a uniform `ToolDescriptor`.
2. **Enrich** each tool with intent phrase, IO kinds, synonyms, example queries.
3. **Cluster leaves** (tools → L3 groups) using HDBSCAN-style density clustering.
4. **Cluster upward** recursively on child descriptions (L3 → L2 → L1).
5. **Validate** — structural, sibling discriminability, synthetic recall@k.
6. **Freeze** into an immutable, versioned snapshot.

## Install

```bash
pip install -e .
```

## Build a tree

```bash
python -m tool_index build \
  --config configs/default.yaml \
  --input tests/fixtures/mini_tools.json \
  --output data/snapshots
```

Outputs `data/snapshots/v0/{tree.json, embeddings.json, build_trace.json, seed_eval_set.jsonl}`.

## Tests

```bash
pytest tests/ -x
```

## Layout

- `src/tool_index/schema/` — data contracts (ToolDescriptor, Enrichment, Node, Tree).
- `src/tool_index/providers/` — LLM / embedding adapters; fake providers for tests.
- `src/tool_index/pipeline/` — the six stages plus `orchestrator.py::build_tree_index`.
- `src/tool_index/clustering/` — agglomerative + HDBSCAN-emulating clusterers.
- `src/tool_index/labeling/` — contrastive cluster description.
- `src/tool_index/validation/` — structural, discriminability, recall checks.
- `src/tool_index/retrieval/` — minimal top-down traverser (for validation).
- `src/tool_index/storage/` — snapshot read/write, versioning.
