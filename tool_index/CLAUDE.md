# tool_index

Hierarchical Tool Index construction pipeline. Six stages:

1. **Normalize** raw tool defs → `ToolDescriptor`.
2. **Enrich** each tool (intent phrase, IO kinds, synonyms, examples).
3. **Cluster leaves** — tools → L3 groups (HDBSCAN-style).
4. **Cluster upward** — L3 → L2 → L1 on child descriptions.
5. **Validate** — structural, sibling discriminability, synthetic recall@k.
6. **Freeze** — immutable versioned snapshot.

## Workspace

This package is a member of the uv workspace at the repo root. The shared
`.venv/` and `uv.lock` live at `../`; do not create a per-member venv here.
Run `uv sync` from the repo root to install.

## Commands

Run from this directory (`tool_index/`). The pipeline is split into six
stage scripts so each step can be inspected and re-run in isolation.

```bash
uv run scripts/verify_openai_client.py                    # sanity-check .env
uv run scripts/stage_normalize.py     --run <run-name>
uv run scripts/stage_enrich.py        --run <run-name>
uv run scripts/stage_cluster.py       --run <run-name>    # stages 3 + 4a + 4b + tree assembly
uv run scripts/stage_synth_queries.py --run <run-name>
uv run scripts/stage_validate.py      --run <run-name>
uv run scripts/stage_freeze.py        --run <run-name>
uv run scripts/build_all.py           --run <run-name>    # convenience wrapper
pytest tests/ -x                                          # run tests
```

`<run-name>` is the snapshot subdirectory (e.g. `raw-tools`). Stage
scripts read prior outputs from `data/snapshots/<run-name>/` and write
their own.

## Layout

- `src/tool_index/` — package source (see per-module CLAUDE.md).
- `tests/` — unit, integration, golden.
- `configs/` — YAML configs (default/dev/prod).
- `data/` — raw inputs, cache, snapshots (gitignored outputs).
  - `data/rawTools/` — TS source catalogs (`*.tools.ts`).
  - `data/generateTools/` — generated `tools.py` (the `raw_tools` list)
    and `test_cases.py` (natural-language query fixtures).
  - `data/cache/` — provider cache (LLM + embedding).
  - `data/snapshots/<run-name>/` — per-run intermediate stage files
    (`01_descriptors.json`, `02_enrichments.json`, `tree_draft.json`,
    `04_synth_queries.jsonl`, `05_validation.json`) plus frozen
    versions `v<N>/`.
- `scripts/` — pipeline stage scripts + `_pipeline_config.py` (shared
  config: proxy URL, model, embedder, thresholds) + `inspect_tree.py`.

## Conventions

- Orchestrator entrypoint is `pipeline.orchestrator.build_tree_index`
  (used by tests). Production runs go through the stage scripts.
- All LLM/embedding access goes through `providers/` — never import
  SDKs directly elsewhere.
- LLM responses may be wrapped in markdown code fences; provider and
  parse sites strip them before `json.loads`.
- Determinism matters: use seeded RNGs, stable hashing
  (`utils.hashing`), stable IDs (`utils.ids`).

## Tuning notes

- Cluster `distance_threshold` is **cosine distance**: lower = stricter
  (less merging), higher = looser (more merging).
- Sibling discriminability check is `O(siblings²)` LLM calls per
  non-leaf node. Trees with many singleton clusters at small scale
  produce huge sibling counts — expect long validate runs until
  thresholds are tuned for the catalog size.
- Snapshots are never overwritten; each freeze allocates the next
  `v<N>/` slot.
