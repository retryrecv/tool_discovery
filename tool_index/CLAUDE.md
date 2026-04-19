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
  - `scripts/explore/` — one-off scripts from exploratory spikes (kept
    for reference; not part of the production stage chain).
- `explore/` — recall-improvement spike records. Each direction has a
  JSON spec (hypothesis, paper refs, decision rule, result), plus
  `results.md` (the cross-direction scoreboard) and `merge_plan.md`
  (the productization recipe used to land winners). Add new spikes
  here before touching `tasks.json`; promote winners after measurement.

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

## Exploration workflow

Recall-improvement (or any architectural) spikes live under `explore/`
and are kept separate from `tasks.json` until they prove out:

1. Drop a `direction<N>_<name>.json` in `explore/` with hypothesis,
   paper refs, baseline numbers, files-to-touch, ablation plan, and a
   `decision_rule` for promotion.
2. Run each spike in its own git worktree (`git worktree add ../tds-<name>
   explore/<branch>`) so multiple snapshots can coexist on disk and
   `data/cache/` can be shared via symlink for cache reuse.
3. Use a unique `--run` name per spike (e.g. `raw-tools-doc2query`) and
   seed it by copying the relevant `0X_*.json` files from the baseline
   snapshot — never let two spikes write to the same snapshot dir.
4. Update `explore/results.md` (the scoreboard) after each measurement.
5. If `decision_rule` is met, draft entries in `explore/proposed_tasks.json`
   and follow the `merge_plan.md` recipe (cherry-pick → tests →
   `tasks.json` append → end-to-end validation). Otherwise mark the JSON
   `"status": "archived"` with a `result` block citing the regression.
