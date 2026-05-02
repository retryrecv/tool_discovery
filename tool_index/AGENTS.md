# tool_index

Hierarchical Tool Index construction pipeline. Build stages:

1. **Normalize** raw tool definitions into `ToolDescriptor` records.
2. **Enrich** each tool with intent phrase, IO kinds, synonyms, and examples.
3. **Cluster leaves**: tools to L3 groups.
4. **Cluster upward**: L3 to L2 to L1 using child descriptions.
5. **Freeze** immutable versioned snapshots.

Tree quality is measured externally with `scripts/eval_real_cases.py`
against the natural-language `eval_queries.py` corpus. There is no
in-pipeline validation step.

## Workspace

This package is a member of the uv workspace at the repo root. The shared
`.venv/` and `uv.lock` live at `../`; do not create a per-member venv here.
Run `uv sync` from the repo root to install dependencies.

## Commands

Run these from `tool_index/` unless noted otherwise:

```bash
uv run scripts/verify_openai_client.py
uv run scripts/stage_normalize.py --run <run-name>
uv run scripts/stage_enrich.py --run <run-name>
uv run scripts/stage_cluster.py --run <run-name>
uv run scripts/stage_freeze.py --run <run-name>
uv run scripts/build_all.py --run <run-name>
uv run scripts/eval_real_cases.py --run <run-name> --k 10 --decompose
pytest tests/ -x
```

From the repo root, the full workspace test command is:

```bash
uv run pytest
```

`<run-name>` is the snapshot subdirectory under `data/snapshots/`, such as
`raw-tools`. Stage scripts read prior outputs from that run directory and
write their own stage outputs there.

## Layout

- `src/tool_index/` — package source.
- `tests/` — unit and integration tests.
- `configs/` — YAML configs.
- `data/corpus/` — `catalog.py` raw tools and `eval_queries.py` fixtures.
- `data/cache/` — provider cache for LLM and embedding calls.
- `data/snapshots/<run-name>/` — stage outputs and frozen versions.
- `scripts/` — pipeline stage scripts and shared pipeline config.
- `explore/` — measured recall-improvement spike records.

## Conventions

- Keep `tool_index/` self-contained: package code, tests, data, scripts, and
  task tracking belong under this directory.
- Put package code under `src/tool_index/`; put runnable one-off or pipeline
  helpers under `scripts/`.
- Orchestrator entrypoint is `tool_index.pipeline.orchestrator.build_tree_index`.
  Production runs should go through the stage scripts.
- All LLM and embedding access goes through `providers/`; do not import vendor
  SDKs directly elsewhere.
- LLM responses may be wrapped in markdown code fences; provider and parse sites
  should strip fences before `json.loads`.
- Determinism matters: use seeded RNGs, stable hashing, and stable IDs.
- Snapshots are immutable. Freezing allocates the next `v<N>/` slot instead of
  overwriting existing versions.

## Task Tracking

Use `tasks.json` as the source of truth for planned and shipped package work.
For promoted exploration work, preserve the extended fields already used in
that file, including status, source branch, evaluation record, and alternatives
tried.

Before adding major architectural work to `tasks.json`, record the spike under
`explore/`, measure it, update `explore/results.md`, and promote only if its
decision rule passes.
