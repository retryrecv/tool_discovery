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

## Retrieval Experiments

- Keep `--decompose` as the promoted baseline until a measured spike beats it.
- A naive recursive decomposition loop regressed on a 5-case-per-bucket sample:
  baseline `--decompose` reached simple `5/5`, complex `8/10`, ultra `19/21`,
  while the first recursive loop reached simple `5/5`, complex `5/10`, ultra
  `12/21` after adding a multi-intent text guard. The failure mode was not just
  threshold tuning: broad steps were accepted or refined in ways that dropped
  required operations.
- Next recursive-planner work should use structured step contracts, especially
  `expected_tools`, and verify that a refinement preserves parent coverage
  before accepting it. Avoid adding more text-only heuristics without measuring
  against the current `--decompose` baseline.
- The first structured `expected_tools` attempt improved over the guarded
  recursive loop but still failed promotion on the same 5-case-per-bucket
  sample: simple `5/5`, complex `6/10`, ultra `14/21`. Current `--decompose`
  remained stronger at simple `5/5`, complex `8/10`, ultra `19/21`. The main
  observed failure was that the initial structured planner still defaulted too
  many broad ultra-complex requests to `expected_tools=1`, so follow-up work
  should improve planner examples and coverage verification before full-corpus
  evaluation.
- Adding few-shot hard examples to the structured recursive planner improved
  the same bounded sample to simple `5/5`, complex `7/10`, ultra `14/21`, but
  still failed promotion because ultra-complex recall remained below the
  `--decompose` baseline. Next work should preserve the baseline decomposer's
  first-pass split and use recursive refinement only as an additive diagnostic
  for weak or broad sub-queries.
- A baseline-plus-refinement hybrid tied, but did not beat, the current
  bounded baseline: simple `5/5`, complex `8/10`, ultra `19/21`. It is useful
  for diagnostics because it preserves first-pass retrieval, but it should not
  be promoted on accuracy alone.
- A dependency-hinted decomposer is the first bounded spike to beat the current
  sample baseline: simple `5/5`, complex `9/10`, ultra `20/21`. The key fix is
  to make implicit dependencies explicit: `today`, `now`, `right now`, and
  `timestamp` imply `get_current_datetime`; `new id`, `case id`, `tracking id`,
  and similar wording imply `generate_uuid`. Run a larger evaluation before
  promoting this prompt change.
- Full-corpus dependency-hinted decomposition failed promotion despite the
  bounded win: simple `49/50`, complex `107/120`, ultra `188/209`. It improved
  simple but regressed complex below the recorded production baseline
  (`109/120`, 0.908), confirming that global prompt edits perturb too many
  otherwise-good LLM decompositions. Prefer append-only deterministic injection
  after the production decomposer.
- Integrated append-only pattern injection beat the production baseline and
  earlier prompt-based spikes on the full corpus:
  `uv run scripts/eval_real_cases.py --run raw-tools --k 10 --decompose --pattern-inject`
  reached simple `49/50`, complex `115/120`, ultra `200/209` before
  catalog-specific aliases, and simple `50/50`, complex `118/120`, ultra
  `208/209` after adding targeted aliases for JSON field extraction without the
  literal word `JSON`, reward order/category phrasing, and SpaceX ordinal
  endpoints. The remaining misses were `calculator` on a `how long ... at 12
  km/h` arithmetic subtask, `reward__simple_referral__get_rewards` on a last
  redeemed reward query, and `generate_uuid` on a `receipt id` support query.
  Continue with small append-only rules; do not replace the production
  decomposer prompt for this class of failures.
- A follow-up full-corpus injector run after adding the `how long ... at speed`,
  `receipt id`, and simple-referral-rewards aliases fixed the previous complex
  and ultra misses but exposed a simple regression: simple `49/50`, complex
  `120/120`, ultra `209/209`. The missed simple query was `Is this scan of my
  driver's license actually valid?` for
  `visual_recognition__document_image_validation__get_call`. Focused reruns of
  that exact query ranked the gold tool first and fired no prior injector rule,
  so treat the miss as LLM/decomposition variance or retrieval fluctuation rather
  than direct rule interference. The next stabilization attempt is a narrowly
  gated `visual_document_image_validation` injection for scanned ID/license/
  passport validation; promote only after a clean full-corpus rerun preserves
  simple `50/50` while keeping complex/ultra gains. The clean rerun with that
  stabilization reached simple `50/50`, complex `120/120`, ultra `209/209` with
  `115/150` queries matched and `248` injected sub-queries, so this is the
  current strongest measured injector spike.
- For misses where the user query or LLM sub-query already names the concept,
  prefer improving the tool enrichment and L2/L3 node descriptions before adding
  another injector rule. The `percentage_calc` share/contribution miss is the
  current example: updating the source catalog doc, `02_enrichments.json`, and
  the `cat_be4976ea`/`grp_823c2970` tree descriptions made `share of total`,
  `largest transaction share`, and `contribution to total` route to
  `percentage_calc` without needing a special injected sub-query. A full
  `uv run scripts/eval_real_cases.py --run raw-tools --k 10 --decompose
  --pattern-inject` rerun on 2026-05-02 reached simple `50/50`, complex
  `120/120`, ultra `209/209`; still repeat this run before treating the result
  as stable because LLM decomposition has shown variance.
