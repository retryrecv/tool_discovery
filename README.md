# Tool Discovery System

Tool Discovery System is an experimental workspace for building and evaluating a hierarchical tool index. The core idea is to route natural-language tool requests through a compact tree, `L1 -> L2 -> L3 -> tool`, instead of asking an LLM to choose from a flat catalog.

The active implementation lives in [`tool_index/`](tool_index/). The root workspace also contains [`planner/`](planner/), but that package is currently separate and is not used by the Tool Index retrieval pipeline.

## What This Project Tests

Large tool catalogs create two practical problems:

- putting every tool in context is expensive and brittle
- multi-intent user requests often require several tools, not one

The Tool Index pipeline builds a versioned tree over a tool catalog, then evaluates whether decomposed user requests can retrieve the correct tool set.

Current strongest measured path:

```bash
cd tool_index
uv run scripts/eval_real_cases.py --run raw-tools --k 10 --decompose --pattern-inject
```

On the current 150-case evaluation corpus, this run reached:

```text
simple recall@10:        50/50 = 1.000
complex set-recall@10:  120/120 = 1.000
ultra set-recall@10:    209/209 = 1.000
```

The comparison run without regex/pattern injection reached `339/379` gold-tool hits. With pattern injection, the traced run reached `379/379` gold-tool hits.

## Repository Layout

```text
tool_index/                 Core Python package and pipeline
  src/tool_index/           Package source
  scripts/                  Stage runners and evaluation scripts
  data/corpus/              Raw tool catalog and eval queries
  tests/                    Unit and integration tests

experiments/                Measured spikes, traces, and result notes
  01_glue_tool_ceiling/     Oracle/glue-tool ceiling experiment
  02_glue_prompt_hints/     Prompt-hint decomposition spike
  03_pattern_injector/      Deterministic pattern injection spike
  04_eval_traces/           JSONL traces for failure-level analysis

papers/plan/                Paper notes used to guide decomposition work
planner/                    Separate workspace member; not used by tool_index
```

## Pipeline

The package-level build pipeline is documented in [`tool_index/README.md`](tool_index/README.md). At a high level it:

1. normalizes raw tool definitions into descriptors
2. enriches tools with intent phrases, IO kinds, synonyms, and examples
3. clusters tools into L3 groups
4. clusters L3 groups upward into L2 and L1 nodes
5. freezes immutable snapshots
6. evaluates routing quality on natural-language cases

## Retrieval Flow

The current evaluation flow for hard queries is:

1. decompose the user query with an LLM
2. optionally append deterministic glue-tool subqueries with `--pattern-inject`
3. embed each subquery
4. route each subquery through the tree
5. rerank candidate tools
6. union the best per-subquery results

Pattern injection is append-only. It preserves LLM-generated subqueries and adds explicit intents for recurring implicit operations, such as JSON field extraction, URL fetching, hashing, UUID generation, current time lookup, and arithmetic.

## Traces

The evaluator can write one JSONL trace per case:

```bash
cd tool_index
uv run scripts/eval_real_cases.py \
  --run raw-tools \
  --k 10 \
  --decompose \
  --pattern-inject \
  --trace-out ../experiments/04_eval_traces/raw_tools_decompose_pattern_trace.jsonl
```

Each trace records:

- original query
- gold tools and predicted tools
- LLM subqueries
- injected rules and final subqueries
- L1/L2/L3 route path
- candidate and reranked tools
- whether each gold tool was a `hit`, `route_miss`, `rerank_miss`, or `union_miss`

This is the main diagnostic artifact for understanding which level failed.

## Development

Install dependencies from the repo root:

```bash
uv sync
```

Run all tests:

```bash
cd tool_index
uv run pytest
```

Run the main evaluation:

```bash
cd tool_index
uv run scripts/eval_real_cases.py --run raw-tools --k 10 --decompose --pattern-inject
```

Run the comparison without pattern injection:

```bash
cd tool_index
uv run scripts/eval_real_cases.py --run raw-tools --k 10 --decompose
```

## Notes

- `blog.md` is intentionally ignored and kept as a local drafting file.
- `planner/` is part of the uv workspace, but it is not called by `tool_index` today.
- The result numbers above are tied to the current `raw-tools` snapshot and evaluation corpus.
