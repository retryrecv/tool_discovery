# pipeline

The six build stages + orchestrator.

- `orchestrator.py` — `build_tree_index(config, tools)`: runs stages 1–6 end-to-end, emits `build_trace.json`.
- `stage1_normalize.py` — raw tools → `ToolDescriptor` list; dedupe by stable ID.
- `stage2_enrich.py` — LLM-generated intent/synonyms/examples per tool; cached via `providers/cache`.
- `stage3_cluster_leaves.py` — embed enriched descriptions, cluster into L3 groups.
- `stage4_cluster_upward.py` — recursively cluster group descriptions into L2, then L1.
- `stage5_validate.py` — runs all validators; raises `ValidationError` on fatal failures.
- `stage6_freeze.py` — writes immutable snapshot via `storage.snapshot`.

## Conventions

- Every stage takes plain data in + plain data out (no hidden globals).
- Stages are individually testable — tests live in `tests/unit/test_stageN_*.py`.
- Add a new stage only if it can't be expressed as a step inside an existing one.
