# Recall-improvement scoreboard

Single source of truth for comparing the three exploration spikes against the v7 baseline. Update this file as each branch produces numbers; do not edit other branches' rows.

**Eval set**: `data/snapshots/raw-tools/v7/04_synth_queries.jsonl` (180 queries). Frozen — do not regenerate.
**Metric**: recall@10 from `uv run scripts/stage_validate.py --run raw-tools`.

## Results

| Variant | Branch | Recall@10 | Δ vs baseline | Errors | Low-disc pairs | Avg nodes visited | Notes |
|---|---|---|---|---|---|---|---|
| **Baseline (v7, beam=3)** | `phase1-router` | 0.917 | — | 3 | 28 | TBD | reference snapshot |
| Direction 1 — ColBERT rerank | `explore/colbert-rerank` | **0.856** | **+0.017** vs same-script baseline (0.839) | 3 | 28 | n/a | rerank top-20 → top-10 via MaxSim over per-tool intent+queries; gain real but small |
| **Stack: Direction 3 + Direction 1** | `explore/multivector-plus-rerank` | **0.994** | **+0.077** vs v7 baseline (0.917) | 4 | 20 | n/a | 179/180 queries hit; +0.016 over Direction 3 alone; gains are additive |
| Direction 2 — Doc2Query | `explore/doc2query` | **0.767** | **−0.150** | 2 | 20 | n/a | +5 LLM queries/tool diluted intent_phrase signal; eval queries match the un-augmented distribution |
| Direction 3 — Multi-vector nodes | `explore/multivector-nodes` | **0.978** | **+0.061** | 4 | 20 | n/a | child-embedding MaxSim, no rebuild needed |

## Real-test eval (post 9-tool uncomment, 99-tool catalog, productized D3+D1 stack)

Tracks decomposition spikes (Directions 4-7) against the **real natural-language `test_cases.py` corpus** (50 simple + 50 complex), not the synthetic 180-query set. Eval script: `uv run scripts/eval_real_cases.py --run raw-tools --k 10 [--decompose]`.

| Variant | Branch | Simple recall@10 | Complex set-recall@10 | Complex full-cover | Notes |
|---|---|---|---|---|---|
| Real-test baseline (no decomposition) | `phase1-router` | 49/50 = 0.980 | 96/120 = 0.800 | 31/50 | 99 tools, productized D3+D1 stack, single retrieval per query |
| **Direction 4 — L2M decomposition + union** | `explore/l2m-decomposition` | **50/50 = 1.000** | **113/120 = 0.942** | **43/50** | LLM splits 45/100 queries (avg 1.67 sub-queries); +0.142 complex, +0.020 simple; decision rule met |

## How to fill a row

After running the spike on its branch:
1. Read `data/snapshots/raw-tools/<run>/05_validation.json` → `recall_at_k`, `len(errors)`, count of `low discriminability` warnings.
2. Run the per-category miss diagnostic; capture cat_391687b6 miss-rate in **Notes**.
3. Compute `Avg nodes visited` = mean candidates inspected per query (instrument `retrieve()` once, share the helper).
4. Commit the row update on this file to `main` via PR — do **not** merge the spike branch yet.

## Promotion rule

A spike is promoted to `tasks.json` only if its row meets the `decision_rule` in its own JSON file (`direction*.json`). Losers stay in this table for the postmortem; their JSON gets `"status": "archived"` plus a `"result"` block.

## Stacking

Once any direction lands on `main`, re-baseline this table — promoted gains stack into the new reference row, and remaining spikes re-measure against it.
