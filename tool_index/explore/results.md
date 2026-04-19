# Recall-improvement scoreboard

Single source of truth for comparing the three exploration spikes against the v7 baseline. Update this file as each branch produces numbers; do not edit other branches' rows.

**Eval set**: `data/snapshots/raw-tools/v7/04_synth_queries.jsonl` (180 queries). Frozen — do not regenerate.
**Metric**: recall@10 from `uv run scripts/stage_validate.py --run raw-tools`.

## Results

| Variant | Branch | Recall@10 | Δ vs baseline | Errors | Low-disc pairs | Avg nodes visited | Notes |
|---|---|---|---|---|---|---|---|
| **Baseline (v7, beam=3)** | `phase1-router` | 0.917 | — | 3 | 28 | TBD | reference snapshot |
| Direction 1 — ColBERT rerank | `explore/colbert-rerank` | TBD | TBD | TBD | TBD | TBD | rerank top-2k → k |
| Direction 2 — Doc2Query | `explore/doc2query` | TBD | TBD | TBD | TBD | TBD | N=5 expansions per tool |
| Direction 3 — Multi-vector nodes | `explore/multivector-nodes` | **0.978** | **+0.061** | 4 | 20 | n/a | child-embedding MaxSim, no rebuild needed |

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
