"""Stage 5 — validate the assembled tree against pre-built synthetic queries.

Reads tree_draft.json + 02_enrichments.json + 04_synth_queries.jsonl,
writes 05_validation.json.

Run with:
    uv run scripts/stage_validate.py --run raw-tools
"""
from __future__ import annotations

import argparse
import json

from _pipeline_config import make_config, run_dir
from tool_index.pipeline.stage5_validate import validate_tree
from tool_index.schema import Enrichment, Tree


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()

    cfg = make_config()
    out = run_dir(args.run)

    tree_payload = json.loads((out / "tree_draft.json").read_text())
    meta = tree_payload.pop("_meta", {})
    expected_depth = meta.get("expected_depth", 5)
    tree = Tree.from_dict(tree_payload)

    enrichments = {tid: Enrichment.from_dict(d) for tid, d in json.loads((out / "02_enrichments.json").read_text()).items()}

    queries = []
    with (out / "04_synth_queries.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))

    print(f"Input: tree depth={tree.depth()}, {len(enrichments)} enrichments, {len(queries)} synthetic queries")

    report = validate_tree(
        tree, enrichments, cfg.embedder, cfg.judge_llm,
        labeler_llm=cfg.labeler_llm,
        fanout=cfg.fanout,
        expected_depth=expected_depth,
        discriminability_threshold=cfg.thresholds["discriminability"],
        synthetic_per_tool=cfg.synthetic_queries_per_tool,
        recall_k=cfg.recall_k,
        min_recall=cfg.thresholds["min_recall"],
        queries=queries,
    )

    payload = report.summary()
    payload["seed_eval_set"] = report.seed_eval_set
    path = out / "05_validation.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {path}")
    print(f"  passed={report.passed}  recall@{cfg.recall_k}={report.recall_at_k:.3f}")
    print(f"  errors={len(report.errors)}  warnings={len(report.warnings)}")
    for e in report.errors[:5]:
        print(f"    ERROR: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
