"""Stage 3+4 — cluster leaves into groups, then upward to categories and domains.

Reads 01_descriptors.json + 02_enrichments.json, writes tree_draft.json.

Run with:
    uv run scripts/stage_cluster.py --run raw-tools
"""
from __future__ import annotations

import argparse
import json

from _pipeline_config import make_config, run_dir
from tool_index.pipeline.orchestrator import assemble_tree
from tool_index.pipeline.stage3_cluster_leaves import cluster_tools_into_groups
from tool_index.pipeline.stage4_cluster_upward import cluster_upward
from tool_index.schema import (
    LEVEL_CATEGORY,
    LEVEL_DOMAIN,
    Enrichment,
    ToolDescriptor,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()

    cfg = make_config()
    out = run_dir(args.run)

    descriptors = [ToolDescriptor.from_dict(d) for d in json.loads((out / "01_descriptors.json").read_text())]
    enrichments = {tid: Enrichment.from_dict(d) for tid, d in json.loads((out / "02_enrichments.json").read_text()).items()}
    print(f"Input: {len(descriptors)} descriptors, {len(enrichments)} enrichments")

    groups = cluster_tools_into_groups(
        descriptors, enrichments, cfg.embedder, cfg.labeler_llm,
        fanout_tool=cfg.fanout["tool"],
        distance_threshold=cfg.thresholds.get("group", 0.3),
    )
    print(f"Stage 3: {len(groups)} groups")

    categories = cluster_upward(
        groups, LEVEL_CATEGORY, cfg.fanout["category"],
        cfg.thresholds["category"], cfg.embedder, cfg.labeler_llm,
    )
    print(f"Stage 4a: {len(categories)} categories")

    min_dom, _ = cfg.fanout["domain"]
    if len(categories) <= min_dom:
        for c in categories:
            c.level = LEVEL_DOMAIN
        domains = categories
        categories_separate = False
    else:
        domains = cluster_upward(
            categories, LEVEL_DOMAIN, cfg.fanout["domain"],
            cfg.thresholds["domain"], cfg.embedder, cfg.labeler_llm,
        )
        categories_separate = True
    print(f"Stage 4b: {len(domains)} domains (categories_separate={categories_separate})")

    tree = assemble_tree(descriptors, groups, categories, domains, categories_separate, cfg.embedder)

    payload = tree.to_dict()
    payload["_meta"] = {
        "categories_separate": categories_separate,
        "expected_depth": 5 if categories_separate else 4,
    }
    path = out / "tree_draft.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {path}  (depth={tree.depth()}, nodes={len(tree.nodes_by_id)}, leaves={len(tree.tools_by_id)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
