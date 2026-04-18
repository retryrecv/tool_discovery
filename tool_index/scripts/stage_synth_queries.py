"""Stage 5a — generate synthetic eval queries from enrichments.

Reads 02_enrichments.json, writes 04_synth_queries.jsonl.

Run with:
    uv run scripts/stage_synth_queries.py --run raw-tools
"""
from __future__ import annotations

import argparse
import json

from _pipeline_config import make_config, run_dir
from tool_index.schema.enrichment import Enrichment
from tool_index.validation import generate_synthetic_queries


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()

    cfg = make_config()
    out = run_dir(args.run)

    enrich_path = out / "02_enrichments.json"
    raw = json.loads(enrich_path.read_text())
    enrichments = {tid: Enrichment.from_dict(d) for tid, d in raw.items()}
    print(f"Input: {len(enrichments)} enrichments from {enrich_path}")

    queries = generate_synthetic_queries(enrichments, cfg.labeler_llm, cfg.synthetic_queries_per_tool)
    print(f"Output: {len(queries)} synthetic queries ({cfg.synthetic_queries_per_tool} per tool)")

    path = out / "04_synth_queries.jsonl"
    with path.open("w") as f:
        for row in queries:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {path}")

    print("\nFirst 6 samples:")
    for row in queries[:6]:
        print(f"  {row['tool_id']}  →  {row['query']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
