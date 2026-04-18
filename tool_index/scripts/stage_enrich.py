"""Stage 2 — LLM-enrich each descriptor.

Reads 01_descriptors.json, writes 02_enrichments.json.

Run with:
    uv run scripts/stage_enrich.py --run raw-tools
"""
from __future__ import annotations

import argparse
import json

from _pipeline_config import make_config, run_dir
from tool_index.pipeline.stage2_enrich import enrich_all
from tool_index.schema.descriptor import ToolDescriptor


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()

    cfg = make_config()
    out = run_dir(args.run)

    descriptors_path = out / "01_descriptors.json"
    descriptors = [ToolDescriptor.from_dict(d) for d in json.loads(descriptors_path.read_text())]
    print(f"Input: {len(descriptors)} descriptors from {descriptors_path}")

    enrichments = enrich_all(
        descriptors,
        cfg.enricher_llm,
        cache=cfg.cache,
        batch_size=cfg.enrich_batch_size,
    )
    print(f"Output: {len(enrichments)} enrichments")

    path = out / "02_enrichments.json"
    payload = {tid: e.to_dict() for tid, e in enrichments.items()}
    path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {path}")

    sample_id = next(iter(enrichments))
    print(f"\nSample ({sample_id}):")
    print(json.dumps(enrichments[sample_id].to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
