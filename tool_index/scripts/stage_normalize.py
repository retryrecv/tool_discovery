"""Stage 1 — normalize raw tools into ToolDescriptors.

Run with:
    uv run scripts/stage_normalize.py --run raw-tools
"""
from __future__ import annotations

import argparse
import json

from _pipeline_config import make_config, raw_tools, run_dir
from tool_index.pipeline.stage1_normalize import normalize_and_dedupe


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()

    cfg = make_config()
    out = run_dir(args.run)

    print(f"Input: {len(raw_tools)} raw tools")
    descriptors = normalize_and_dedupe(raw_tools, cfg.embedder, cfg.thresholds["near_dup"])
    print(f"Output: {len(descriptors)} descriptors")

    path = out / "01_descriptors.json"
    path.write_text(json.dumps([d.to_dict() for d in descriptors], indent=2))
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
