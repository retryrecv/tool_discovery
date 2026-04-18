"""Stage 6 — freeze tree_draft.json + 05_validation.json into v<N>/.

Run with:
    uv run scripts/stage_freeze.py --run raw-tools
"""
from __future__ import annotations

import argparse
import json

from _pipeline_config import make_config, run_dir
from tool_index.pipeline.stage6_freeze import freeze
from tool_index.schema import Tree, ValidationReport


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()

    cfg = make_config()
    out = run_dir(args.run)

    tree_payload = json.loads((out / "tree_draft.json").read_text())
    tree_payload.pop("_meta", None)
    tree = Tree.from_dict(tree_payload)

    val = json.loads((out / "05_validation.json").read_text())
    report = ValidationReport(
        passed=val["passed"],
        errors=list(val.get("errors", [])),
        warnings=list(val.get("warnings", [])),
        recall_at_k=val.get("recall_at_k", 0.0),
        seed_eval_set=list(val.get("seed_eval_set", [])),
        details=val.get("details", {}),
    )

    frozen = freeze(tree, report, cfg, out)
    print(f"Frozen as {frozen.version} under {out}/{frozen.version}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
