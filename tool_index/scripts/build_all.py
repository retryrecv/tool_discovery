"""Run all 6 stage scripts in order.

Run with:
    uv run scripts/build_all.py --run raw-tools
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

STAGES = [
    "stage_normalize.py",
    "stage_enrich.py",
    "stage_cluster.py",
    "stage_synth_queries.py",
    "stage_validate.py",
    "stage_freeze.py",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()

    here = Path(__file__).parent
    for s in STAGES:
        cmd = ["uv", "run", str(here / s), "--run", args.run]
        print(f"\n=== {s} ===")
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"FAILED at {s}", file=sys.stderr)
            return r.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
