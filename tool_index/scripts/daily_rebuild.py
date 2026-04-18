"""Daily rebuild for a single customer.

    uv run scripts/daily_rebuild.py --customer acme

Reads the project's `_pipeline_config.make_config()` and uses the
shared `raw_tools` list. For multi-customer deployments, swap this for
a per-customer catalog loader.

Exit codes:
    0 = promoted, 2 = rebuilt but rejected, 3 = skipped, 1 = error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _pipeline_config import make_config, raw_tools  # type: ignore  # noqa: E402

from tool_index.rebuild import rebuild_customer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--customer", required=True)
    ap.add_argument("--snapshots", default="data/snapshots")
    ap.add_argument("--epsilon", type=float, default=0.02)
    ap.add_argument("--feedback-delta", type=int, default=50)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = make_config()
    out = rebuild_customer(
        args.customer,
        raw_tools,
        cfg,
        snapshots_root=args.snapshots,
        feedback_delta_threshold=args.feedback_delta,
        epsilon=args.epsilon,
        force=args.force,
    )
    print(json.dumps(out.to_dict(), indent=2))

    if out.skipped:
        return 3
    if out.promotion and out.promotion.promoted:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
