"""Score a frozen snapshot against the customer's golden samples and
gate promotion behind the auto-rollback rule.

Run AFTER `stage_freeze.py`:

    uv run scripts/promote_snapshot.py \
        --customer acme \
        --version v3 \
        --snapshots data/snapshots

Reads:
    data/snapshots/<customer>/<version>/tree.json
    data/snapshots/<customer>/samples.jsonl
Writes:
    data/snapshots/<customer>/quality/<version>.json
    data/snapshots/<customer>/active.json   (only if gate passes)

Exit code 0 = promoted, 2 = rejected (rolled back), 1 = error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tool_index.providers import make_embedding
from tool_index.router import (
    CustomerLayout,
    SnapshotRegistry,
    compute_quality_score,
    promote_if_better,
)
from tool_index.storage import load_snapshot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--customer", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--snapshots", default="data/snapshots")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--beam", type=int, default=2)
    ap.add_argument("--epsilon", type=float, default=0.02)
    ap.add_argument("--embedder", default="azure_openai")
    ap.add_argument("--dim", type=int, default=3072)
    args = ap.parse_args()

    layout = CustomerLayout.for_customer(args.snapshots, args.customer)
    layout.ensure()

    version_dir = layout.version_dir(args.version)
    if not version_dir.exists():
        print(f"error: {version_dir} not found", file=sys.stderr)
        return 1

    tree = load_snapshot(version_dir)
    embedder = make_embedding(args.embedder, dim=args.dim)
    quality = compute_quality_score(
        tree, layout.samples_path(), embedder, k=args.k, beam=args.beam
    )

    result = promote_if_better(layout, args.version, quality, epsilon=args.epsilon)
    print(json.dumps({
        "quality": quality.to_dict(),
        "promotion": result.to_dict(),
    }, indent=2))

    return 0 if result.promoted else 2


if __name__ == "__main__":
    raise SystemExit(main())
