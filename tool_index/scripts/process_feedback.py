"""Convert a customer's daily request log into labeled feedback.

    uv run scripts/process_feedback.py --customer acme --date 2026-04-17
    uv run scripts/process_feedback.py --customer acme --date yesterday

Writes data/snapshots/<customer>/feedback/<date>.jsonl. Idempotent —
re-runs overwrite the day's file by default.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone

from tool_index.feedback import process_day


def _resolve_date(s: str) -> str:
    if s == "today":
        return datetime.now(timezone.utc).date().isoformat()
    if s == "yesterday":
        return (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--customer", required=True)
    ap.add_argument("--date", required=True, help="YYYY-MM-DD or 'today'/'yesterday'")
    ap.add_argument("--snapshots", default="data/snapshots")
    ap.add_argument("--gap-seconds", type=float, default=300.0)
    ap.add_argument("--append", action="store_true", help="append instead of overwriting")
    args = ap.parse_args()

    summary = process_day(
        args.snapshots,
        args.customer,
        _resolve_date(args.date),
        gap_seconds=args.gap_seconds,
        overwrite=not args.append,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
