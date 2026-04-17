"""CLI entrypoint — ``python -m tool_index build …`` lives here.

Subcommands:
    build — stage1→6 from a JSON / JSONL catalog into a snapshot dir.

Kept intentionally thin — it's a wrapper around `build_tree_index` plus
some input-shape handling.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from .config import load_config, default_config
from .pipeline import build_tree_index


def _load_tools(path: Path) -> list[dict]:
    """Load a tool catalog from JSON or JSONL.

    Accepts three shapes for flexibility:
        • ``.jsonl`` — one tool per line.
        • ``.json`` top-level list of tool dicts.
        • ``.json`` ``{"tools": [...]}`` object with the list under ``tools``.

    Anything else raises at the ``assert`` so the user gets a clear error
    instead of a cryptic downstream failure.
    """
    text = path.read_text()
    if path.suffix == ".jsonl":
        # Skip blank lines so JSONL files with trailing newlines parse cleanly.
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, dict) and "tools" in data:
        return data["tools"]
    assert isinstance(data, list), "expected JSON array or {tools: [...]} or JSONL"
    return data


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. Returns a process exit code.

    Called by ``__main__.py`` when invoked as ``python -m tool_index``.
    Also directly callable from tests.
    """
    parser = argparse.ArgumentParser(prog="tool_index")
    sub = parser.add_subparsers(dest="cmd", required=True)

    build = sub.add_parser("build", help="Build the tree index")
    build.add_argument("--config", default=None)
    build.add_argument("--input", required=True)
    build.add_argument("--output", default="data/snapshots")
    # ``--lax`` inverts the orchestrator's ``strict`` default: freeze
    # even if validation failed, useful for debugging tree shape without
    # meeting the recall floor.
    build.add_argument("--lax", action="store_true", help="Do not raise on validation failure")

    args = parser.parse_args(argv)

    if args.cmd == "build":
        cfg = load_config(args.config) if args.config else default_config()
        tools = _load_tools(Path(args.input))
        tree = build_tree_index(tools, cfg, out_root=args.output, strict=not args.lax)
        # Success summary on stderr so stdout stays reserved for
        # machine-readable output if we ever add any.
        print(f"Built {tree.version} with {len(tree.tools_by_id)} tools", file=sys.stderr)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
