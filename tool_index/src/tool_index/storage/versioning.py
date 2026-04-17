"""Snapshot version slot allocation.

Snapshots live under ``<out_root>/v<N>/`` and are never overwritten. This
module finds the next unused ``N``. Version strings are opaque to the
rest of the code — we don't parse them outside this file.
"""
from __future__ import annotations
from pathlib import Path
import re


def next_version(snapshots_dir: str | Path) -> str:
    """Return the next unused ``v<N>`` string under ``snapshots_dir``.

    Scans existing children matching ``v\\d+`` and returns one past the
    maximum. Missing directory → ``"v0"``. Non-matching children (e.g.
    ``README.md``) are ignored so the snapshots dir can coexist with
    other files.
    """
    p = Path(snapshots_dir)
    if not p.exists():
        return "v0"
    existing = []
    for child in p.iterdir():
        m = re.match(r"v(\d+)$", child.name)
        if m:
            existing.append(int(m.group(1)))
    if not existing:
        return "v0"
    return f"v{max(existing) + 1}"
