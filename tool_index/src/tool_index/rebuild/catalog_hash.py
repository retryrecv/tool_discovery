"""Stable content hash of the raw tool catalog.

The hash is computed over `(name, signature, doc, sorted examples)` of
every tool, sorted by name. Two catalogs that differ only in tool order
hash the same — order is not load-bearing for the pipeline.

We persist `data/snapshots/<customer>/catalog.hash` so the daily
rebuild can short-circuit when nothing changed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


def compute_catalog_hash(raw_tools: Iterable[dict]) -> str:
    canonical = []
    for t in raw_tools:
        canonical.append({
            "name": t["name"],
            "signature": t.get("signature", ""),
            "doc": t.get("doc", ""),
            "examples": sorted(
                (json.dumps(e, sort_keys=True) for e in t.get("examples", []) or []),
            ),
        })
    canonical.sort(key=lambda r: r["name"])
    blob = json.dumps(canonical, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def hash_path(snapshots_root: str | Path, customer_id: str) -> Path:
    return Path(snapshots_root) / customer_id / "catalog.hash"


def read_catalog_hash(snapshots_root: str | Path, customer_id: str) -> str | None:
    p = hash_path(snapshots_root, customer_id)
    if not p.exists():
        return None
    return p.read_text().strip() or None


def write_catalog_hash(snapshots_root: str | Path, customer_id: str, h: str) -> None:
    p = hash_path(snapshots_root, customer_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(h)
