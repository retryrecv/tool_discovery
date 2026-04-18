"""Sharded cache — wraps `DiskCache` to split entries across hash-prefix
subdirectories.

Why: a single directory with 10k+ small files is slow on most
filesystems (linear `readdir`, slow ENOENT lookups under `getattr`).
Splitting by the first 2 hex chars of the SHA gives us 256 buckets,
each holding ~40 files at 10k catalog scale.

This wraps the existing DiskCache rather than replacing it: the
underlying file format is unchanged, just the directory layout. A
parallel `ShardedDiskCache` instance pointed at an old cache will
simply miss until entries are re-stored — no migration needed because
LLM cache misses are recoverable.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..utils.hashing import stable_hash


class ShardedDiskCache:
    """Drop-in replacement for `DiskCache.get` / `put` with shard prefix."""

    def __init__(self, root: str | Path, *, shard_chars: int = 2):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if shard_chars < 1 or shard_chars > 4:
            raise ValueError("shard_chars must be in [1, 4]")
        self.shard_chars = shard_chars

    @property
    def id(self) -> str:
        return f"sharded:{self.shard_chars}"

    def _key_path(self, model_id: str, prompt: str) -> Path:
        h = stable_hash(f"{model_id}::{prompt}")
        shard = h[: self.shard_chars]
        return self.root / model_id.replace("/", "_") / shard / f"{h}.json"

    def get(self, model_id: str, prompt: str):
        p = self._key_path(model_id, prompt)
        if p.exists():
            return json.loads(p.read_text())
        return None

    def put(self, model_id: str, prompt: str, value) -> None:
        p = self._key_path(model_id, prompt)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value))
