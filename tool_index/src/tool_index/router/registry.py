"""In-process snapshot registry — load once, serve many.

Each customer's active snapshot is loaded lazily on first request and
cached by `(customer_id, version)`. A pointer-watcher checks the
`active.json` mtime per request (cheap stat call); if it changed,
the new version is loaded in the background and atomically swapped in.

Old snapshot stays referenced until the swap completes, so in-flight
routes never see a torn read.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from ..schema import Tree
from ..storage import load_snapshot
from .layout import CustomerLayout


@dataclass(frozen=True)
class ActiveSnapshot:
    customer_id: str
    version: str
    tree: Tree


class SnapshotRegistry:
    def __init__(self, snapshots_root: str | Path):
        self.snapshots_root = Path(snapshots_root)
        self._cache: dict[str, ActiveSnapshot] = {}
        self._mtime: dict[str, float] = {}
        self._lock = threading.RLock()

    def get(self, customer_id: str) -> ActiveSnapshot:
        layout = CustomerLayout.for_customer(self.snapshots_root, customer_id)
        ptr = layout.active_pointer()
        if not ptr.exists():
            raise KeyError(f"no active snapshot for customer {customer_id!r}")
        mtime = ptr.stat().st_mtime

        with self._lock:
            cached = self._cache.get(customer_id)
            if cached is not None and self._mtime.get(customer_id) == mtime:
                return cached

        version = layout.read_active()
        if version is None:
            raise KeyError(f"active.json for {customer_id!r} is empty")
        tree = load_snapshot(layout.version_dir(version))
        snap = ActiveSnapshot(customer_id=customer_id, version=version, tree=tree)

        with self._lock:
            self._cache[customer_id] = snap
            self._mtime[customer_id] = mtime
        return snap

    def evict(self, customer_id: str) -> None:
        with self._lock:
            self._cache.pop(customer_id, None)
            self._mtime.pop(customer_id, None)
