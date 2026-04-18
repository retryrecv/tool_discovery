"""Per-customer snapshot directory layout.

Old layout (single tenant):
    data/snapshots/<run-name>/v<N>/

New layout (multi-tenant):
    data/snapshots/<customer_id>/v<N>/        — frozen tree
    data/snapshots/<customer_id>/active.json  — { "version": "v3" }
    data/snapshots/<customer_id>/quality/v<N>.json
    data/snapshots/<customer_id>/samples.jsonl
    data/snapshots/<customer_id>/requests/<YYYY-MM-DD>.jsonl

`active.json` is the only mutable file. Pointing it at a new version is
the atomic promotion step. Snapshot dirs themselves stay immutable.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

_VERSION_RE = re.compile(r"v(\d+)$")


@dataclass(frozen=True)
class CustomerLayout:
    root: Path
    customer_id: str

    @classmethod
    def for_customer(cls, snapshots_root: str | Path, customer_id: str) -> "CustomerLayout":
        return cls(root=Path(snapshots_root) / customer_id, customer_id=customer_id)

    def ensure(self) -> None:
        (self.root / "quality").mkdir(parents=True, exist_ok=True)
        (self.root / "requests").mkdir(parents=True, exist_ok=True)

    def version_dir(self, version: str) -> Path:
        return self.root / version

    def quality_path(self, version: str) -> Path:
        return self.root / "quality" / f"{version}.json"

    def samples_path(self) -> Path:
        return self.root / "samples.jsonl"

    def active_pointer(self) -> Path:
        return self.root / "active.json"

    def requests_path(self, date_str: str) -> Path:
        return self.root / "requests" / f"{date_str}.jsonl"

    def list_versions(self) -> list[str]:
        if not self.root.exists():
            return []
        out = []
        for child in self.root.iterdir():
            m = _VERSION_RE.match(child.name)
            if m and child.is_dir():
                out.append((int(m.group(1)), child.name))
        return [name for _, name in sorted(out)]

    def read_active(self) -> str | None:
        p = self.active_pointer()
        if not p.exists():
            return None
        return json.loads(p.read_text()).get("version")

    def write_active(self, version: str) -> None:
        self.ensure()
        tmp = self.active_pointer().with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"version": version}))
        tmp.replace(self.active_pointer())
