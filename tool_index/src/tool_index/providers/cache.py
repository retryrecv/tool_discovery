"""On-disk cache for LLM / embedding responses.

Keyed by ``(model_id, prompt)`` → SHA-256 path. Survives process restarts
so re-runs on the same inputs skip the LLM entirely. Used by stage 2
(enrichment) and could be used by stages 3, 4, and 5 if their LLM calls
were prompt-stable (currently they aren't, because neighbors vary).
"""
from __future__ import annotations
import json
from pathlib import Path

from ..utils.hashing import stable_hash


class DiskCache:
    """Content-addressed disk cache with one file per entry.

    Layout: ``<root>/<sanitized_model_id>/<sha256>.json``. Missing
    subdirectories are created on first ``put``.
    """

    def __init__(self, root: str | Path):
        """Create the cache root if needed.

        Args:
            root: Base directory. Relative paths resolve against the current
                working directory — typically the pipeline is invoked from
                ``tool_index/`` so paths like ``data/cache`` work out of the box.
        """
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _key_path(self, model_id: str, prompt: str) -> Path:
        """Map ``(model_id, prompt)`` to a deterministic file path.

        The ``"/"`` replacement keeps model IDs like ``"openai/gpt-4"``
        from escaping the cache root via the filesystem.
        """
        h = stable_hash(f"{model_id}::{prompt}")
        return self.root / model_id.replace("/", "_") / f"{h}.json"

    def get(self, model_id: str, prompt: str):
        """Return the cached value, or ``None`` on miss.

        Values are whatever JSON-serializable type was stored — this
        interface is type-agnostic by design.
        """
        p = self._key_path(model_id, prompt)
        if p.exists():
            return json.loads(p.read_text())
        return None

    def put(self, model_id: str, prompt: str, value) -> None:
        """Store ``value`` under the key. Creates parent dirs as needed."""
        p = self._key_path(model_id, prompt)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value))
