"""Freeze the assembled tree into an immutable snapshot.

"Freeze" means three things happen atomically from the caller's POV:
    1. Pick the next version string (``v0``, ``v1``, …) by scanning
       ``out_root`` for existing versions.
    2. Attach a `BuildTrace` to the tree recording exactly which providers,
       thresholds, and fanout were used.
    3. Write every artifact (``tree.json``, ``embeddings.json``,
       ``build_trace.json``) under ``<out_root>/<version>/``.

Snapshots are never overwritten — `next_version` always returns an
unused slot. Consumers read via `storage.load_snapshot(path)`.
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path

from ..schema import Tree, BuildTrace
from ..storage import write_immutable_snapshot, next_version


def freeze(
    tree: Tree,
    config,
    out_root: str | Path,
) -> Tree:
    """Write ``tree`` to disk under a new version directory.

    Args:
        tree: The assembled tree. Its ``version`` field is mutated in place
            to the chosen version string.
        config: The `Config` used for this build. Only its provider IDs
            and knob values are read; nothing is mutated.
        out_root: Parent directory under which the versioned subdirectory
            is created.

    Returns:
        The same ``tree`` instance, with ``version`` and ``build_trace``
        populated. Returned (rather than None) so callers can chain.
    """
    version = next_version(out_root)
    tree.version = version

    tree.build_trace = BuildTrace(
        enricher_llm=getattr(config.enricher_llm, "id", ""),
        labeler_llm=getattr(config.labeler_llm, "id", ""),
        embedding_model=getattr(config.embedder, "id", ""),
        thresholds=dict(config.thresholds),
        fanout={k: list(v) for k, v in config.fanout.items()},
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )

    out_dir = Path(out_root) / version
    write_immutable_snapshot(tree, out_dir)
    return tree
