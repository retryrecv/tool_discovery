"""Phase 3 — daily rebuild orchestration with auto-tuning + diff report.

Modules:
    catalog_hash  — stable content hash of raw tool list (skip rebuild if unchanged)
    eval_adapter  — turn feedback/*.jsonl into a held-out eval set (positive + mined negative)
    autotune      — small grid sweep over thresholds + fanout, pick best score
    diff          — node-level diff between two snapshots
    rebuild       — orchestrate: hash check → build candidates → score → promote → diff

These read from `data/snapshots/<customer>/` and write back into the
same per-customer directory. Phase 1's promotion gate is the final
checkpoint: a winner from autotune that still falls below the active
score's tolerance is rolled back.
"""
from .catalog_hash import compute_catalog_hash, read_catalog_hash, write_catalog_hash
from .eval_adapter import build_eval_set, EvalQuery
from .autotune import sweep_configs, TuneResult, ConfigPoint
from .diff import diff_snapshots, SnapshotDiff
from .rebuild import rebuild_customer, RebuildOutcome

__all__ = [
    "compute_catalog_hash", "read_catalog_hash", "write_catalog_hash",
    "build_eval_set", "EvalQuery",
    "sweep_configs", "TuneResult", "ConfigPoint",
    "diff_snapshots", "SnapshotDiff",
    "rebuild_customer", "RebuildOutcome",
]
