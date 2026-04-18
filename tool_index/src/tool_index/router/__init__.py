"""Router service — Phase 1 of the production tool index.

Serves `POST /route` per customer, hot-loads the active snapshot from
`data/snapshots/<customer_id>/`, and gates promotion of new snapshots
behind a sample-based quality score.

Modules:
    layout        — per-customer snapshot paths + active-version pointer
    quality       — recall@k against curated golden samples
    promotion     — auto-rollback gate
    telemetry     — append-only request logger
    registry      — in-process cache of loaded snapshots, hot-swap aware
    service       — FastAPI app
"""
from .layout import CustomerLayout
from .quality import compute_quality_score, QualityScore
from .promotion import promote_if_better, PromotionResult
from .telemetry import RequestLogger, RouteRecord
from .registry import SnapshotRegistry, ActiveSnapshot

__all__ = [
    "CustomerLayout",
    "compute_quality_score", "QualityScore",
    "promote_if_better", "PromotionResult",
    "RequestLogger", "RouteRecord",
    "SnapshotRegistry", "ActiveSnapshot",
]
