"""Phase 4 — scale to 10k tools.

All modules here are additive. Existing pipeline files are not edited;
callers opt into Phase 4 behavior by importing these modules instead of
the originals.

Modules:
    sharded_cache    — DiskCache wrapper that partitions by hash prefix
    async_enrich     — concurrent stage 2 (semaphore-bounded asyncio)
    ann_neighbors    — FAISS/HNSW neighbor graph (with NumPy fallback)
    incremental      — content-hash per tool; pick the changed subset only

Each module ships with a fake/fallback path so tests stay offline and
the optional deps (faiss, anthropic async client) are not hard
requirements.
"""
from .sharded_cache import ShardedDiskCache
from .async_enrich import enrich_all_async, AsyncEnrichConfig
from .ann_neighbors import build_neighbor_graph, NeighborConfig
from .incremental import (
    tool_content_hash,
    diff_descriptors,
    DescriptorDiff,
    plan_incremental_rebuild,
    IncrementalPlan,
)
from .soft_cluster import (
    SoftClusterConfig,
    assign_soft,
    assignment_entropy,
    alpha_schedule,
    soft_cluster,
)
from .residual_cluster import (
    ResidualClusterConfig,
    cluster_upward_residual,
    reconstruction_residuals,
)
from .em_refine import (
    EMRefineConfig,
    build_kmeans_centroids,
    em_refine,
)

__all__ = [
    "ShardedDiskCache",
    "enrich_all_async", "AsyncEnrichConfig",
    "build_neighbor_graph", "NeighborConfig",
    "tool_content_hash", "diff_descriptors", "DescriptorDiff",
    "plan_incremental_rebuild", "IncrementalPlan",
    "SoftClusterConfig", "assign_soft", "assignment_entropy",
    "alpha_schedule", "soft_cluster",
    "ResidualClusterConfig", "cluster_upward_residual",
    "reconstruction_residuals",
    "EMRefineConfig", "build_kmeans_centroids", "em_refine",
]

