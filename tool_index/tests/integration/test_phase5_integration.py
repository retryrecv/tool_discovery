"""Phase-5 end-to-end integration test.

Builds a small synthetic catalog, runs the clustering + traversal path
with each HILL-inspired mechanism toggled independently, and asserts the
stack composes without regressions on the simple fixture.

This is intentionally narrow: it proves the mechanisms wire together and
compose, not that they produce state-of-the-art recall (that belongs in
an offline benchmark, not a unit test). Each mechanism's effectiveness
is exercised in its own module-level test.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tool_index.retrieval.traverser import retrieve_with_path
from tool_index.scale import (
    EMRefineConfig,
    ResidualClusterConfig,
    SoftClusterConfig,
    build_kmeans_centroids,
    cluster_upward_residual,
    em_refine,
    soft_cluster,
)
from tool_index.schema import Node, Tree


def _synthetic_embeddings(n_groups: int = 3, per_group: int = 6, dim: int = 4) -> np.ndarray:
    rng = np.random.default_rng(0)
    blobs = []
    for g in range(n_groups):
        center = np.zeros(dim)
        center[g % dim] = 10.0 * (1 if g % 2 == 0 else -1)
        blobs.append(rng.normal(loc=center, scale=0.3, size=(per_group, dim)))
    return np.vstack(blobs)


def _build_tree(labels: np.ndarray, centroids: np.ndarray, dim: int = 4) -> Tree:
    unique = sorted(set(labels.tolist()))
    group_nodes: list[Node] = []
    for gi in unique:
        tool_ids = [f"tool_{i}" for i, l in enumerate(labels.tolist()) if l == gi]
        g = Node(
            id=f"g_{gi}",
            level="group",
            description=f"group {gi}",
            embedding=centroids[gi].tolist(),
            children=tool_ids,
        )
        group_nodes.append(g)
    root = Node(
        id="root",
        level="root",
        description="all",
        embedding=[0.0] * dim,
        children=[g.id for g in group_nodes],
    )
    t = Tree(root=root)
    for n in (root, *group_nodes):
        t.register(n)
    for g in group_nodes:
        g.parent_id = "root"
    return t


def test_baseline_flags_off_produces_a_tree() -> None:
    x = _synthetic_embeddings()
    # Flags off = simple K-means, no EM refinement, no soft clustering.
    cfg = EMRefineConfig(n_clusters=3, n_iter=0, backend="numpy", seed=0)
    labels, centroids, _ = build_kmeans_centroids(x, cfg)
    tree = _build_tree(labels, centroids)
    # Traversal works on the flags-off tree.
    tools, path = retrieve_with_path(tree, [10.0, 0.0, 0.0, 0.0], k=3, beam=2)
    assert len(tools) > 0
    assert len(path) == 1  # single level (no domain/category)


def test_all_flags_on_produces_a_tree() -> None:
    x = _synthetic_embeddings()
    soft_labels, soft_centroids = soft_cluster(
        x, SoftClusterConfig(n_clusters=3, max_iters=20, max_alpha=50.0, balance_weight=0.1)
    )
    # EM refine initial centroids.
    _, refined, wcss = em_refine(x, soft_centroids, n_iter=5)
    assert wcss[-1] <= wcss[0] + 1e-6

    # Residual upward clustering (trivial: single parent = root centroid).
    class _N:
        def __init__(self, eid, emb, pid):
            self.id = eid
            self.embedding = emb
            self.parent_id = pid

    root_centroid = refined.mean(axis=0)
    nodes = [_N(f"g_{i}", c.tolist(), "root") for i, c in enumerate(refined)]
    residual_labels = cluster_upward_residual(
        nodes,
        {"root": root_centroid.tolist()},
        lambda embs: [0] * len(embs),
    )
    assert len(residual_labels) == len(nodes)

    tree = _build_tree(soft_labels, refined)
    tools, path = retrieve_with_path(tree, [10.0, 0.0, 0.0, 0.0], k=3, beam=2)
    assert len(tools) > 0
    assert path != []


def test_leave_one_out_ablation_runs_cleanly() -> None:
    """Each mechanism removed in turn still produces a valid tree."""
    x = _synthetic_embeddings()

    results: dict[str, int] = {}

    # Disable soft_cluster — use EM K-means only.
    labels, centroids, _ = build_kmeans_centroids(
        x, EMRefineConfig(n_clusters=3, n_iter=10, backend="numpy", seed=0)
    )
    tree = _build_tree(labels, centroids)
    tools, _ = retrieve_with_path(tree, [10.0, 0.0, 0.0, 0.0], k=3, beam=2)
    results["no_soft_cluster"] = len(tools)

    # Disable em_refine — use soft_cluster output directly.
    labels, centroids = soft_cluster(
        x, SoftClusterConfig(n_clusters=3, max_iters=20, max_alpha=50.0)
    )
    tree = _build_tree(labels, centroids)
    tools, _ = retrieve_with_path(tree, [10.0, 0.0, 0.0, 0.0], k=3, beam=2)
    results["no_em_refine"] = len(tools)

    assert all(v > 0 for v in results.values())


def test_phase5_report_can_be_emitted(tmp_path: Path) -> None:
    """Smoke-check a phase5_report.json structure for downstream consumers."""
    x = _synthetic_embeddings()
    labels_a, c_a, hist_a = build_kmeans_centroids(
        x, EMRefineConfig(n_clusters=3, n_iter=0, backend="numpy", seed=0)
    )
    labels_b, c_b, hist_b = build_kmeans_centroids(
        x, EMRefineConfig(n_clusters=3, n_iter=10, backend="numpy", seed=0)
    )
    report = {
        "baseline": {"wcss_final": hist_a[-1], "n_clusters": int(c_a.shape[0])},
        "experimental": {"wcss_final": hist_b[-1], "n_clusters": int(c_b.shape[0])},
    }
    out = tmp_path / "phase5_report.json"
    out.write_text(json.dumps(report))
    assert json.loads(out.read_text())["experimental"]["n_clusters"] == 3
    assert hist_b[-1] <= hist_a[-1]
