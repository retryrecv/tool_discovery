"""End-to-end build orchestrator — runs the build stages and assembles a `Tree`.

`build_tree_index` is the one public entrypoint consumers should use. The
CLI, tests, and the verification script all call it. Individual stages
remain importable for fine-grained testing, but production code should go
through the orchestrator so stage ordering and tree assembly stay in one
place.

Stage flow:
    1. normalize_and_dedupe  — raw dicts → ToolDescriptors
    2. enrich_all            — LLM-generated hints per tool
    3. cluster_tools_into_groups  (leaves → L3)
    4a. cluster_upward       (L3 → L2)
    4b. cluster_upward       (L2 → L1), OR relabel L2 as L1 if too few
    5. freeze                — write immutable snapshot
"""
from __future__ import annotations
from pathlib import Path

from ..schema import RawTool, Tree, Node, LEVEL_ROOT, LEVEL_DOMAIN, LEVEL_CATEGORY, LEVEL_GROUP
from ..config.loader import Config
from ..utils.ids import new_id
from ..utils.logging import get_logger

from .stage1_normalize import normalize_and_dedupe
from .stage2_enrich import enrich_all
from .stage3_cluster_leaves import cluster_tools_into_groups
from .stage4_cluster_upward import cluster_upward
from .stage6_freeze import freeze


log = get_logger(__name__)


def _register_all(tree: Tree, nodes: list[Node], parent_id: str) -> None:
    """Attach ``nodes`` as children of ``parent_id`` and register them.

    Mutates each node's ``parent_id`` in place — safe because clusterers
    produce fresh `Node` instances that nothing else holds references to yet.
    """
    for n in nodes:
        n.parent_id = parent_id
        tree.register(n)


def assemble_tree(
    descriptors,
    groups: list[Node],
    categories: list[Node],
    domains: list[Node],
    categories_separate: bool,
    embedder,
) -> Tree:
    """Wire stage-3/4 cluster output into a fully-registered `Tree`.

    Shared by `build_tree_index` and `scripts/stage_cluster.py` so the
    two paths cannot drift. Creates the synthetic L0 root, registers
    every domain/category/group, and attaches `tools_by_id`.

    Args:
        descriptors: Stage-1 output; becomes ``tree.tools_by_id``.
        groups: L3 nodes from stage 3.
        categories: L2 nodes from stage 4a (may be the same list as
            ``domains`` when stage 4b collapsed).
        domains: L1 nodes from stage 4b.
        categories_separate: True if stage 4b actually clustered (5-level
            tree); False if it relabeled categories as domains (4-level).
        embedder: Used to embed the synthetic root description.
    """
    root = Node(
        id=new_id("root", "all"),
        level=LEVEL_ROOT,
        description="All tools",
        embedding=embedder.embed("all tools"),
        children=[d.id for d in domains],
    )
    tree = Tree(root=root, version="v0-draft")
    tree.register(root)
    _register_all(tree, domains, root.id)
    if categories_separate:
        for dom in domains:
            _register_all(tree, [c for c in categories if c.id in dom.children], dom.id)
        for cat in categories:
            _register_all(tree, [g for g in groups if g.id in cat.children], cat.id)
    else:
        for dom in domains:
            _register_all(tree, [g for g in groups if g.id in dom.children], dom.id)
    tree.tools_by_id = {d.id: d for d in descriptors}
    return tree


def build_tree_index(
    raw_tools: list[RawTool],
    config: Config,
    *,
    out_root: str | Path = "data/snapshots",
) -> Tree:
    """Build and freeze a tool index from raw tool definitions.

    Args:
        raw_tools: Catalog in whatever shape it arrived in. Each dict must
            carry enough fields for stage 1 to produce a `ToolDescriptor`
            (at minimum ``name`` + some description of the tool's behavior).
        config: A fully-built `tool_index.config.Config` — its providers
            (``enricher_llm``, ``labeler_llm``, ``embedder``, ``cache``)
            must already be instantiated.
        out_root: Directory where the freeze stage will write the versioned
            snapshot subdirectory (``<out_root>/<version>/``).

    Returns:
        The frozen `Tree`, with ``version`` updated by the freeze stage to
        the final immutable version string.
    """
    # Stage 1 — normalize + dedupe near-identical tools.
    log.info("Stage 1: normalize & dedupe (%d raw)", len(raw_tools))
    descriptors = normalize_and_dedupe(raw_tools, config.embedder, config.thresholds["near_dup"])
    log.info("Stage 1: %d descriptors kept", len(descriptors))

    # Stage 2 — LLM-generated `Enrichment` per tool, cached on disk by the
    # provider cache so re-runs don't re-bill the LLM.
    log.info("Stage 2: enrich")
    enrichments = enrich_all(
        descriptors, config.enricher_llm, cache=config.cache, batch_size=config.enrich_batch_size
    )

    # Stage 3 — cluster leaves into L3 groups. Labeler LLM writes each
    # group's description from its members.
    log.info("Stage 3: cluster leaves into groups")
    groups = cluster_tools_into_groups(
        descriptors, enrichments,
        config.embedder, config.labeler_llm,
        fanout_tool=config.fanout["tool"],
        distance_threshold=config.thresholds.get("group", 0.3),
    )
    log.info("Stage 3: %d groups", len(groups))

    # Stage 4a — L3 → L2.
    log.info("Stage 4a: cluster groups into categories")
    categories = cluster_upward(
        groups, LEVEL_CATEGORY, config.fanout["category"],
        config.thresholds["category"], config.embedder, config.labeler_llm,
    )
    log.info("Stage 4a: %d categories", len(categories))

    # Stage 4b — L2 → L1. If there are fewer categories than the minimum
    # domain fanout, upward clustering would collapse to a single node. In
    # that case, the design's pseudocode says to just relabel categories as
    # domains, producing a shorter but valid tree.
    min_dom, _ = config.fanout["domain"]
    if len(categories) <= min_dom:
        log.info("Stage 4b: too few categories (%d) — relabeling as domains", len(categories))
        for c in categories:
            c.level = LEVEL_DOMAIN
        domains = categories
    else:
        log.info("Stage 4b: cluster categories into domains")
        domains = cluster_upward(
            categories, LEVEL_DOMAIN, config.fanout["domain"],
            config.thresholds["domain"], config.embedder, config.labeler_llm,
        )
    log.info("Stage 4b: %d domains", len(domains))

    # Assemble the tree — single synthetic root owns every L1 domain.
    # Detect whether stage 4b collapsed L2→L1 (same list object) so we
    # don't register categories twice.
    categories_separate = categories is not domains
    tree = assemble_tree(descriptors, groups, categories, domains, categories_separate, config.embedder)

    # Freeze — write the snapshot and bump ``version``.
    log.info("Freeze stage: writing snapshot")
    frozen = freeze(tree, config, out_root)
    log.info("Freeze stage: frozen as %s", frozen.version)
    return frozen
