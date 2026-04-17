"""Stage 4 — recursive upward clustering (L3 → L2, then L2 → L1).

Unlike stage 3 (which clusters tools), this stage clusters *descriptions* —
the children's textual summaries, already embedded by a prior pass. That
matters: by the time we're building categories and domains, we want the
index to reflect conceptual neighborhoods of whole groups, not of
individual tools.

Called twice by the orchestrator, once per level. The derived
``parent_level`` is sanity-checked against the caller's ``parent_level``
arg but the caller has the final say.
"""
from __future__ import annotations

from ..schema import Node, LEVEL_CATEGORY, LEVEL_DOMAIN
from ..clustering import agglomerative_cluster, rebalance_clusters, nearest_clusters
from ..labeling import llm_describe_cluster
from ..utils.ids import new_id


# child_level → (parent_level_constant, id_prefix). Lookup table keeps the
# level-promotion logic in one place and prevents typos in level strings.
_LEVEL_PARENT = {
    "L3": (LEVEL_CATEGORY, "cat"),
    "L2": (LEVEL_DOMAIN, "dom"),
}


def cluster_upward(
    children: list[Node],
    parent_level: str,
    fanout: tuple[int, int],
    distance_threshold: float,
    embedder,
    llm,
) -> list[Node]:
    """Cluster a level of `Node`s into their parent level.

    Args:
        children: Nodes at a single level (all L3, or all L2).
        parent_level: The level to assign the new parents. Passed explicitly
            by the orchestrator for clarity — the function also derives it
            from ``children[0].level`` as a sanity check.
        fanout: ``(min, max)`` children per parent at the target level.
        distance_threshold: Cosine distance cutoff for agglomerative
            merges. Larger thresholds at higher levels produce broader
            buckets — typical: 0.3 (group), 0.45 (category), 0.7 (domain).
        embedder: Used to embed the *new parent descriptions* so retrieval
            can score them.
        llm: Labeler LLM — writes contrastive descriptions from child
            descriptions + nearest-neighbor clusters.

    Returns:
        A list of parent `Node`s at ``parent_level`` whose ``children``
        contain child node IDs. Not yet attached to a `Tree`.

    Raises:
        ValueError: if ``children[0].level`` isn't a level we know how to
            promote (i.e. not L3 or L2).
    """
    # Derive parent level from the actual data; caller's `parent_level`
    # should agree. Empty input defaults to L3 just to produce a sensible
    # error — an empty list can't be clustered meaningfully.
    child_level = children[0].level if children else "L3"
    if child_level not in _LEVEL_PARENT:
        raise ValueError(f"cannot cluster upward from level {child_level}")
    parent_level_const, prefix = _LEVEL_PARENT[child_level]
    if parent_level != parent_level_const:
        # Allow caller to request explicit level but keep derived if blank
        pass

    # Embeddings here are the *description* embeddings of the children,
    # which were computed by the previous stage.
    embeddings = [c.embedding for c in children]
    min_size, max_size = fanout
    clusters = agglomerative_cluster(embeddings, distance_threshold, max_size)
    clusters = rebalance_clusters(clusters, embeddings, min_size, max_size)

    # For the labeler: show it *child descriptions*, not tool texts — this
    # level of the tree summarizes subtrees, not individual tools.
    child_desc_per = [[children[i].description for i in c] for c in clusters]
    parents: list[Node] = []
    for idx, c in enumerate(clusters):
        nbr_idxs = nearest_clusters(c, clusters, embeddings, k=3)
        neighbors = [child_desc_per[j] for j in nbr_idxs]
        desc = llm_describe_cluster(child_desc_per[idx], neighbors, llm, contrastive=True)
        node_emb = embedder.embed(desc)
        node = Node(
            id=new_id(prefix, desc),
            level=parent_level_const,
            description=desc,
            embedding=node_emb,
            children=[children[i].id for i in c],
        )
        parents.append(node)
    return parents
