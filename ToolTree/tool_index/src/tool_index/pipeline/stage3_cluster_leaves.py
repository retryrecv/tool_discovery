"""Stage 3 — cluster tool leaves into L3 "group" nodes.

Flow:
    1. Compose a leaf text per tool from its `Enrichment` and embed it.
    2. Density-cluster (HDBSCAN-style) those embeddings into groups.
    3. Rebalance so each group sits inside the configured fanout bounds.
    4. For each group, call the labeler LLM with the group's members *and*
       its nearest-neighbor groups — that contrastive context nudges the
       LLM to write a description that discriminates from siblings.
    5. Wrap each group in a `Node` at level L3.
"""
from __future__ import annotations

from ..schema import ToolDescriptor, Enrichment, Node, LEVEL_GROUP
from ..clustering import hdbscan_cluster, rebalance_clusters, nearest_clusters
from ..labeling import llm_describe_cluster
from ..utils.ids import new_id


def cluster_tools_into_groups(
    descriptors: list[ToolDescriptor],
    enrichments: dict[str, Enrichment],
    embedder,
    llm,
    fanout_tool: tuple[int, int],
    distance_threshold: float,
) -> list[Node]:
    """Build L3 group nodes from enriched tool descriptors.

    Args:
        descriptors: Stage 1 output; the source of truth for tool identity.
        enrichments: Stage 2 output, keyed by ``descriptor.id``. Must cover
            every descriptor.
        embedder: `EmbeddingProvider` for both leaf and group embeddings.
            Using the same model for both keeps cosine distances comparable
            across levels during retrieval.
        llm: Labeler `LLMProvider` — writes the group descriptions.
        fanout_tool: ``(min, max)`` number of tools per group. Rebalance
            splits oversized clusters and merges undersized ones to honor
            these bounds.
        distance_threshold: Cosine distance cutoff for the initial HDBSCAN
            merge. Smaller = tighter, more numerous groups.

    Returns:
        A list of L3 `Node`s whose ``children`` contain tool IDs (not
        other node IDs). Not yet registered in a `Tree` — the orchestrator
        does that.
    """
    # Leaf text format comes from `Enrichment.compose_leaf_text`; see that
    # docstring for why the format is load-bearing.
    texts = [enrichments[d.id].compose_leaf_text() for d in descriptors]
    embeddings = embedder.embed_batch(texts)

    min_size, max_size = fanout_tool
    clusters = hdbscan_cluster(embeddings, min_size, max_size, distance_threshold)
    clusters = rebalance_clusters(clusters, embeddings, min_size, max_size)

    # Labeling needs *text* for members + neighbors — embeddings aren't
    # human-readable. Cache this list so we can index it alongside the
    # `clusters` index list without recomputing.
    cluster_texts_per = [[texts[i] for i in c] for c in clusters]
    nodes: list[Node] = []
    for idx, c in enumerate(clusters):
        # k=3 neighbors is a pragmatic default — enough for the labeler to
        # see contrasts, not so many that the prompt blows up.
        nbr_idxs = nearest_clusters(c, clusters, embeddings, k=3)
        neighbors = [cluster_texts_per[j] for j in nbr_idxs]
        desc = llm_describe_cluster(cluster_texts_per[idx], neighbors, llm, contrastive=True)
        # Node embedding re-embeds the *description*, not the member texts
        # — this is what the retrieval traverser scores against at L3.
        node_emb = embedder.embed(desc)
        node = Node(
            id=new_id("grp", desc),
            level=LEVEL_GROUP,
            description=desc,
            embedding=node_emb,
            children=[descriptors[i].id for i in c],
        )
        nodes.append(node)
    return nodes
