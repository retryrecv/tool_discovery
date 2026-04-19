"""Top-down beam traverser — the reference retrieval path.

Given a query embedding, walks the tree from the root, keeping the top
``beam`` nodes by cosine similarity at each level. At the leaf-parent
level (groups, L3) it collects tool IDs ranked by the parent group's
score — simple but effective.

Used by:
    • ``validation/recall_benchmark.py`` — the single gate for tree quality.
    • External consumers as a reference implementation. Real production
      retrieval would likely add per-tool re-ranking at the final step.
"""
from __future__ import annotations
import numpy as np

from ..schema import Tree, Node, LEVEL_TOOL


def _cos(a: list[float], b: list[float]) -> float:
    """Cosine similarity with safe handling of zero vectors."""
    va = np.array(a, dtype=np.float64)
    vb = np.array(b, dtype=np.float64)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0 or nb == 0:
        return 0.0
    return float(va @ vb / (na * nb))


def retrieve(tree: Tree, query_embedding: list[float], k: int = 30, beam: int = 2) -> list[str]:
    """Return up to ``k`` tool IDs for ``query_embedding``, ranked.

    Algorithm:
        1. Start at root's domain children.
        2. At each level, keep the top ``beam`` nodes by cosine similarity
           to the query.
        3. Expand their inner children and repeat.
        4. When a level has no inner children (i.e. we've reached leaf
           parents / groups), collect tool IDs scored by their parent
           group's similarity.

    Args:
        tree: The assembled tree.
        query_embedding: Query vector in the same embedding space used to
            build the tree.
        k: Maximum tool IDs to return.
        beam: Branching factor at each level. Higher = more exhaustive
            (better recall, more cost). 2 is a reasonable default for
            balanced trees.

    Returns:
        Ranked list of tool IDs, length ≤ ``k``. Deduped — a tool ID
        appears at most once.
    """
    # Start one level below root (i.e. at domains). Root has no useful
    # description of its own.
    current_nodes: list[Node] = [tree.nodes_by_id[cid] for cid in tree.root.children if cid in tree.nodes_by_id]
    while True:
        # Rank the current frontier by similarity and keep the top beam.
        scored = sorted(current_nodes, key=lambda n: -_cos(n.embedding, query_embedding))
        frontier = scored[:beam]

        # Expand frontier's inner children. Tool IDs (not in nodes_by_id)
        # signal we've hit the leaf-parent level and need to switch to
        # tool-collection mode.
        next_nodes: list[Node] = []
        for node in frontier:
            for cid in node.children:
                if cid in tree.nodes_by_id:
                    next_nodes.append(tree.nodes_by_id[cid])
        if not next_nodes:
            # Leaf-parent level reached. Score tools by their parent
            # group's similarity — we don't re-embed tools at query time.
            tool_scores: list[tuple[float, str]] = []
            for node in frontier:
                ns = _cos(node.embedding, query_embedding)
                for tid in node.children:
                    tool_scores.append((ns, tid))
            tool_scores.sort(key=lambda x: -x[0])
            # Dedupe in case two groups in the frontier both contain the
            # same tool ID (shouldn't happen in a clean tree, but be safe).
            seen: set[str] = set()
            out: list[str] = []
            for _, tid in tool_scores:
                if tid in seen:
                    continue
                seen.add(tid)
                out.append(tid)
                if len(out) >= k:
                    break
            return out
        current_nodes = next_nodes


def retrieve_with_path(
    tree: Tree, query_embedding: list[float], k: int = 30, beam: int = 2
) -> tuple[list[str], list[str]]:
    """Like ``retrieve`` but also returns the descended node path (top-1 each level).

    Path lists node_ids from the top-level domain down to the leaf-parent
    (group) chosen at each level by highest cosine similarity to the query.
    Tools themselves are not part of the path.
    """
    path: list[str] = []
    current_nodes: list[Node] = [tree.nodes_by_id[cid] for cid in tree.root.children if cid in tree.nodes_by_id]
    while True:
        scored = sorted(current_nodes, key=lambda n: -_cos(n.embedding, query_embedding))
        frontier = scored[:beam]
        if frontier:
            path.append(frontier[0].id)

        next_nodes: list[Node] = []
        for node in frontier:
            for cid in node.children:
                if cid in tree.nodes_by_id:
                    next_nodes.append(tree.nodes_by_id[cid])
        if not next_nodes:
            tool_scores: list[tuple[float, str]] = []
            for node in frontier:
                ns = _cos(node.embedding, query_embedding)
                for tid in node.children:
                    tool_scores.append((ns, tid))
            tool_scores.sort(key=lambda x: -x[0])
            seen: set[str] = set()
            out: list[str] = []
            for _, tid in tool_scores:
                if tid in seen:
                    continue
                seen.add(tid)
                out.append(tid)
                if len(out) >= k:
                    break
            return out, path
        current_nodes = next_nodes
