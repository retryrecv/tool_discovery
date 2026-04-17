"""Sibling discriminability validator.

For each non-leaf node we ask the judge LLM to score how distinguishable
each sibling-description pair is. Low-scoring pairs become warnings (not
errors) — a tree can still be useful even if a few sibling pairs overlap
semantically, but too many overlaps mean retrieval will struggle to pick
the right subtree at that level.
"""
from __future__ import annotations
from itertools import combinations

from ..schema import Tree, ValidationReport
from .. import prompts as prompt_pkg


def check_sibling_discriminability(tree: Tree, llm, threshold: float, report: ValidationReport) -> None:
    """Score every sibling pair and record low-scoring pairs.

    Args:
        tree: Tree to check.
        llm: Judge `LLMProvider`. Returns a numeric score in ``[0, 1]``
            where higher means more distinguishable.
        threshold: Score below which a pair is flagged.
        report: Mutated with warnings and a ``low_discriminability_pairs``
            count in ``details``.
    """
    template = prompt_pkg.load("judge_discriminability.txt")
    # Tuples of ``(id_a, id_b, score)``. Only low-scoring pairs are kept —
    # high-scoring pairs don't need tracking.
    low_pairs: list[tuple[str, str, float]] = []
    for parent in tree.all_nodes():
        # Only inner children matter — tool leaves don't have descriptions
        # at this level of granularity.
        inner = [cid for cid in parent.children if cid in tree.nodes_by_id]
        if len(inner) < 2:
            continue
        for a_id, b_id in combinations(inner, 2):
            a = tree.nodes_by_id[a_id].description
            b = tree.nodes_by_id[b_id].description
            out = llm.call(template.format(a=a, b=b), schema="judge_discriminability").strip()
            try:
                score = float(out)
            except ValueError:
                # Malformed LLM output → neutral score, not a failure.
                # We'd rather skip this pair than tank the whole build on
                # a parse error.
                score = 0.5
            if score < threshold:
                low_pairs.append((a_id, b_id, score))
    report.details["low_discriminability_pairs"] = len(low_pairs)
    # Cap warnings at 20 to keep logs readable — the count in ``details``
    # still reflects the true total.
    for a, b, s in low_pairs[:20]:
        report.warn(f"low discriminability {a} vs {b}: {s:.2f}")
