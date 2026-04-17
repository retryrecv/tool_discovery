"""Prompt-context builders for contrastive cluster labeling.

The labeler LLM gets two blocks: a bulleted list of *this* cluster's
members, and a bulleted list of each neighbor cluster's members. Keeping
the prompt shape here (rather than in ``describe.py``) makes it easy to
tweak formatting without touching the LLM-call layer.
"""
from __future__ import annotations


def summarize_members(members: list[str], limit: int = 10) -> str:
    """Render up to ``limit`` member texts as a bulleted list.

    Truncating keeps the prompt bounded — for large clusters the LLM
    doesn't need every example to write a good description, and long
    prompts hurt latency and cost.
    """
    lines = members[:limit]
    return "\n".join(f"- {m}" for m in lines)


def build_contrastive_prompt_context(
    members_text: list[str],
    neighbors_text: list[list[str]],
) -> tuple[str, str]:
    """Build the ``(members_block, neighbors_block)`` pair for the prompt.

    Args:
        members_text: Textual representation of each member in the cluster
            being labeled.
        neighbors_text: One text-list per nearest-neighbor cluster.
            Pass ``[]`` to disable the contrastive block entirely (the
            labeler falls back to plain description).

    Returns:
        ``(members_block, neighbors_block)``. ``neighbors_block`` is the
        literal string ``"(none)"`` when there are no neighbors — the
        prompt template expects a non-empty placeholder.
    """
    m = summarize_members(members_text)
    n_blocks = []
    for k, nbr in enumerate(neighbors_text):
        # Neighbor summaries use a tighter limit (5) than the focal
        # cluster — they're context, not the thing we're describing.
        n_blocks.append(f"neighbor_{k}:\n{summarize_members(nbr, limit=5)}")
    n = "\n".join(n_blocks) if n_blocks else "(none)"
    return m, n
