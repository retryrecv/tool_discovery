"""LLM-powered cluster description.

Given a cluster's members plus (optionally) its nearest-neighbor clusters,
produces a short single-line description that summarizes what this
cluster is *and* what distinguishes it from its siblings. The single-line
constraint matters: these descriptions are embedded and used directly in
retrieval; multi-line text dilutes the semantic signal.
"""
from __future__ import annotations
from .. import prompts as prompt_pkg
from .contrastive import build_contrastive_prompt_context


def llm_describe_cluster(
    members_text: list[str],
    neighbors_text: list[list[str]],
    llm,
    contrastive: bool = True,
) -> str:
    """Generate a one-line description for a cluster via the labeler LLM.

    Args:
        members_text: Textual form of each member (leaf text for stage 3,
            child description for stage 4).
        neighbors_text: One text-list per neighbor cluster. Used for
            contrastive framing; ignored when ``contrastive=False``.
        llm: Labeler `LLMProvider`.
        contrastive: If False, the neighbors block is omitted from the
            prompt — the LLM writes a standalone description with no
            sibling awareness. Useful for testing but produces lower-quality
            retrieval behavior.

    Returns:
        A single non-empty line. If the LLM returns an empty string, we
        substitute ``"Unlabeled cluster"`` so downstream code never has
        to handle blank descriptions.
    """
    template = prompt_pkg.load("describe_cluster.txt")
    members_block, neighbors_block = build_contrastive_prompt_context(
        members_text, neighbors_text if contrastive else []
    )
    prompt = template.format(members=members_block, neighbors=neighbors_block)
    desc = llm.call(prompt, schema="describe_cluster").strip()
    if not desc:
        desc = "Unlabeled cluster"
    # Force single line — the description is embedded as-is and extra
    # lines hurt retrieval quality.
    return desc.splitlines()[0]
