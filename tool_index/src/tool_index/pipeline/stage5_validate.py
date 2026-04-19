"""Stage 5 — run every validator against the assembled tree.

Produces a single `ValidationReport`. Fatal issues set ``passed=False`` via
`report.fail()`; the orchestrator decides (via ``strict``) whether that
aborts the build. Non-fatal observations go to `report.warn()`.

The three validator families:
    • structural — shape invariants (depth, fanout, orphans)
    • discriminability — sibling descriptions must be meaningfully distinct
    • recall — synthetic queries must route to the right tool top-k of the time
"""
from __future__ import annotations

from ..schema import Tree, Enrichment, ValidationReport
from ..validation import (
    check_structural,
    check_sibling_discriminability,
    generate_synthetic_queries,
    run_retrieval_benchmark,
)


def validate_tree(
    tree: Tree,
    enrichments: dict[str, Enrichment],
    embedder,
    judge_llm,
    *,
    labeler_llm,
    fanout: dict,
    expected_depth: int,
    discriminability_threshold: float,
    synthetic_per_tool: int,
    recall_k: int,
    min_recall: float,
    queries: list[dict] | None = None,
    recall_beam: int = 2,
    rerank_k: int | None = None,
    tool_vectors: dict[str, list[list[float]]] | None = None,
) -> ValidationReport:
    """Run all validators and return a populated report.

    Args:
        tree: The fully-assembled `Tree` (root + all inner nodes + tools).
        enrichments: Stage 2 output. Used to synthesize eval queries — the
            LLM paraphrases each tool's intent into realistic user queries.
        embedder: Used by the recall benchmark to embed queries at eval time.
        judge_llm: LLM that scores pairwise sibling discriminability.
            Can be the same as the labeler — they read different prompts.
        labeler_llm: LLM that synthesizes eval queries from enrichments.
        fanout: Per-level ``{name: (min, max)}`` bounds from config.
        expected_depth: Depth the structural validator should check for.
            The orchestrator passes the intended build shape: ``5`` for the
            full root→domain→category→group→tool tree, or ``4`` when stage
            4b legally collapses category into domain.
        discriminability_threshold: Score below which a sibling pair is
            considered indistinguishable. Warnings, not errors.
        synthetic_per_tool: Queries to generate per tool for the recall
            benchmark. More queries → more statistical power, more LLM cost.
        recall_k: Top-k used for the recall@k metric.
        min_recall: Hard floor. Below this, the report fails.

    Returns:
        A `ValidationReport`. ``passed`` may be True or False — callers
        decide what to do about it.
    """
    report = ValidationReport()

    check_structural(tree, fanout, expected_depth, report)
    check_sibling_discriminability(tree, judge_llm, discriminability_threshold, report)

    queries = queries if queries is not None else generate_synthetic_queries(enrichments, labeler_llm, synthetic_per_tool)
    recall = run_retrieval_benchmark(
        tree, queries, embedder, recall_k,
        beam=recall_beam, rerank_k=rerank_k, tool_vectors=tool_vectors,
    )
    report.recall_at_k = recall
    # Store the eval set on the report so stage 6 can persist it.
    report.seed_eval_set = queries

    if recall < min_recall:
        report.fail(f"synthetic recall {recall:.3f} below {min_recall}")

    return report
