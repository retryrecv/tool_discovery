# RQ-RAG: Learning to Refine Queries for Retrieval Augmented Generation

- Year: 2024
- arXiv: 2404.00610
- URL: https://arxiv.org/abs/2404.00610

## Key Idea

Adaptively decide whether a query needs refinement, and what kind of refinement
is needed. Refinement can include rewriting, decomposing, or disambiguating.

For `tool_index`, this supports a gated multi-circle planner:

- do not always ask the LLM to split
- inspect cheap retrieval signals first
- refine only when the current step is weak, ambiguous, or broad

## Important Metrics

- Accuracy improvement on single-hop and multi-hop QA.
- Refinement trigger precision and recall.
- Cost saved by avoiding unnecessary refinement.
- Quality of rewritten or decomposed queries.

For this repo, track:

- LLM call rate
- false refine rate on simple queries
- missed refine rate on complex queries
- improvement per additional circle
- p50 and p95 latency

## Recommended Steps

1. Start every step with cheap retrieval and scoring.
2. Trigger refinement only if score, margin, or path evidence is weak.
3. Keep max circles low at first: `max_circles = 2` or `3`.
4. Add a threshold sweep over score and margin.
5. Measure recall gain versus extra LLM cost.

## Investigation Notes

- This is directly relevant to production cost.
- It helps protect simple queries from unnecessary decomposition.
- It also gives a principled reason to split again only when needed.

## Fit For Next Goal

Use RQ-RAG for the adaptive gate around recursive decomposition.
