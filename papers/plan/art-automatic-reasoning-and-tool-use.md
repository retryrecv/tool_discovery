# ART: Automatic Multi-Step Reasoning and Tool-Use for Large Language Models

- Authors: Bhargavi Paranjape et al.
- Year: 2023
- arXiv: 2303.09014
- URL: https://arxiv.org/abs/2303.09014

## Key Idea

Use task demonstrations to generate multi-step reasoning programs that include
tool calls. The system can reuse examples from similar tasks and compose tool
use over multiple steps.

For `tool_index`, this supports building a case library:

- known complex query patterns
- expected atomic steps
- expected tool per step
- recovery behavior for unresolved steps

## Important Metrics

- Task accuracy across tool-using benchmarks.
- Correctness of generated intermediate programs.
- Tool-call success and error recovery.
- Benefit from retrieving similar demonstrations.

For this repo, track:

- planner accuracy with and without retrieved examples
- examples retrieved per query
- improvement on ultra-complex cases
- rate of invalid or unsupported planned operations

## Recommended Steps

1. Store successful tool plans from eval cases as examples.
2. Retrieve similar examples before prompting the planner.
3. Prompt the planner to emit structured steps using the retrieved examples.
4. Compare no-example versus example-conditioned recursive planning.
5. Add examples only if they improve coverage without overfitting simple cases.

## Investigation Notes

- Example retrieval is likely useful for repeated workflow shapes: fetch-format-query,
  rewards-account-redemption, date-convert-diff.
- The risk is memorizing current eval cases instead of improving general routing.
- Keep a held-out set of ultra-complex cases for honest measurement.

## Fit For Next Goal

Use ART after the recursive planner exists, as a way to improve hard and
ultra-complex decomposition quality.
