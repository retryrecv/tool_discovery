# Demonstrate-Search-Predict: Composing Retrieval and Language Models

- Authors: Omar Khattab et al.
- Year: 2022
- arXiv: 2212.14024
- URL: https://arxiv.org/abs/2212.14024

## Key Idea

Compose demonstrations, retrieval, and prediction into a modular pipeline. The
paper is relevant because it treats retrieval as a component inside a larger
program, not as a one-shot pre-processing step.

For `tool_index`, tool retrieval should be one module inside a planner program:

- demonstrate expected tool plans
- search for tools per atomic step
- predict accept/refine/fail decisions

## Important Metrics

- End-task accuracy from composed pipelines.
- Retrieval contribution versus demonstration contribution.
- Robustness across multi-hop and knowledge-intensive tasks.
- Cost of additional retrieval and LM calls.

For this repo, track:

- plan-level exact coverage
- per-step retrieval recall
- effect of few-shot plan examples
- latency and LLM-call count per query

## Recommended Steps

1. Define the planner pipeline explicitly as small modules:
   split, retrieve, score, verify, refine.
2. Add evaluation logs for each module.
3. Add few-shot demonstrations only to the split/refine prompt, not to retrieval.
4. Compare modular variants with ablations.

## Investigation Notes

- This paper supports system design discipline: separate modules and measure each
  one.
- It argues against hiding all behavior inside one large prompt.

## Fit For Next Goal

Use DSP as architectural support for a modular recursive planner with measurable
interfaces.
