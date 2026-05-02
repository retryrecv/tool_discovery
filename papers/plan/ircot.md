# IRCoT: Interleaving Retrieval with Chain-of-Thought Reasoning

- Authors: Harsh Trivedi et al.
- Year: 2022
- arXiv: 2212.10509
- URL: https://arxiv.org/abs/2212.10509

## Key Idea

Interleave reasoning and retrieval instead of retrieving all context once at the
beginning. Each reasoning step can trigger another retrieval step, improving
multi-hop tasks where later needs depend on earlier results.

For `tool_index`, the equivalent is:

- plan a tool-intent step
- retrieve candidate tool
- inspect candidate evidence
- refine or continue planning based on that evidence

## Important Metrics

- Answer exact match / F1 on multi-hop QA.
- Retrieval recall across hops.
- Number of retrieval calls per question.
- Improvement over retrieve-once baselines.

For this repo, track:

- per-circle tool recall
- candidate-pool gold coverage before rerank
- first failed level: L1, L2, L3, candidate, rerank
- average circles and retrieval calls per query

## Recommended Steps

1. Add per-step retrieval diagnostics as first-class data.
2. Feed diagnostics back into the planner for the next circle.
3. Use different prompts for different failure types:
   low score, low margin, L2 prune, L3 prune, repeated broad step.
4. Measure whether the second circle fixes under-split cases.

## Investigation Notes

- Our collected failures already show useful observations: L2 and L3 pruning,
  no rerank failures.
- IRCoT supports using those observations to guide the next retrieval step.

## Fit For Next Goal

Use IRCoT to justify retrieval-aware planning rather than static decomposition.
