# ReAct: Synergizing Reasoning and Acting in Language Models

- Authors: Shunyu Yao et al.
- Year: 2022
- arXiv: 2210.03629
- URL: https://arxiv.org/abs/2210.03629

## Key Idea

Interleave reasoning with actions and observations. The model does not decide
everything up front. It reasons, acts, observes the result, then decides the
next action.

For `tool_index`, retrieval confidence is the observation:

- reason: what operation is this step asking for?
- act: retrieve candidate tools
- observe: top score, margin, path stability, candidate family
- decide: accept one tool, split again, or fail

## Important Metrics

- Task success rate in environments requiring actions.
- Number of action steps.
- Error recovery after a bad action or observation.
- Interpretability of reasoning/action traces.

For this repo, track:

- accepted tool precision
- unresolved rate
- average retrieval attempts per query
- refinement depth distribution
- failure reason counts: low score, low margin, route prune, repeated split

## Recommended Steps

1. Add a trace object for every planner circle.
2. Treat retrieval scores and level paths as observations.
3. Use thresholds to decide whether to accept, refine, or fail.
4. Preserve the trace in evaluation output so failures are explainable by level.
5. Keep the loop bounded with `max_circles` and repeated-text detection.

## Investigation Notes

- This paper supports iterative control, not just better prompting.
- The LLM should not blindly split forever. It should react to concrete retrieval
  evidence.
- The trace will be valuable for comparing L2 versus L3 failures.

## Fit For Next Goal

Use ReAct as the control-loop model for recursive decomposition and tool
resolution.
