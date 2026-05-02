# Decomposed Prompting: A Modular Approach for Solving Complex Tasks

- Authors: Tushar Khot et al.
- Year: 2022
- arXiv: 2210.02406
- URL: https://arxiv.org/abs/2210.02406

## Key Idea

Use a top-level controller to decompose a task and dispatch each sub-task to a
specialized handler. This is closer to our tool-routing problem than plain
chain-of-thought because every sub-task is meant for a module, not just a text
answer.

For `tool_index`, the modules are retrieved tools or tool families:

- split query into steps
- route each step to one tool-search context
- execute or mark unresolved

## Important Metrics

- End-task accuracy versus monolithic prompting.
- Correct module selection rate.
- Sub-task completion accuracy.
- Error propagation from a bad decomposition or wrong module route.

For this repo, track:

- step-to-tool exact match rate
- route correctness at L1/L2/L3
- unresolved step rate
- number of refinement circles per query
- wrong-route rate where the gold tool becomes unreachable

## Recommended Steps

1. Introduce a structured planner output:
   `[{step, intent, expected_capability, target_level_hint}]`.
2. Retrieve per step, not per original query.
3. Add a route diagnostic: did the selected L1/L2/L3 path contain the gold tool?
4. If route confidence is weak, ask the LLM to rewrite or split the step.
5. Keep any domain/category restriction soft until routing accuracy is measured.

## Investigation Notes

- The repo already tried a routed-decomposition direction and archived it when
  routing reduced recall in one setup.
- That does not invalidate modular routing. It means hard L1/L2 restriction is
  risky unless route confidence is high.
- Start with diagnostics and soft hints before enforcing route constraints.

## Fit For Next Goal

This is the main paper for the one-step-one-handler design. It supports moving
from fuzzy retrieval union to explicit planned tool calls.
