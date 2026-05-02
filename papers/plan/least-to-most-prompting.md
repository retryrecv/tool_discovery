# Least-to-Most Prompting Enables Complex Reasoning in Large Language Models

- Authors: Denny Zhou et al.
- Year: 2022
- arXiv: 2205.10625
- URL: https://arxiv.org/abs/2205.10625

## Key Idea

Split a complex task into a sequence of simpler sub-problems, then solve those
sub-problems in order. The important move is separating decomposition from
solving: first produce the smaller questions, then answer each one with the
previous answers available as context.

For `tool_index`, this supports the first circle of planning:

- user query -> atomic sub-queries
- each sub-query should be small enough to map to one tool search
- later circles refine only the sub-queries that remain unresolved

## Important Metrics

- Task accuracy after decomposition versus direct chain-of-thought.
- Generalization to harder compositions than those shown in examples.
- Number of generated sub-problems per original query.
- Failure rate from wrong or incomplete decomposition.

For this repo, track:

- `complex_set_recall@10`
- full-cover complex cases
- average sub-queries per query
- missed tools caused by under-decomposition
- false split rate on simple queries

## Recommended Steps

1. Keep the current `decompose_query` as the baseline one-shot L2M step.
2. Change output from `list[str]` to structured steps with stable IDs.
3. Add expected cardinality per step: ideally `expected_tools: 1`.
4. Evaluate whether each generated step maps to one confident tool.
5. Send weak or multi-intent steps into another refinement circle.

## Investigation Notes

- L2M alone does not guarantee that each sub-query maps to one tool.
- Current failure example: `Fetch the page at the provided URL and return the response`
  still hides both `url_parse` and `http_get`.
- L2M is best used as the first planner pass, not the final decision loop.

## Fit For Next Goal

Use this paper as the base justification for decomposition, but combine it with
ReAct or Self-Ask style loops to decide when to split again.
