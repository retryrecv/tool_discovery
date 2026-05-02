# Self-Ask With Search: Measuring and Narrowing the Compositionality Gap

- Authors: Ofir Press et al.
- Year: 2022
- arXiv: 2210.03350
- URL: https://arxiv.org/abs/2210.03350

## Key Idea

Have the model explicitly ask follow-up questions before answering a complex
question. Each follow-up should be simpler and independently answerable, often
with search.

For `tool_index`, follow-up questions become tool-intent steps:

- if a step cannot be resolved by one tool, ask a follow-up split question
- each follow-up should target one operation
- unresolved follow-ups become explicit `cannot_find` results

## Important Metrics

- Final answer accuracy on compositional questions.
- Quality and necessity of generated follow-up questions.
- Search success per follow-up.
- Reduction in compositionality gap versus direct answering.

For this repo, track:

- recursive split success rate
- under-split rate after each circle
- repeated or redundant sub-query rate
- per-step tool resolution rate

## Recommended Steps

1. Add a `needs_followup_split` decision after retrieval.
2. Prompt the LLM with the unresolved step plus top candidates and failure reason.
3. Ask it to produce smaller follow-up tool-intent steps or declare no matching
   operation.
4. Stop when every leaf step is resolved or rejected.
5. Evaluate by comparing final tool set to gold calls.

## Investigation Notes

- Self-Ask is useful when the first decomposition is incomplete.
- It naturally supports multi-circle refinement.
- It also supports explicit `cannot_find` instead of forcing a weak tool match.

## Fit For Next Goal

Use this paper to justify recursive splitting of unresolved sub-queries.
