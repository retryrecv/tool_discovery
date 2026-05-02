# Toolformer: Language Models Can Teach Themselves to Use Tools

- Authors: Timo Schick et al.
- Year: 2023
- arXiv: 2302.04761
- URL: https://arxiv.org/abs/2302.04761

## Key Idea

Train or adapt a language model to decide when and how to call external tools.
The paper focuses on self-supervised data generation for tool-use examples.

For `tool_index`, the most relevant lesson is tool-call granularity:

- a tool call should correspond to a useful atomic operation
- the model should learn when a tool is needed
- weak or irrelevant tool calls should be filtered out

## Important Metrics

- Downstream task improvement from tool-use augmentation.
- Correct tool selection and argument placement.
- Whether inserted tool calls improve model likelihood or final accuracy.
- Cost of generating and filtering tool-use examples.

For this repo, track:

- atomic step quality
- accepted tool-call precision
- false positive tool calls for simple queries
- training/eval examples where a tool call is unnecessary

## Recommended Steps

1. Convert successful eval cases into `query -> planned tool steps` examples.
2. Use failed cases to generate negative examples: weak top score, wrong route,
   repeated split, no matching tool.
3. Add few-shot examples to the planner prompt showing one operation per tool.
4. Later, consider distilling the planner into a cheaper local model or cached
   classifier.

## Investigation Notes

- This is not the first implementation step because it leans toward training or
  data generation.
- It becomes valuable after the recursive planner has traces and accepted/rejected
  examples.

## Fit For Next Goal

Use Toolformer as a later-stage data strategy: collect traces now so we can
train or prompt better tool-call decisions later.
