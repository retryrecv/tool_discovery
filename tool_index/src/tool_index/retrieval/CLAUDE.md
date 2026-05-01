# retrieval

Minimal top-down traverser. Used by `scripts/eval_real_cases.py` and as the reference implementation for downstream consumers.

- `traverser.py` — `traverse(tree, query, k)` → ranked tool IDs. Walks L1 → L2 → L3 → tools, keeping top-k at each level.

## Conventions

- Keep this module tiny and dependency-light. It's a reference, not a production retriever.
- Production retrieval (if built) should live in a separate package that depends on the frozen snapshot format.
