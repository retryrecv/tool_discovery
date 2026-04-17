# labeling

LLM-generated descriptions for cluster nodes.

- `describe.py` — generate a node description from its children (tools or sub-nodes).
- `contrastive.py` — generate descriptions that emphasize what makes a cluster *different* from its siblings. Improves top-down retrieval discriminability.

## Conventions

- All LLM access via `providers.LLMProvider` — never import SDKs.
- Prompts live in `prompts/`, not inline strings here.
- Contrastive labeling needs sibling context; pass siblings explicitly rather than reading from a tree object.
