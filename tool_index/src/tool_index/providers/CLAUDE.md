# providers

All external model access lives here. Nothing else in the codebase imports `anthropic` / `openai` directly.

- `base.py` — `LLMProvider`, `EmbeddingProvider` protocols.
- `llm_anthropic.py` — Anthropic Claude adapter. Use prompt caching for repeated system prompts.
- `llm_fake.py` — deterministic fake LLM for tests (canned responses keyed by prompt hash).
- `embedding_openai.py` — OpenAI embeddings adapter.
- `embedding_fake.py` — deterministic fake embedder (hash → vector) for tests.
- `cache.py` — on-disk cache keyed by `(provider, model, input_hash)`. Cache dir: `data/cache/<provider>-<model>/`.

## Conventions

- Every provider method must be deterministic given `(inputs, model, temperature=0, seed)`.
- Fakes must produce stable outputs across runs — tests rely on this for golden files.
- When adding a real provider, also add a fake counterpart so tests stay offline.
