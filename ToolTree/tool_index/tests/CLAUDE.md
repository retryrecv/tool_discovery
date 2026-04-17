# tests

- `unit/` — fast, isolated, use fake providers. One test file per module under test.
- `integration/` — end-to-end pipeline on small fixtures. Still uses fake providers — no network.
- `golden/` — snapshot tests: build a tree on a fixture, compare to a stored expected output. Regenerate with the golden-update flag (see `conftest.py`).
- `fixtures/` — small tool catalogs for tests (`mini_tools.json`, etc.).
- `conftest.py` — shared pytest fixtures (fake providers, tmp snapshot dirs).

## Conventions

- No network calls in tests, ever. Use `providers/llm_fake.py` + `embedding_fake.py`.
- Golden files are committed — review diffs carefully; a golden change means the pipeline output changed.
- Run with `pytest tests/ -x` from the `tool_index/` directory.
