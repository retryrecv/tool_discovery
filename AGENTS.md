# ToolTree — root

Workspace for the **Tool Index** project: a hierarchical index over large tool catalogs (~10k tools) so an LLM can route queries via a top-down L1 → L2 → L3 → tool traversal instead of scanning a flat list.

## Layout

- `tool_index/` — the actual Python package and pipeline. All real work happens here.
- `.Codex/` — Codex settings/hooks for this project.

See `tool_index/AGENTS.md` for build/test/dev commands.

## Conventions

- Keep `tool_index/` self-contained: pyproject, tests, data, and scripts all live inside it.
- Don't introduce top-level Python files at this root — add them under `tool_index/src/tool_index/` or `tool_index/scripts/`.
