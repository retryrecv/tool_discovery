# scripts

One-off utilities. Not part of the package API.

- `inspect_tree.py` — pretty-print a frozen snapshot for debugging.

## Conventions

- Scripts are standalone Python files, runnable via `python scripts/<name>.py`.
- Import from `tool_index.*` freely — the package is installed in editable mode.
- If a script becomes load-bearing (used in CI or by multiple people), promote it into the CLI (`src/tool_index/cli.py`) instead.
