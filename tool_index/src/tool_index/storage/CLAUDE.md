# storage

Snapshot I/O. Snapshots are immutable once written.

- `snapshot.py` — `write_snapshot(tree, path)` / `read_snapshot(path)`. Writes `tree.json`, `embeddings.json`, `build_trace.json`.
- `formats.py` — JSON (de)serializers for schema types.
- `versioning.py` — snapshot schema version constant + migration hooks.

## Conventions

- Never mutate a snapshot in place — write a new version directory (`v0`, `v1`, ...).
- Bump the version in `versioning.py` whenever `schema/` types change shape.
- Embeddings are stored separately from `tree.json` so the tree stays diff-friendly.
