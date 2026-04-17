# utils

Small cross-cutting helpers. No domain logic.

- `batching.py` — chunk iterables for batched API calls.
- `hashing.py` — stable content hashing (used for cache keys + IDs). Must be deterministic across runs and Python versions.
- `ids.py` — stable tool / node ID generation from content.
- `logging.py` — structured logging setup.

## Conventions

- If something here grows domain knowledge, promote it to its own module.
- Never add I/O-heavy or model-calling code here.
