# config

YAML config loading.

- `loader.py` — load + validate a config YAML into a typed object. Config files live in `tool_index/configs/`.

## Conventions

- All tunable knobs (fanout targets, thresholds, model names, seeds) come from config — no magic numbers scattered through code.
- Validate on load; fail fast with a clear error if a required field is missing.
