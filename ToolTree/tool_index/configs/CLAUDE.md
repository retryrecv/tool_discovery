# configs

YAML configs consumed by `config/loader.py`.

- `default.yaml` — baseline; referenced by tests and the README example.
- `dev.yaml` — small fanouts, fake providers, fast iteration.
- `prod.yaml` — real providers, production thresholds.

## Conventions

- All tunable knobs live here — never hardcode thresholds, fanouts, or model names in Python.
- Keep `default.yaml` runnable end-to-end on the mini fixture so the README example always works.
