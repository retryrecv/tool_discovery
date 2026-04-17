"""Module-runnable entrypoint: ``python -m tool_index``.

Delegates to `cli.main`. Kept as a separate file (rather than inlined in
``cli.py``) to follow the standard pattern and let `python -m tool_index`
work without import side effects beyond what ``cli.py`` already has.
"""
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
