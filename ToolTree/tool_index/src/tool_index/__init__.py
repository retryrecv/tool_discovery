"""tool_index — hierarchical tool index construction pipeline.

Public surface (what importers should use):

    from tool_index import build_tree_index, Config, load_config, default_config

The single entrypoint is `build_tree_index(raw_tools, config)`. Everything
else (stages, clusterers, labelers, validators) is reachable via
submodules but isn't part of the stability contract.

On import we best-effort load a ``.env`` from the nearest ancestor
directory so hosted-provider credentials (Anthropic, Azure OpenAI) are
available without the caller having to export them manually. No error
if python-dotenv isn't installed or no ``.env`` is found.
"""
try:
    from dotenv import load_dotenv
    # `find_dotenv` walks up from CWD; safe no-op if nothing's found.
    load_dotenv()
except Exception:
    # dotenv is optional — missing dep or load failure must not break imports.
    pass

from .pipeline import build_tree_index
from .config import load_config, default_config, Config

__version__ = "0.1.0"
__all__ = ["build_tree_index", "load_config", "default_config", "Config"]
