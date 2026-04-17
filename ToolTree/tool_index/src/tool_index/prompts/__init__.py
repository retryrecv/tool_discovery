"""Prompt template loader — reads `.txt` files from this package.

Templates live as resource files alongside this module; `load(name)`
returns the raw text, which callers then format with `str.format`.
Keeping templates as package resources means they travel with an
installed wheel and don't rely on CWD.
"""
from importlib import resources


def load(name: str) -> str:
    """Return the raw text of the named prompt template.

    Args:
        name: Filename inside the ``prompts/`` package (e.g.
            ``"enrich_tool.txt"``). No extension is assumed — pass the
            full filename.

    Raises:
        FileNotFoundError: If the named file isn't packaged. Typically
            means a typo in the call site; add the file or fix the name.
    """
    return resources.files(__package__).joinpath(name).read_text()
