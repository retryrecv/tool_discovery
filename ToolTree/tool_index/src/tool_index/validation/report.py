"""Re-export of `ValidationReport` from ``schema``.

The report dataclass lives in ``schema/tree.py`` (where it's defined
alongside ``Tree`` and ``BuildTrace``) because snapshots serialize all
three together. We re-export it here so validation callers can import it
from the same package they get the checkers from.
"""
from __future__ import annotations
from ..schema import ValidationReport  # re-export for convenience

__all__ = ["ValidationReport"]
