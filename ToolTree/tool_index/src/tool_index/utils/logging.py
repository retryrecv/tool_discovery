"""Structured logging setup — one logger per module, shared formatter.

`get_logger` is idempotent on first-call configuration. Callers use the
standard ``get_logger(__name__)`` pattern; we standardize the format and
level here so every stage logs consistently without each having to
configure ``logging`` itself.
"""
from __future__ import annotations
import logging
import sys

# Tracks whether `basicConfig` has run — subsequent `get_logger` calls
# just return the named logger without re-configuring the root handler.
_configured = False


def get_logger(name: str = "tool_index") -> logging.Logger:
    """Return a configured logger, setting up root handlers on first call.

    All pipeline logs go to stderr at INFO level by default. Override via
    the standard ``logging`` API if you need something different.
    """
    global _configured
    if not _configured:
        logging.basicConfig(
            stream=sys.stderr,
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        _configured = True
    return logging.getLogger(name)
