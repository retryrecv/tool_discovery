"""Deterministic string hashing.

Every cache key and content-derived ID passes through here. Using SHA-256
(not Python's built-in ``hash``) guarantees stability across processes,
Python versions, and PYTHONHASHSEED — which is the whole point: re-runs
on the same inputs must produce the same IDs.
"""
from __future__ import annotations
import hashlib


def stable_hash(s: str) -> str:
    """Full 64-char hex SHA-256 of the UTF-8 encoding of ``s``.

    Used wherever the full digest is wanted (cache file names, etc.).
    """
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def short_hash(s: str, n: int = 12) -> str:
    """First ``n`` hex characters of `stable_hash(s)`.

    Collision risk at n=12 is ~5e-8 for 1M items — acceptable for tool
    and node IDs. Callers that need ironclad uniqueness should use the
    full `stable_hash` instead.
    """
    return stable_hash(s)[:n]
