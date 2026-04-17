"""Batching helper — chunk a list into fixed-size sub-lists.

Used by stage 2 to define cache-flush boundaries, and available to any
future stage that wants to bound its memory or concurrency footprint.
"""
from __future__ import annotations
from typing import Iterable, TypeVar, Iterator

# Generic so the helper preserves the element type — callers get back
# `list[T]` chunks without casting.
T = TypeVar("T")


def chunks(seq: list[T], size: int) -> Iterator[list[T]]:
    """Yield successive slices of ``seq`` of length ``size`` (last may be shorter).

    Example:
        >>> list(chunks([1, 2, 3, 4, 5], 2))
        [[1, 2], [3, 4], [5]]
    """
    for i in range(0, len(seq), size):
        yield seq[i:i + size]
