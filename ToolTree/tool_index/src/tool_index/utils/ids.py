"""ID generation for tools and nodes.

IDs look like ``grp_ab12cd34`` — a short prefix indicating the kind
followed by an 8-char content-ish hash. The hash includes a per-process
counter, so two nodes with identical content still get distinct IDs
within a single run. This is intentional: two groups whose labeler
produced identical descriptions should still be different nodes.

Determinism note: because the counter is monotonic within a run, IDs are
stable *within a run* but not *across runs* unless the call order also
matches. That's fine for our use — snapshots carry their own IDs, so
consumers use the snapshot's IDs, not regenerated ones.
"""
from __future__ import annotations
from .hashing import short_hash

# Mutable dict-in-dict so `reset_id_counter` can rebind without globals.
# Single-process; not thread-safe (the pipeline is sequential).
_counter = {"n": 0}


def new_id(prefix: str, seed: str = "") -> str:
    """Return a fresh ID like ``{prefix}_{8charhash}``.

    Args:
        prefix: Short tag (``"tool"``, ``"grp"``, ``"cat"``, ``"dom"``,
            ``"root"``) that makes IDs easy to skim visually in logs.
        seed: Content hint folded into the hash. Doesn't need to be
            unique — the counter provides uniqueness.
    """
    _counter["n"] += 1
    tag = short_hash(f"{prefix}:{seed}:{_counter['n']}", 8)
    return f"{prefix}_{tag}"


def reset_id_counter() -> None:
    """Reset the monotonic counter.

    Useful in tests that want byte-stable IDs across two consecutive
    calls in the same process. Not needed in normal pipeline use.
    """
    _counter["n"] = 0
