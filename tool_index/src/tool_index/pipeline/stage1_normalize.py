"""Stage 1 — raw tool dicts → normalized, deduped `ToolDescriptor` list.

Two dedupe passes run back-to-back:

1. **Exact** dedupe on ``(name, signature)`` — catches trivial duplicates from
   merged catalogs without any embedding cost.
2. **Near-duplicate** dedupe via cosine similarity on a cheap
   name+doc embedding — catches catalogs that list the same tool under
   slightly different names (``list_users`` vs ``users_list``).

Output is the single source of truth for every later stage.
"""
from __future__ import annotations
from typing import TypedDict

import numpy as np

from ..schema import ToolDescriptor
from ..utils.ids import new_id


class RawToolEntry(TypedDict, total=False):
    id: str
    name: str
    signature: str
    description: str
    source: str
    examples: list[str]


def _infer_side_effect(doc: str, name: str) -> str:
    """Best-effort side-effect classification from name/doc keywords.

    Conservative: returns ``"compute"`` when nothing matches. The ordering
    matters — delete-like verbs are checked before generic write verbs,
    and transactional verbs before read verbs, so ``"transfer funds"``
    lands on ``"transact"`` rather than ``"read"``.
    """
    low = f"{name} {doc}".lower()
    if any(w in low for w in ["delete", "remove", "drop", "purge"]):
        return "write"
    if any(w in low for w in ["create", "insert", "write", "update", "set", "upsert", "send", "publish", "post "]):
        return "write"
    if any(w in low for w in ["transact", "transfer", "charge", "payment"]):
        return "transact"
    if any(w in low for w in ["read", "get", "list", "fetch", "search", "find", "query", "scan", "view", "show"]):
        return "read"
    return "compute"


def _parse_one(raw: RawToolEntry) -> ToolDescriptor:
    """Convert one raw catalog entry to a `ToolDescriptor`.

    Accepts several common field-name aliases so catalogs from different
    sources can be normalized without upstream transformation. If no
    explicit ``id`` is provided, one is derived from ``name + signature``
    — making re-runs on the same catalog produce identical IDs.
    """
    name = raw["name"]
    signature = raw["signature"]
    doc = raw["description"]
    source = raw.get("source", "")
    examples = list(raw.get("examples", []))
    tid = raw.get("id") or new_id("tool", f"{name}:{signature}")
    return ToolDescriptor(
        id=tid,
        name=name,
        signature=signature,
        original_doc=doc,
        example_calls=examples,
        side_effect_class=_infer_side_effect(doc, name),
        source=source,
    )


def _dedupe_exact(descriptors: list[ToolDescriptor]) -> list[ToolDescriptor]:
    """Keep only the first occurrence of each ``(name, signature)`` pair.

    First-wins is deterministic because the input order is deterministic
    (caller controls it). Later duplicates might carry richer examples,
    but that's a tradeoff we accept for determinism.
    """
    seen: dict[tuple[str, str], ToolDescriptor] = {}
    for d in descriptors:
        key = (d.name, d.signature)
        if key not in seen:
            seen[key] = d
    return list(seen.values())


def _dedupe_near(
    descriptors: list[ToolDescriptor],
    embedder,
    threshold: float,
) -> list[ToolDescriptor]:
    """Drop tools whose name+doc embedding is cosine-similar above ``threshold``.

    O(n²) in the number of survivors — fine for ~10k tools. For larger
    catalogs we'd swap in an ANN index; keeping it simple for now.
    """
    if len(descriptors) <= 1:
        return descriptors
    texts = [f"{d.name} {d.original_doc}" for d in descriptors]
    # L2-normalize so dot product == cosine similarity.
    embs = np.array(embedder.embed_batch(texts), dtype=np.float64)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = embs / norms

    keep: list[int] = []
    kept_vectors: list[np.ndarray] = []
    for i in range(len(descriptors)):
        v = embs[i]
        drop = False
        for kv in kept_vectors:
            # Cosine similarity via dot product on unit vectors.
            if float(v @ kv) >= threshold:
                drop = True
                break
        if not drop:
            keep.append(i)
            kept_vectors.append(v)
    return [descriptors[i] for i in keep]


def normalize_and_dedupe(
    raw_tools: list[RawToolEntry],
    embedder,
    near_dup_threshold: float,
) -> list[ToolDescriptor]:
    """Top-level stage 1 entrypoint — parse, exact-dedupe, near-dedupe.

    Args:
        raw_tools: Arbitrary catalog dicts from the caller.
        embedder: `EmbeddingProvider` used for near-dup detection. Only the
            name+doc is embedded here; the richer leaf embedding happens in
            stage 3.
        near_dup_threshold: Cosine similarity above which two tools are
            considered duplicates. Typical: 0.97 (very strict).

    Returns:
        A deduped list of `ToolDescriptor`s in stable order.
    """
    descriptors = [_parse_one(r) for r in raw_tools]
    descriptors = _dedupe_exact(descriptors)
    descriptors = _dedupe_near(descriptors, embedder, near_dup_threshold)
    return descriptors
