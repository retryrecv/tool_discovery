"""`ToolDescriptor` — the normalized, deduped form of a tool definition.

Stage 1 (`pipeline/stage1_normalize.py`) converts arbitrary raw tool dicts
(whatever shape the catalog arrived in) into this uniform structure. Every
later stage consumes `ToolDescriptor`s, never the raw input.

The ``id`` is content-derived (see `utils/ids.py`) so the same tool always
lands on the same ID across runs — this keeps snapshots diff-friendly and
lets the disk cache survive catalog reshuffles.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict


@dataclass
class ToolDescriptor:
    """A single tool, normalized for indexing.

    Instances are treated as immutable after stage 1 — downstream code reads
    these fields but never mutates them. Enrichment (LLM-generated metadata)
    lives in a separate `Enrichment` object keyed by the same ``id``.
    """

    # Content-derived stable ID (hash of name + signature). Used as the key
    # in `Tree.tools_by_id`, in cache lookups, and as the leaf identifier
    # inside `Node.children` at the L3 level.
    id: str

    # Canonical tool name, e.g. ``"db_users_read"``. Not guaranteed unique
    # across catalogs — always pair with ``id`` for identity checks.
    name: str

    # Callable signature as a single string, e.g. ``"foo(x, y) -> z"``.
    # Used by enrichment prompts to ground the LLM's understanding of
    # input/output shapes.
    signature: str

    # The tool's docstring exactly as it arrived in the raw catalog. Kept
    # verbatim so we can re-enrich later without losing source fidelity.
    original_doc: str

    # Optional worked examples: list of ``{"args": ..., "returns": ...}``.
    # Fed into enrichment prompts when present; empty list is fine.
    example_calls: list[dict] = field(default_factory=list)

    # One of `schema.constants.SIDE_EFFECTS`. Default ``"compute"`` is the
    # most conservative (no external I/O) — stage 1 may upgrade it based on
    # doc heuristics. Consumed by validators and future routing policies.
    side_effect_class: str = "compute"

    # Free-form provenance string (e.g. catalog filename, API spec URL).
    # Informational only; never parsed.
    source: str = ""

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON snapshots."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ToolDescriptor":
        """Rehydrate from a snapshot dict, ignoring unknown keys.

        Silently dropping unknown keys lets us read older snapshots after
        adding new optional fields without a migration.
        """
        fields = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in fields})
