"""`Enrichment` — LLM-generated metadata that makes a tool retrievable.

Stage 2 (`pipeline/stage2_enrich.py`) produces one `Enrichment` per
`ToolDescriptor`. The two structures are kept parallel (same id space) but
separate so we can re-enrich without touching the raw descriptor, and so
provider caches are keyed by the enrichment prompt rather than the tool's
full definition.

The output of `compose_leaf_text` is what stage 3 actually embeds when
clustering leaves — tweaking its format changes every embedding downstream.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict


@dataclass
class Enrichment:
    """LLM-generated retrieval hints for a single tool.

    Fields are intentionally short and high-signal — this is what the
    embedding model sees, not prose documentation.
    """

    # One-sentence paraphrase of what the tool *does* from the user's POV,
    # e.g. "read a user record by primary key". Dominates the embedding's
    # semantic position, so quality here drives retrieval quality everywhere.
    intent_phrase: str

    # Coarse tag for the primary input shape, e.g. ``"id"``, ``"query"``,
    # ``"file_path"``. Used by the labeler to describe what a cluster
    # consumes, and by validators to catch mismatched siblings.
    input_kind: str

    # Coarse tag for the primary output shape, e.g. ``"record"``,
    # ``"records"``, ``"bytes"``. Same roles as ``input_kind``.
    output_kind: str

    # Alternate phrasings the tool might be described by ("fetch user",
    # "get user details"). Widen the embedding's surface area so paraphrased
    # queries still land nearby.
    synonyms: list[str] = field(default_factory=list)

    # Realistic natural-language queries a user might ask that this tool
    # should answer. Fed into the embedding alongside the intent phrase.
    example_queries: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON snapshots."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Enrichment":
        """Rehydrate from a snapshot dict, ignoring unknown keys."""
        fields = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in fields})

    def compose_leaf_text(self) -> str:
        """Build the text that represents this tool in embedding space.

        Called by stage 3 right before embedding. The exact format matters —
        changing it invalidates every cached embedding for this tool.
        """
        parts = [
            self.intent_phrase,
            f"input: {self.input_kind}",
            f"output: {self.output_kind}",
            "queries: " + " | ".join(self.example_queries),
        ]
        return "\n".join(parts)
