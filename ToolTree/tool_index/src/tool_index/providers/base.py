"""Provider protocols — the seams where the pipeline meets external models.

Every module that needs text generation or embeddings depends on these
protocols, never on a concrete SDK. Concrete implementations live alongside
(`llm_anthropic.py`, `llm_fake.py`, `embedding_openai.py`,
`embedding_fake.py`). Tests wire in the fakes so they stay offline and
deterministic.
"""
from __future__ import annotations
from typing import Protocol


class LLMProvider(Protocol):
    """Text-in / text-out LLM adapter.

    Implementations must be deterministic given
    ``(prompt, schema, model, temperature=0)`` so that provider caches and
    golden tests remain stable across runs.
    """

    # Short stable identifier used in cache keys and `BuildTrace`. Typically
    # the model name (e.g. ``"claude-haiku-4-5"`` or ``"fake-llm-v1"``).
    id: str

    def call(self, prompt: str, *, schema: str = "") -> str:
        """Run a single prompt and return the raw text response.

        Args:
            prompt: Fully-rendered prompt. Templates from ``prompts/`` should
                already be filled in by the caller.
            schema: Optional JSON-schema hint; implementations that support
                structured output (e.g. Anthropic tool-use) may use it to
                constrain the response. Passing an empty string disables it.

        Returns:
            The model's text output. Callers are responsible for parsing
            JSON / markdown fences if the prompt asked for them.
        """
        ...


class EmbeddingProvider(Protocol):
    """Text-to-vector embedding adapter.

    Used for near-duplicate detection in stage 1, clustering in stages 3-4,
    and similarity scoring during retrieval and validation.
    """

    # Short stable identifier (e.g. ``"text-embedding-3-small"``). Persisted
    # in snapshots so we can detect mixed-embedding-model contamination.
    id: str

    # Output dimensionality. Consumers allocate numpy arrays using this, so
    # it must match the actual vector length returned by ``embed``.
    dim: int

    def embed(self, text: str) -> list[float]:
        """Embed a single string. Returns a ``dim``-length vector."""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed many strings at once.

        Implementations should batch-call the underlying API where possible;
        order of the result must match the input order one-for-one.
        """
        ...
