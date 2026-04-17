"""Deterministic hash-based embedder — the test/dev embedding provider.

Implements the `EmbeddingProvider` protocol without any network calls.
Uses a simple bag-of-words representation where each lowercase alphanumeric
token hashes to a slot in a ``dim``-sized vector; counts are L2-normalized.

Retrieval intuition: tools that share vocabulary land near each other in
this space, which is good enough for unit and integration tests with a
curated corpus. Not meant to be used in production.
"""
from __future__ import annotations
import hashlib
import math
import re

import numpy as np


# Tokens are runs of lowercase letters/digits. Case-folding happens before
# the regex runs.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization. No stemming, no stop-word removal."""
    return _TOKEN_RE.findall(text.lower())


class FakeEmbeddingProvider:
    """Hash-BOW embedder — deterministic, fast, offline."""

    def __init__(self, dim: int = 64, model_id: str = "fake-embed-64"):
        """
        Args:
            dim: Output vector dimensionality. Default 64 is plenty for
                test corpora; production-scale collisions would hurt recall.
            model_id: Stable identifier used in cache keys and build traces.
        """
        self.dim = dim
        self.id = model_id

    def _slot(self, token: str) -> int:
        """Map a token to a slot index via MD5 (stable across processes).

        MD5 is fine here — we're not using it for security, just as a
        cheap hash function with good distribution. Only the first 4 bytes
        are used, which is enough for ``dim`` up to ~4 billion.
        """
        h = hashlib.md5(token.encode("utf-8")).digest()
        return int.from_bytes(h[:4], "big") % self.dim

    def embed(self, text: str) -> list[float]:
        """Embed one string.

        Zero-vector (empty text) is returned as-is without division — the
        downstream clusterer handles zero-norm vectors explicitly.
        """
        vec = np.zeros(self.dim, dtype=np.float64)
        for tok in _tokenize(text):
            vec[self._slot(tok)] += 1.0
        norm = math.sqrt(float((vec * vec).sum()))
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed each text independently. No actual batching happens here —
        the fake is cheap enough that a loop is fine."""
        return [self.embed(t) for t in texts]
