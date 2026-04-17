"""Config loader and default config.

The `Config` dataclass is the single object threaded through the pipeline.
It owns both the knobs (thresholds, fanout, batch sizes) and the
instantiated providers (LLMs, embedder, cache). Keeping them together means
every stage can ask the config for what it needs without importing the
provider registry directly.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..providers import make_llm, make_embedding, DiskCache, LLMProvider, EmbeddingProvider


@dataclass
class Config:
    """Fully-resolved pipeline configuration.

    Two phases of use:
        1. Construction — either via `load_config(yaml_path)` or by
           instantiating directly and filling in providers manually (see
           `scripts/verify_dynamic_tools.py` for the latter).
        2. Consumption — `build_tree_index(tools, config)` reads every
           field; nothing is mutated during a build.
    """

    # --- Provider selectors ---
    # Which kind to instantiate via `make_llm` / `make_embedding`. ``"fake"``
    # keeps the build offline and deterministic — used by tests and local dev.
    enricher_llm_kind: str = "fake"
    labeler_llm_kind: str = "fake"
    judge_llm_kind: str = "fake"
    embedding_kind: str = "fake"

    # Output dimensionality requested from the embedder. Must match the
    # real model's dim when using a hosted embedder (see `embedding_openai.py`).
    embedding_dim: int = 64

    # Per-level ``(min, max)`` children count. Enforced by rebalance and
    # validated by the structural check. Keys match the level names used
    # throughout the pipeline (``domain/category/group/tool``).
    fanout: dict[str, tuple[int, int]] = field(default_factory=lambda: {
        "domain": (2, 20),
        "category": (2, 10),
        "group": (2, 10),
        "tool": (3, 8),
    })

    # Numeric knobs that tune clustering and validation:
    #   near_dup          — cosine similarity above which two tools are
    #                       treated as duplicates (stage 1)
    #   group/category/
    #     domain          — agglomerative distance cutoffs per level
    #   discriminability  — sibling description score below which we warn
    #   min_recall        — hard floor for recall@k; below this, build fails
    thresholds: dict[str, float] = field(default_factory=lambda: {
        "near_dup": 0.97,
        "group": 0.30,
        "category": 0.45,
        "domain": 0.70,
        "discriminability": 0.4,
        "min_recall": 0.6,
    })

    # Outer chunk size for stage 2 — unit of work between cache flushes.
    enrich_batch_size: int = 20
    # Synthetic queries per tool for the recall benchmark. Higher is more
    # robust but linearly more expensive at eval time.
    synthetic_queries_per_tool: int = 5
    # Top-k depth for recall@k measurement.
    recall_k: int = 30
    # Disk cache directory — provider caches live under this path.
    cache_dir: str = "data/cache"

    # --- Instantiated providers, filled in by `build_providers()` ---
    # Typed as `Any` because Protocol types don't play well with dataclass
    # defaults; the actual types are `LLMProvider` / `EmbeddingProvider`.
    enricher_llm: LLMProvider = None
    labeler_llm: LLMProvider = None
    judge_llm: LLMProvider = None
    embedder: EmbeddingProvider = None
    cache: DiskCache = None

    def build_providers(self) -> None:
        """Instantiate the four providers + cache from their ``_kind`` selectors.

        Called once at the end of `load_config` / `default_config`. Overwrites
        any previously-assigned provider instances, so callers that want to
        wire in custom providers (e.g. the verification script using the
        Anthropic proxy) must skip this and assign directly.
        """
        self.enricher_llm = make_llm(self.enricher_llm_kind)
        self.labeler_llm = make_llm(self.labeler_llm_kind)
        self.judge_llm = make_llm(self.judge_llm_kind)
        self.embedder = make_embedding(self.embedding_kind, dim=self.embedding_dim)
        self.cache = DiskCache(self.cache_dir)


def load_config(path: str | Path) -> Config:
    """Parse a YAML file into a fully-built `Config`.

    Unknown keys are silently ignored so configs can carry forward-compat
    metadata. Missing keys fall back to the `Config` dataclass defaults —
    every field is optional in the YAML.
    """
    data = yaml.safe_load(Path(path).read_text()) or {}
    c = Config()
    # String knobs — tolerate missing keys via ``.get`` with the current
    # default as fallback.
    c.enricher_llm_kind = data.get("enricher_llm", c.enricher_llm_kind)
    c.labeler_llm_kind = data.get("labeler_llm", c.labeler_llm_kind)
    c.judge_llm_kind = data.get("judge_llm", c.judge_llm_kind)
    c.embedding_kind = data.get("embedding_model", c.embedding_kind)
    c.embedding_dim = int(data.get("embedding_dim", c.embedding_dim))
    # YAML represents tuples as lists — convert back so `fanout` matches
    # the annotated type and `min, max = cfg.fanout[k]` keeps working.
    if "fanout" in data:
        c.fanout = {k: tuple(v) for k, v in data["fanout"].items()}
    # Thresholds merge with defaults so partial overrides are legal.
    if "thresholds" in data:
        c.thresholds = {**c.thresholds, **{k: float(v) for k, v in data["thresholds"].items()}}
    c.enrich_batch_size = int(data.get("enrich_batch_size", c.enrich_batch_size))
    c.synthetic_queries_per_tool = int(data.get("synthetic_queries_per_tool", c.synthetic_queries_per_tool))
    c.recall_k = int(data.get("recall_k", c.recall_k))
    c.cache_dir = data.get("cache_dir", c.cache_dir)
    c.build_providers()
    return c


def default_config() -> Config:
    """Return a `Config` with all defaults and providers instantiated.

    Used when the CLI is invoked without ``--config``; also handy for tests.
    Everything defaults to fakes, so this is always offline.
    """
    c = Config()
    c.build_providers()
    return c
