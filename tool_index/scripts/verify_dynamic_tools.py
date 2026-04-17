"""Verify `build_tree_index` works on dynamically-created tool defs using
the local Agent Maestro Anthropic proxy.

Mirrors the TypeScript pattern:

    new Anthropic({
      baseURL: "http://localhost:23333/api/anthropic",
      apiKey: "***REDACTED***",
    })

The 5 in-memory tools below are the "dynamic" input — nothing on disk,
nothing in a fixture file. The script builds a full snapshot end-to-end
and asserts basic post-conditions on the resulting tree.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from tool_index.config import Config
from tool_index.pipeline import build_tree_index
from tool_index.providers import DiskCache, make_embedding, EmbeddingProvider
from tool_index.providers.llm_anthropic import AnthropicLLMProvider


# --- Proxy wiring (mirrors the TS snippet) ---
# Base URL of the local Agent Maestro Anthropic-compatible endpoint.
PROXY_BASE_URL = "http://localhost:23333/api/anthropic"
# Placeholder key — the proxy accepts this literally.
PROXY_API_KEY = "***REDACTED***"
# Small/fast Claude model; enrichment and labeling don't need a frontier model.
MODEL = "claude-haiku-4-5-20251001"


def _pick_embedder() -> tuple[EmbeddingProvider, str]:
    """Prefer Azure OpenAI when the env is configured, fall back to fake.

    Returns a tuple ``(embedder, description)`` where ``description`` is
    a one-line human-readable label for the chosen embedder — surfaced
    in the script's banner so it's obvious which mode ran.
    """
    if os.environ.get("AZURE_OPENAI_ENDPOINT") and os.environ.get("AZURE_OPENAI_API_KEY"):
        # text-embedding-3-large outputs 3072 dims; keep this in sync if
        # you point AZURE_EMBEDDINGS_DEPLOYMENT_NAME at a different model.
        emb = make_embedding("azure_openai", dim=3072)
        return emb, f"Azure OpenAI embeddings (deployment={emb.model}, dim={emb.dim})"
    return make_embedding("fake", dim=64), "Fake hash-BOW embeddings (64-dim)"


def make_five_tools() -> list[dict]:
    """Return 5 hand-written tool dicts covering distinct domains.

    Chosen to be semantically varied so clustering has something to work
    with — if we used 5 near-duplicates, stage 1 would dedupe them down
    to one and the rest of the pipeline wouldn't exercise.
    """
    return [
        {"name": "calendar_create_event",
         "signature": "calendar_create_event(title, start, end) -> id",
         "doc": "create a new event on the user's calendar with title and time range"},
        {"name": "calendar_list_events",
         "signature": "calendar_list_events(date) -> events",
         "doc": "list all calendar events for a given date"},
        {"name": "email_send",
         "signature": "email_send(to, subject, body) -> ok",
         "doc": "send an email message to a recipient with subject and body"},
        {"name": "weather_forecast",
         "signature": "weather_forecast(city, days) -> forecast",
         "doc": "fetch a multi-day weather forecast for a named city"},
        {"name": "translate_text",
         "signature": "translate_text(text, target_lang) -> text",
         "doc": "translate a text string into the target language"},
    ]


def build_config() -> Config:
    """Wire a `Config` that uses the Anthropic proxy for LLM + best-available embedder.

    Skips `Config.build_providers()` (which would instantiate all fakes)
    and assigns providers directly. Embeddings prefer Azure OpenAI when
    the env is set; otherwise fall back to the deterministic fake so the
    script stays runnable as a smoke test.
    """
    c = Config()
    llm = AnthropicLLMProvider(model=MODEL, base_url=PROXY_BASE_URL, api_key=PROXY_API_KEY)
    # Same LLM for all three roles — the proxy only exposes one model.
    c.enricher_llm = llm
    c.labeler_llm = llm
    c.judge_llm = llm
    embedder, embedder_desc = _pick_embedder()
    c.embedder = embedder
    c._embedder_desc = embedder_desc  # surfaced in the banner
    c.cache = DiskCache("data/cache")
    # With only 5 tools, the recall floor is unrealistic — relax it so
    # stage 5 reports recall without tripping strict mode.
    c.thresholds["min_recall"] = 0.0
    # Fewer queries per tool keeps the proxy call count low.
    c.synthetic_queries_per_tool = 2
    c.recall_k = 5
    return c


def main() -> int:
    """Run the verification and return process exit code."""
    tools = make_five_tools()
    print(f"Input: {len(tools)} dynamically-created tools")
    for t in tools:
        print(f"  - {t['name']}")

    config = build_config()
    print(f"\nLLM: Anthropic via {PROXY_BASE_URL} (model={MODEL})")
    print(f"Embedder: {getattr(config, '_embedder_desc', 'unknown')}\n")

    out_dir = Path("data/snapshots/verify-dynamic")
    tree = build_tree_index(tools, config, out_root=out_dir, strict=False)

    leaves = list(tree.tools_by_id.keys())
    print("\n--- Result ---")
    print(f"Snapshot version : {tree.version}")
    print(f"Leaves (tools)   : {len(leaves)}")
    print(f"Tree depth       : {tree.depth()}")
    print(f"Total nodes      : {len(tree.nodes_by_id)}")

    # Post-conditions — if any fail, the script exits non-zero via
    # AssertionError, which distinguishes "build ran but didn't produce
    # what we expected" from "build errored out".
    assert len(leaves) == 5, f"expected 5 leaves, got {len(leaves)}"
    assert tree.depth() >= 2, f"expected tree depth >= 2, got {tree.depth()}"

    snapshot_path = out_dir / tree.version / "tree.json"
    assert snapshot_path.exists(), f"snapshot not written: {snapshot_path}"
    data = json.loads(snapshot_path.read_text())
    print(f"Snapshot written : {snapshot_path} ({len(json.dumps(data))} bytes)")

    print("\nPASS: pipeline handled 5 dynamically-created tools via the local proxy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
