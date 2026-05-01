"""Shared config + paths for stage scripts.

Importable from any `scripts/stage_*.py`. One source of truth for the
proxy URL, model name, embedder, cache, and per-run snapshot directory.
"""
from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)

_spec = importlib.util.spec_from_file_location("tools", ROOT / "data/generateTools/tools.py")
_tools_mod = importlib.util.module_from_spec(_spec)
_pkg = types.ModuleType("generateTools")
_pkg.tools = _tools_mod
sys.modules["generateTools"] = _pkg
sys.modules["generateTools.tools"] = _tools_mod
_spec.loader.exec_module(_tools_mod)

raw_tools = _tools_mod.raw_tools

from tool_index.config import Config
from tool_index.providers import DiskCache, make_embedding
from tool_index.providers.llm_anthropic import AnthropicLLMProvider

PROXY_BASE_URL = "http://localhost:23333/api/anthropic"
PROXY_API_KEY = "***REDACTED***"
MODEL = "claude-haiku-4-5-20251001"


def make_config() -> Config:
    c = Config()
    llm = AnthropicLLMProvider(model=MODEL, base_url=PROXY_BASE_URL, api_key=PROXY_API_KEY)
    c.enricher_llm = c.labeler_llm = llm
    c.embedder = make_embedding("azure_openai", dim=3072)
    c.cache = DiskCache("data/cache")
    c.thresholds["group"] = 0.45
    c.thresholds["category"] = 0.65
    c.thresholds["domain"] = 0.85
    c.fanout["tool"] = (1, 8)
    c.fanout["group"] = (1, 10)
    c.fanout["category"] = (1, 10)
    c.fanout["domain"] = (1, 20)
    c.recall_k = 10
    return c


def run_dir(run: str) -> Path:
    p = ROOT / "data/snapshots" / run
    p.mkdir(parents=True, exist_ok=True)
    return p
