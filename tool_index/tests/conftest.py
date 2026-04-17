from __future__ import annotations
import json
from pathlib import Path

import pytest

from tool_index.config import default_config
from tool_index.utils.ids import reset_id_counter


@pytest.fixture(autouse=True)
def _reset_ids():
    reset_id_counter()


@pytest.fixture
def config(tmp_path):
    cfg = default_config()
    cfg.cache_dir = str(tmp_path / "cache")
    cfg.build_providers()
    return cfg


@pytest.fixture
def mini_tools() -> list[dict]:
    p = Path(__file__).parent / "fixtures" / "mini_tools.json"
    return json.loads(p.read_text())
