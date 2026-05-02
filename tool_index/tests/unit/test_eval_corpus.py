from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def _load_eval_queries():
    repo_root = Path(__file__).parents[2]
    corpus_root = repo_root / "data/corpus"

    catalog_spec = importlib.util.spec_from_file_location("corpus.catalog", corpus_root / "catalog.py")
    catalog_mod = importlib.util.module_from_spec(catalog_spec)
    corpus_pkg = types.ModuleType("corpus")
    corpus_pkg.__path__ = [str(corpus_root)]
    corpus_pkg.catalog = catalog_mod
    sys.modules["corpus"] = corpus_pkg
    sys.modules["corpus.catalog"] = catalog_mod
    catalog_spec.loader.exec_module(catalog_mod)

    eval_spec = importlib.util.spec_from_file_location("corpus.eval_queries", corpus_root / "eval_queries.py")
    eval_mod = importlib.util.module_from_spec(eval_spec)
    sys.modules["corpus.eval_queries"] = eval_mod
    eval_spec.loader.exec_module(eval_mod)
    return catalog_mod, eval_mod


def test_ultra_complex_corpus_has_50_cases_with_at_least_4_calls():
    catalog, eval_queries = _load_eval_queries()
    active_tool_names = {
        tool["name"] if isinstance(tool, dict) else tool.name
        for tool in catalog.raw_tools
    }

    cases = eval_queries.ULTRA_COMPLEX_CASES

    assert len(cases) == 50
    assert all(len(case["calls"]) >= 4 for case in cases)
    assert all(call in active_tool_names for case in cases for call in case["calls"])
