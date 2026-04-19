from __future__ import annotations

import yaml

from tool_index.config.loader import Config, default_config, load_config


def test_default_single_labeler_used_for_every_level() -> None:
    cfg = default_config()
    assert cfg.labeler_llm_per_level is None
    for level in ("domain", "category", "group"):
        assert cfg.select_labeler_llm(level) is cfg.labeler_llm
    # Unknown/None falls back to base labeler.
    assert cfg.select_labeler_llm(None) is cfg.labeler_llm
    assert cfg.select_labeler_llm("anything") is cfg.labeler_llm


def test_per_level_dispatch_uses_distinct_providers() -> None:
    cfg = Config()
    cfg.labeler_llm_kind_per_level = {
        "domain": "fake",
        "category": "fake",
        "group": "fake",
    }
    cfg.build_providers()
    assert cfg.labeler_llm_per_level is not None
    d = cfg.select_labeler_llm("domain")
    g = cfg.select_labeler_llm("group")
    # Each level should have its own provider instance.
    assert d is cfg.labeler_llm_per_level["domain"]
    assert g is cfg.labeler_llm_per_level["group"]
    assert d is not g


def test_missing_level_falls_back_to_base_labeler() -> None:
    cfg = Config()
    cfg.labeler_llm_kind_per_level = {"domain": "fake"}
    cfg.build_providers()
    assert cfg.select_labeler_llm("domain") is cfg.labeler_llm_per_level["domain"]
    # Category is not in the per-level map — falls back to base.
    assert cfg.select_labeler_llm("category") is cfg.labeler_llm


def test_yaml_config_propagates_per_level(tmp_path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({
        "labeler_llm": "fake",
        "labeler_llm_per_level": {"domain": "fake", "group": "fake"},
    }))
    cfg = load_config(p)
    assert cfg.labeler_llm_kind_per_level == {"domain": "fake", "group": "fake"}
    assert cfg.labeler_llm_per_level is not None
    assert "domain" in cfg.labeler_llm_per_level


def test_yaml_without_per_level_keeps_single_provider() -> None:
    import pathlib, tempfile
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "c.yaml"
        p.write_text(yaml.safe_dump({"labeler_llm": "fake"}))
        cfg = load_config(p)
    assert cfg.labeler_llm_per_level is None
    assert cfg.select_labeler_llm("domain") is cfg.labeler_llm
