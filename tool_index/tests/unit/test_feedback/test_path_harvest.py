from __future__ import annotations

from tool_index.feedback import (
    FeedbackLabel,
    FeedbackRecord,
    PathHarvestConfig,
    Polarity,
    harvest_path_labels,
)


def _rec(req_id: str, polarity: Polarity, conf: float = 0.5) -> FeedbackRecord:
    return FeedbackRecord(
        customer_id="c1",
        query="q",
        routed_tool_id="tool_a",
        snapshot_version="v1",
        label=FeedbackLabel(polarity=polarity, confidence=conf, reason="t"),
        request_id=req_id,
        session_id="s1",
        timestamp="2026-04-01T00:00:00Z",
    )


def test_positive_leaf_expands_to_path_depth() -> None:
    recs = [_rec("r1", Polarity.POSITIVE, conf=0.3)]
    paths = {"r1": ["n_l1", "n_l2", "n_l3"]}
    out = harvest_path_labels(recs, paths)
    assert len(out) == 4  # 1 original + 3 per-level
    levels = [r.extra.get("level") for r in out if r.extra.get("harvested")]
    assert levels == [1, 2, 3]


def test_graduated_confidence_floor() -> None:
    recs = [_rec("r1", Polarity.POSITIVE, conf=0.1)]
    paths = {"r1": ["n_l1", "n_l2", "n_l3"]}
    out = harvest_path_labels(
        recs,
        paths,
        PathHarvestConfig(level_floor=(0.4, 0.6, 0.8)),
    )
    harvested = [r for r in out if r.extra.get("harvested")]
    assert harvested[0].label.confidence == 0.4
    assert harvested[1].label.confidence == 0.6
    assert harvested[2].label.confidence == 0.8


def test_non_positive_passthrough() -> None:
    recs = [_rec("r1", Polarity.NEGATIVE), _rec("r2", Polarity.UNKNOWN)]
    paths = {"r1": ["a", "b"], "r2": ["a"]}
    out = harvest_path_labels(recs, paths)
    assert len(out) == 2
    assert all(not r.extra.get("harvested") for r in out)


def test_disabled_returns_records_unchanged() -> None:
    recs = [_rec("r1", Polarity.POSITIVE)]
    paths = {"r1": ["a", "b"]}
    out = harvest_path_labels(recs, paths, PathHarvestConfig(enabled=False))
    assert out == recs


def test_preserves_original_confidence_when_above_floor() -> None:
    recs = [_rec("r1", Polarity.POSITIVE, conf=0.95)]
    paths = {"r1": ["n1"]}
    out = harvest_path_labels(recs, paths)
    harvested = [r for r in out if r.extra.get("harvested")]
    assert harvested[0].label.confidence == 0.95
