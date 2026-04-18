from __future__ import annotations

import json
from pathlib import Path

import pytest

from tool_index.router import (
    CustomerLayout,
    PromotionResult,
    QualityScore,
    promote_if_better,
)


def _write_quality(layout: CustomerLayout, version: str, score: float) -> None:
    layout.ensure()
    layout.quality_path(version).write_text(
        json.dumps({"score": score, "sample_count": 10, "hits": int(score * 10), "k": 10})
    )


def test_first_snapshot_always_promotes(tmp_path: Path) -> None:
    layout = CustomerLayout.for_customer(tmp_path, "acme")
    q = QualityScore(score=0.5, sample_count=10, hits=5, k=10)
    r = promote_if_better(layout, "v0", q)
    assert r.promoted
    assert layout.read_active() == "v0"
    assert "first snapshot" in r.reason


def test_promote_when_score_improves(tmp_path: Path) -> None:
    layout = CustomerLayout.for_customer(tmp_path, "acme")
    layout.write_active("v0")
    _write_quality(layout, "v0", 0.7)
    q = QualityScore(score=0.8, sample_count=10, hits=8, k=10)
    r = promote_if_better(layout, "v1", q)
    assert r.promoted
    assert layout.read_active() == "v1"


def test_promote_within_epsilon(tmp_path: Path) -> None:
    layout = CustomerLayout.for_customer(tmp_path, "acme")
    layout.write_active("v0")
    _write_quality(layout, "v0", 0.80)
    q = QualityScore(score=0.79, sample_count=10, hits=8, k=10)
    r = promote_if_better(layout, "v1", q, epsilon=0.02)
    assert r.promoted, r.reason
    assert layout.read_active() == "v1"


def test_rollback_when_score_drops(tmp_path: Path) -> None:
    layout = CustomerLayout.for_customer(tmp_path, "acme")
    layout.write_active("v0")
    _write_quality(layout, "v0", 0.90)
    q = QualityScore(score=0.50, sample_count=10, hits=5, k=10)
    r = promote_if_better(layout, "v1", q, epsilon=0.02)
    assert not r.promoted
    assert layout.read_active() == "v0"
    assert "rolled back" in r.reason
    assert layout.quality_path("v1").exists()


def test_promote_when_previous_score_missing(tmp_path: Path) -> None:
    layout = CustomerLayout.for_customer(tmp_path, "acme")
    layout.write_active("v0")
    q = QualityScore(score=0.4, sample_count=10, hits=4, k=10)
    r = promote_if_better(layout, "v1", q)
    assert r.promoted
    assert layout.read_active() == "v1"
