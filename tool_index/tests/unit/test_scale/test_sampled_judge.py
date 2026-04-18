from __future__ import annotations

from dataclasses import dataclass

from tool_index.scale import sample_pairs, SampleStrategy


@dataclass
class _Sib:
    id: str
    embedding: list[float]


def test_returns_at_most_max_pairs() -> None:
    sibs = [_Sib(f"s{i}", [float(i), 0.0]) for i in range(10)]
    pairs = sample_pairs(sibs, strategy=SampleStrategy(max_pairs=5))
    assert len(pairs) == 5


def test_closest_pairs_first() -> None:
    sibs = [
        _Sib("a", [1.0, 0.0]),
        _Sib("b", [0.99, 0.01]),
        _Sib("c", [0.0, 1.0]),
    ]
    pairs = sample_pairs(sibs, strategy=SampleStrategy(max_pairs=1, closest_first=True))
    ids = {pairs[0][0].id, pairs[0][1].id}
    assert ids == {"a", "b"}


def test_below_max_returns_all() -> None:
    sibs = [_Sib(f"s{i}", [float(i)]) for i in range(3)]
    pairs = sample_pairs(sibs, strategy=SampleStrategy(max_pairs=10))
    assert len(pairs) == 3


def test_singleton_returns_empty() -> None:
    assert sample_pairs([_Sib("only", [1.0])]) == []


def test_deterministic() -> None:
    sibs = [_Sib(f"s{i}", [float(i % 3), float(i)]) for i in range(8)]
    a = sample_pairs(sibs, strategy=SampleStrategy(max_pairs=4))
    b = sample_pairs(sibs, strategy=SampleStrategy(max_pairs=4))
    assert [(p[0].id, p[1].id) for p in a] == [(p[0].id, p[1].id) for p in b]
