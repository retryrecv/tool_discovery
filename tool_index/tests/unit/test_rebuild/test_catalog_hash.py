from __future__ import annotations

from tool_index.rebuild import compute_catalog_hash


def test_order_independent() -> None:
    a = [{"name": "t1", "signature": "s1", "doc": "d1"},
         {"name": "t2", "signature": "s2", "doc": "d2"}]
    assert compute_catalog_hash(a) == compute_catalog_hash(list(reversed(a)))


def test_changes_with_doc_edit() -> None:
    a = [{"name": "t1", "signature": "s1", "doc": "old"}]
    b = [{"name": "t1", "signature": "s1", "doc": "new"}]
    assert compute_catalog_hash(a) != compute_catalog_hash(b)


def test_examples_normalized() -> None:
    a = [{"name": "t1", "signature": "s1", "doc": "d", "examples": [{"args": {"a": 1}}]}]
    b = [{"name": "t1", "signature": "s1", "doc": "d", "examples": [{"args": {"a": 1}}]}]
    assert compute_catalog_hash(a) == compute_catalog_hash(b)
