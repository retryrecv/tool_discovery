from tool_index.pipeline.stage1_normalize import normalize_and_dedupe
from tool_index.providers import FakeEmbeddingProvider


def test_exact_dedupe():
    emb = FakeEmbeddingProvider(dim=32)
    raw = [
        {"name": "a", "signature": "a()", "doc": "do a"},
        {"name": "a", "signature": "a()", "doc": "do a"},
        {"name": "b", "signature": "b()", "doc": "do b"},
    ]
    out = normalize_and_dedupe(raw, emb, near_dup_threshold=0.999)
    assert {d.name for d in out} == {"a", "b"}


def test_side_effect_inference():
    emb = FakeEmbeddingProvider(dim=32)
    raw = [
        {"name": "get_user", "signature": "", "doc": "read a user record"},
        {"name": "delete_user", "signature": "", "doc": "remove a user record"},
    ]
    out = normalize_and_dedupe(raw, emb, near_dup_threshold=0.999)
    mp = {d.name: d.side_effect_class for d in out}
    assert mp["get_user"] == "read"
    assert mp["delete_user"] == "write"


def test_source_and_examples_parse():
    emb = FakeEmbeddingProvider(dim=32)
    out = normalize_and_dedupe(
        [{
            "name": "lookup",
            "signature": "lookup()",
            "doc": "find a record",
            "source": "catalog.json",
            "examples": [{"args": {"id": 1}, "returns": {"id": 1}}],
        }],
        emb,
        near_dup_threshold=0.999,
    )
    assert out[0].source == "catalog.json"
    assert out[0].example_calls == [{"args": {"id": 1}, "returns": {"id": 1}}]


def test_empty_doc_is_allowed():
    emb = FakeEmbeddingProvider(dim=32)
    out = normalize_and_dedupe(
        [{"name": "lookup", "signature": "lookup()", "doc": ""}],
        emb,
        near_dup_threshold=0.999,
    )
    assert out[0].original_doc == ""
