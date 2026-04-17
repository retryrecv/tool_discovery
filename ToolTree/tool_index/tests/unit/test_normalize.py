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
