from __future__ import annotations

from pathlib import Path

from tool_index.scale import ShardedDiskCache


def test_round_trip(tmp_path: Path) -> None:
    c = ShardedDiskCache(tmp_path)
    assert c.get("model-x", "prompt-a") is None
    c.put("model-x", "prompt-a", {"hello": 1})
    assert c.get("model-x", "prompt-a") == {"hello": 1}


def test_shards_distribute(tmp_path: Path) -> None:
    c = ShardedDiskCache(tmp_path)
    for i in range(64):
        c.put("m", f"prompt-{i}", i)
    model_dir = tmp_path / "m"
    shard_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
    assert len(shard_dirs) > 1


def test_invalid_shard_chars_rejected(tmp_path: Path) -> None:
    import pytest
    with pytest.raises(ValueError):
        ShardedDiskCache(tmp_path, shard_chars=0)
    with pytest.raises(ValueError):
        ShardedDiskCache(tmp_path, shard_chars=5)
