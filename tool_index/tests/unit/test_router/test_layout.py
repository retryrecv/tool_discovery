from __future__ import annotations

from pathlib import Path

from tool_index.router import CustomerLayout


def test_active_pointer_round_trip(tmp_path: Path) -> None:
    layout = CustomerLayout.for_customer(tmp_path, "acme")
    assert layout.read_active() is None
    layout.write_active("v3")
    assert layout.read_active() == "v3"


def test_list_versions_sorted(tmp_path: Path) -> None:
    layout = CustomerLayout.for_customer(tmp_path, "acme")
    layout.ensure()
    for v in ["v2", "v0", "v10", "v1"]:
        (layout.root / v).mkdir()
    (layout.root / "scratch").mkdir()
    assert layout.list_versions() == ["v0", "v1", "v2", "v10"]
