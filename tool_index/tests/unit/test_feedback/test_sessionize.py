from __future__ import annotations

from tool_index.feedback import sessionize


def _r(customer, ts, sid=None, query="q", tool="t1"):
    return {
        "customer_id": customer,
        "timestamp": ts,
        "session_id": sid,
        "request_id": ts,
        "query": query,
        "routed_tool_id": tool,
        "snapshot_version": "v0",
    }


def test_groups_by_explicit_session_id() -> None:
    recs = [
        _r("acme", "2026-04-17T10:00:00Z", sid="s1"),
        _r("acme", "2026-04-17T10:00:30Z", sid="s2"),
        _r("acme", "2026-04-17T10:01:00Z", sid="s1"),
    ]
    sessions = list(sessionize(recs))
    by_sid = {s.session_id: s for s in sessions}
    assert len(by_sid["s1"]) == 2
    assert len(by_sid["s2"]) == 1


def test_implicit_groups_by_time_gap() -> None:
    recs = [
        _r("acme", "2026-04-17T10:00:00Z"),
        _r("acme", "2026-04-17T10:02:00Z"),
        _r("acme", "2026-04-17T10:30:00Z"),
    ]
    sessions = [s for s in sessionize(recs, gap_seconds=300.0) if s.session_id is None]
    assert [len(s) for s in sessions] == [2, 1]


def test_separates_customers() -> None:
    recs = [
        _r("acme", "2026-04-17T10:00:00Z"),
        _r("globex", "2026-04-17T10:00:30Z"),
    ]
    sessions = list(sessionize(recs))
    customers = {s.customer_id for s in sessions}
    assert customers == {"acme", "globex"}
