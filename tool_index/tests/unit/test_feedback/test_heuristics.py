from __future__ import annotations

from tool_index.feedback import label_session, Polarity
from tool_index.feedback.sessionize import Session


def _r(ts, query, tool="t1"):
    return {
        "customer_id": "acme",
        "timestamp": ts,
        "request_id": ts,
        "query": query,
        "routed_tool_id": tool,
        "snapshot_version": "v0",
    }


def test_retry_marks_first_negative() -> None:
    sess = Session("acme", "s", [
        _r("2026-04-17T10:00:00Z", "list files in directory"),
        _r("2026-04-17T10:00:30Z", "list all files in directory"),
    ])
    labels = label_session(sess)
    assert labels[0].label.polarity == Polarity.NEGATIVE
    assert "retry" in labels[0].label.reason


def test_followup_tool_switch_marks_negative() -> None:
    sess = Session("acme", "s", [
        _r("2026-04-17T10:00:00Z", "find python files", tool="t1"),
        _r("2026-04-17T10:02:00Z", "find python files", tool="t2"),
    ])
    labels = label_session(sess)
    assert labels[0].label.polarity == Polarity.NEGATIVE
    assert "switched tool" in labels[0].label.reason


def test_no_followup_is_unknown() -> None:
    sess = Session("acme", "s", [_r("2026-04-17T10:00:00Z", "q")])
    labels = label_session(sess)
    assert labels[0].label.polarity == Polarity.UNKNOWN


def test_unrelated_followup_is_presumed_positive() -> None:
    sess = Session("acme", "s", [
        _r("2026-04-17T10:00:00Z", "list files", tool="t1"),
        _r("2026-04-17T10:01:00Z", "convert pdf to text", tool="t2"),
    ])
    labels = label_session(sess)
    assert labels[0].label.polarity == Polarity.POSITIVE
