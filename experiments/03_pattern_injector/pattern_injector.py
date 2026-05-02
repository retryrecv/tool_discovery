"""Deterministic glue-tool sub-query injector.

Run AFTER the production LLM decomposer. Scans the original user query
with a fixed set of regex rules; for every match, appends the matching
glue-tool's enrichment `intent_phrase` to the sub-query list.

Append-only and dedupe-by-casefold — never replaces an LLM sub-query,
never duplicates an intent the LLM already named.

See `hypothesis.md` for the rule table and rationale.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Rule:
    name: str          # tool name (for diagnostics)
    intent: str        # intent_phrase to append
    pattern: re.Pattern
    requires: tuple[re.Pattern, ...] = ()  # ALL must also match


_I = re.IGNORECASE


def _p(s: str) -> re.Pattern:
    return re.compile(s, _I)


# Order matters only for diagnostics — all rules are evaluated.
RULES: tuple[Rule, ...] = (
    Rule(
        "calculator",
        "evaluate arithmetic expressions",
        _p(r"\b(calculate|compute|how many|total|sum|average|mean|times|multiplied|divided|share of|how much)\b"),
    ),
    Rule(
        "unit_converter",
        "convert values between common measurement units",
        _p(r"\b(convert|in)\b"),
        requires=(
            _p(r"\b\d+\s*(km|kilometers?|miles?|kg|kilograms?|lbs?|pounds?|°[cf]|celsius|fahrenheit|mph|km/?h|meters?|feet|inches?)\b"),
        ),
    ),
    Rule(
        "unit_converter",
        "convert values between common measurement units",
        _p(r"\b\d+\s*(km/?h|mph)\b"),
    ),
    Rule(
        "get_current_datetime",
        "get the current UTC date and time",
        _p(r"\b(now|today|current (date|time)|right now|what time is it|tonight|tomorrow|yesterday)\b"),
    ),
    Rule(
        "day_of_week",
        "get weekday from a date",
        _p(r"\b(day of (the )?week|what day (is|was)|which weekday)\b"),
    ),
    Rule(
        "timezone_convert",
        "convert a datetime between timezones",
        _p(r"\b(time ?zone|in (PST|EST|UTC|GMT|JST|CET|PDT|EDT))\b|\bconvert .* (time|datetime) to .*\b"),
    ),
    Rule(
        "date_diff",
        "compute time difference between two dates",
        _p(r"\b(days? (until|till|since|between)|how long until|how many days (until|since|between))\b"),
    ),
    Rule(
        "url_parse",
        "parse a URL into components",
        _p(r"https?://|\bwww\.|\bURL\b"),
    ),
    Rule(
        "http_get",
        "fetch content from a URL",
        _p(r"\b(fetch|download|get|grab|pull|hit|call)\b"),
        requires=(_p(r"https?://|\bwww\.|\bURL\b|\bendpoint\b|\bAPI\b"),),
    ),
    Rule(
        "json_query",
        "extract values from JSON by path",
        _p(r"\b(extract|pull|pluck|get|read|find)\b"),
        requires=(_p(r"\bJSON\b|\bjson\b"),),
    ),
    Rule(
        "json_format",
        "format JSON",
        _p(r"\b(pretty[- ]?print|format|readable|show me|display)\b"),
        requires=(_p(r"\bJSON\b|\bjson\b"),),
    ),
    Rule(
        "array_sort",
        "sort an array",
        _p(r"\b(sort|order)\b.*\b(by|ascending|descending|alphabetical|asc|desc)\b"),
    ),
    Rule(
        "statistics",
        "compute basic descriptive statistics",
        _p(r"\b(median|stdev|standard deviation|variance|min(imum)?|max(imum)?)\b"),
    ),
    Rule(
        "percentage_calc",
        "calculate percentages",
        _p(r"\b(percent|percentage|%)\b.*\b(of|change|increase|decrease|growth|drop)\b|\b(percent|percentage)\b"),
    ),
    Rule(
        "encode_decode",
        "encode or decode text strings",
        _p(r"\b(base64|url[- ]?encode|url[- ]?decode|hex encode|hex decode|encode this|decode this)\b"),
    ),
    Rule(
        "generate_uuid",
        "generate a random UUID",
        _p(r"\b(uuid|guid|unique id|unique identifier|new id)\b"),
    ),
    Rule(
        "hash_text",
        "compute a SHA-256 hash of text",
        _p(r"\b(sha256|sha-256|hash|fingerprint|checksum|digest)\b"),
    ),
)


def _matches(rule: Rule, query: str) -> bool:
    if not rule.pattern.search(query):
        return False
    return all(req.search(query) for req in rule.requires)


def inject(query: str, sub_queries: Iterable[str]) -> tuple[list[str], list[str]]:
    """Return (augmented_sub_queries, fired_rule_names).

    Append-only. Dedupes against case-folded existing sub-queries so
    intents the LLM already named aren't appended again.
    """
    out = list(sub_queries)
    seen = {sq.casefold().strip() for sq in out}
    fired: list[str] = []
    for rule in RULES:
        if not _matches(rule, query):
            continue
        key = rule.intent.casefold().strip()
        if key in seen:
            continue
        out.append(rule.intent)
        seen.add(key)
        fired.append(rule.name)
    return out, fired
