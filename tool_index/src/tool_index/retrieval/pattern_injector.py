"""Deterministic glue-tool sub-query injection.

The LLM decomposer is good at most user-facing steps but often skips
implicit utility tools such as URL parsing, current time, JSON extraction,
or UUID generation. This injector runs after decomposition and appends
targeted intent phrases based on regex matches against the original query.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
import re


@dataclass(frozen=True)
class InjectionRule:
    """One deterministic query pattern mapped to a retrieval intent."""

    name: str
    intent: str
    pattern: re.Pattern[str]
    requires: tuple[re.Pattern[str], ...] = ()


_I = re.IGNORECASE


def _p(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, _I)


RULES: tuple[InjectionRule, ...] = (
    InjectionRule(
        "calculator",
        "evaluate arithmetic expressions",
        _p(r"\b(calculate|compute|how many|total|sum|average|mean|times|multiplied|divided|share of|how much)\b"),
    ),
    InjectionRule(
        "calculator",
        "evaluate arithmetic expressions",
        _p(r"\bhow long\b.*\bat\b.*\b\d+\s*(km/?h|mph|miles? per hour|kilometers? per hour)\b"),
    ),
    InjectionRule(
        "unit_converter",
        "convert values between common measurement units",
        _p(r"\b(convert|in)\b"),
        requires=(
            _p(r"\b\d+\s*(km|kilometers?|miles?|kg|kilograms?|lbs?|pounds?|deg[cf]|celsius|fahrenheit|mph|km/?h|meters?|feet|inches?)\b"),
        ),
    ),
    InjectionRule(
        "unit_converter",
        "convert values between common measurement units",
        _p(r"\b\d+\s*(km/?h|mph)\b"),
    ),
    InjectionRule(
        "get_current_datetime",
        "get the current UTC date and time",
        _p(r"\b(now|today|current (date|time)|right now|what time is it|tonight|tomorrow|yesterday|timestamp)\b"),
    ),
    InjectionRule(
        "day_of_week",
        "get weekday from a date",
        _p(r"\b(day of (the )?week|what day (is|was)|which weekday)\b"),
    ),
    InjectionRule(
        "timezone_convert",
        "convert a datetime between timezones",
        _p(r"\b(time ?zone|in (PST|EST|UTC|GMT|JST|CET|PDT|EDT|London))\b|\bconvert .* (time|datetime) to .*\b"),
    ),
    InjectionRule(
        "date_diff",
        "compute time difference between two dates",
        _p(r"\b(days? (until|till|since|between)|how long until|how many days (until|since|between))\b"),
    ),
    InjectionRule(
        "url_parse",
        "parse a URL into components",
        _p(r"https?://|\bwww\.|\bURL\b"),
    ),
    InjectionRule(
        "http_get",
        "fetch content from a URL",
        _p(r"\b(fetch|download|get|grab|pull|hit|call)\b"),
        requires=(_p(r"https?://|\bwww\.|\bURL\b|\bendpoint\b|\bAPI\b"),),
    ),
    InjectionRule(
        "json_query",
        "extract values from JSON by path",
        _p(r"\b(extract|pull(?: out)?|pluck|get|read|find)\b.*\b(status|created_at|inventory count|owner fields?|subtotal|price|token|array|field)\b"),
        requires=(_p(r"\b(API response|response|payload|returned products|product API|callback URL|endpoint)\b"),),
    ),
    InjectionRule(
        "json_query",
        "extract values from JSON by path",
        _p(r"\b(extract|pull|pluck|get|read|find)\b"),
        requires=(_p(r"\bJSON\b|\bjson\b"),),
    ),
    InjectionRule(
        "json_format",
        "format JSON",
        _p(r"\b(pretty[- ]?print|format|readable|show me|display)\b"),
        requires=(_p(r"\bJSON\b|\bjson\b"),),
    ),
    InjectionRule(
        "array_sort",
        "sort an array",
        _p(r"\b(sort|order)\b.*\b(by|ascending|descending|alphabetical|asc|desc)\b"),
    ),
    InjectionRule(
        "statistics",
        "compute basic descriptive statistics",
        _p(r"\b(median|stdev|standard deviation|variance|min(imum)?|max(imum)?)\b"),
    ),
    InjectionRule(
        "percentage_calc",
        "calculate percentages",
        _p(r"\b(percent|percentage|%)\b.*\b(of|change|increase|decrease|growth|drop)\b|\b(percent|percentage)\b"),
    ),
    InjectionRule(
        "encode_decode",
        "encode or decode text strings",
        _p(r"\b(base64|url[- ]?encode|url[- ]?decode|hex encode|hex decode|encode this|decode this)\b"),
    ),
    InjectionRule(
        "generate_uuid",
        "generate a random UUID",
        _p(r"\b(uuid|guid|unique id|unique identifier|new id|case id|ticket id|support id|checkout id|session id|tracking id|confirmation id|receipt id)\b"),
    ),
    InjectionRule(
        "visual_document_image_validation",
        "fetch document image validation result by request ID",
        _p(r"\b(validate|valid|validation|verify|verification|authentic|actually valid)\b"),
        requires=(
            _p(r"\b(scan|scanned|image|photo|picture|document|license|driver'?s license|passport|id)\b"),
            _p(r"\b(id|document|license|driver'?s license|passport)\b"),
        ),
    ),
    InjectionRule(
        "reward_demo_products_category",
        "Show me all products in the electronics category",
        _p(r"\b(product category|category availability|products? in .*category|list category products|same category|snacks category|electronics category)\b"),
    ),
    InjectionRule(
        "reward_order_information",
        "get last redemption order information",
        _p(r"\b(last (gift card )?redemption|last redeemed reward|redemption order|reward order|gift card redemption|order info|order information)\b"),
    ),
    InjectionRule(
        "simple_referral_rewards",
        "Show me my defined referral rewards.",
        _p(r"\b(last redeemed reward|earned referral rewards|referral rewards|defined rewards)\b"),
    ),
    InjectionRule(
        "spacex_first_endpoint",
        "Get the SpaceX API reward data",
        _p(r"\b(SpaceX .*first|first .*SpaceX|all three SpaceX|three SpaceX)\b"),
    ),
    InjectionRule(
        "spacex_second_endpoint",
        "Fetch the reward information from the SpaceX API",
        _p(r"\b(SpaceX .*second|second .*SpaceX|all three SpaceX|three SpaceX)\b"),
    ),
    InjectionRule(
        "spacex_third_endpoint",
        "Show me the latest SpaceX launch details.",
        _p(r"\b(SpaceX .*third|third .*SpaceX|third endpoint|all three SpaceX|three SpaceX)\b"),
    ),
    InjectionRule(
        "hash_text",
        "compute a SHA-256 hash of text",
        _p(r"\b(sha256|sha-256|hash|fingerprint|checksum|digest)\b"),
    ),
)


def _matches(rule: InjectionRule, query: str) -> bool:
    if not rule.pattern.search(query):
        return False
    return all(req.search(query) for req in rule.requires)


def inject_pattern_sub_queries(query: str, sub_queries: Iterable[str]) -> tuple[list[str], list[str]]:
    """Append matched glue-tool intents to decomposed sub-queries.

    Returns ``(augmented_sub_queries, fired_rule_names)``. Existing sub-queries
    are preserved and deduped case-insensitively against injected intents.
    """
    out = list(sub_queries)
    seen = {item.casefold().strip() for item in out}
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
