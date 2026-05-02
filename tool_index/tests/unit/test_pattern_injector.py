"""Tests for deterministic glue-tool sub-query injection."""
from __future__ import annotations

from tool_index.retrieval.pattern_injector import inject_pattern_sub_queries


def test_pattern_injector_appends_matching_intents():
    out, fired = inject_pattern_sub_queries(
        "Fetch this API URL, pull the status from JSON, and hash it",
        ["fetch the API URL"],
    )
    assert "fetch the API URL" in out
    assert "parse a URL into components" in out
    assert "fetch content from a URL" in out
    assert "extract values from JSON by path" in out
    assert "compute a SHA-256 hash of text" in out
    assert fired == ["url_parse", "http_get", "json_query", "hash_text"]


def test_pattern_injector_dedupes_existing_intent():
    out, fired = inject_pattern_sub_queries(
        "What day of the week is today?",
        ["get the current UTC date and time"],
    )
    assert out.count("get the current UTC date and time") == 1
    assert "get weekday from a date" in out
    assert fired == ["day_of_week"]


def test_pattern_injector_broadens_uuid_id_patterns():
    out, fired = inject_pattern_sub_queries(
        "Validate this ID scan, generate a case id, hash the case id, and timestamp the record",
        [],
    )
    assert "generate a random UUID" in out
    assert "compute a SHA-256 hash of text" in out
    assert "get the current UTC date and time" in out
    assert fired == [
        "get_current_datetime",
        "generate_uuid",
        "visual_document_image_validation",
        "hash_text",
    ]


def test_pattern_injector_extracts_api_response_fields_without_json_word():
    out, fired = inject_pattern_sub_queries(
        "Pretty-print this API response and pull out the 'status' field",
        ["Pretty-print the API response"],
    )
    assert "extract values from JSON by path" in out
    assert "json_query" in fired


def test_pattern_injector_covers_visual_document_validation():
    out, fired = inject_pattern_sub_queries(
        "Is this scan of my driver's license actually valid?",
        ["Validate driver's license from scan"],
    )
    assert "fetch document image validation result by request ID" in out
    assert "visual_document_image_validation" in fired


def test_pattern_injector_adds_catalog_specific_aliases():
    out, fired = inject_pattern_sub_queries(
        "Compare the first and second SpaceX endpoints, fetch the third endpoint, "
        "then check product category availability and last redemption order info",
        [],
    )
    assert "Show me all products in the electronics category" in out
    assert "get last redemption order information" in out
    assert "Get the SpaceX API reward data" in out
    assert "Fetch the reward information from the SpaceX API" in out
    assert "Show me the latest SpaceX launch details." in out
    assert "reward_demo_products_category" in fired
    assert "reward_order_information" in fired
    assert "spacex_first_endpoint" in fired
    assert "spacex_second_endpoint" in fired
    assert "spacex_third_endpoint" in fired


def test_pattern_injector_covers_remaining_measured_misses():
    out, fired = inject_pattern_sub_queries(
        "Look up my last redeemed reward, calculate how long 26.2 miles takes "
        "at 12 km/h, and make a receipt id",
        [],
    )
    assert "evaluate arithmetic expressions" in out
    assert "generate a random UUID" in out
    assert "get last redemption order information" in out
    assert "Show me my defined referral rewards." in out
    assert "calculator" in fired
    assert "generate_uuid" in fired
    assert "reward_order_information" in fired
    assert "simple_referral_rewards" in fired
