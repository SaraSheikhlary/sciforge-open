"""Responses payload parsing and structured-output validation."""

from decimal import Decimal
from typing import Literal

import pytest

from conftest import responses_payload
from sciforge.llm.client import (
    ModelIncompleteError,
    ModelRefusalError,
    ModelResponseParseError,
    ModelSchemaError,
)
from sciforge.llm.parsing import (
    StrictModel,
    extract_output_text,
    parse_json_text,
    parse_usage,
    strict_json_schema,
    validate_structured,
)


class Finding(StrictModel):
    record_id: str
    claim: str
    confidence: Literal["high", "moderate", "low"]
    notes: str | None = None


class Batch(StrictModel):
    items: list[Finding]
    no_relevant_evidence: bool


GOOD = {"items": [{"record_id": "rec_0123456789abcdef", "claim": "X increases Y.", "confidence": "low",
                   "notes": None}], "no_relevant_evidence": False}


def test_structured_output_to_pydantic():
    batch = validate_structured(GOOD, Batch)
    assert isinstance(batch, Batch) and batch.items[0].confidence == "low"


def test_validate_from_json_string_with_fence():
    text = "```json\n" + '{"items": [], "no_relevant_evidence": true}' + "\n```"
    assert validate_structured(text, Batch).no_relevant_evidence is True


def test_schema_violation_wrong_enum_and_missing_field():
    bad = {"items": [{"record_id": "r", "claim": "c", "confidence": "certain"}]}
    with pytest.raises(ModelSchemaError) as info:
        validate_structured(bad, Batch)
    msg = info.value.message
    assert "items.0.confidence" in msg and "no_relevant_evidence" in msg


def test_extra_unknown_fields_rejected():
    bad = {**GOOD, "doi": "10.1234/fake"}
    with pytest.raises(ModelSchemaError, match="doi"):
        validate_structured(bad, Batch)
    nested = {"items": [{**GOOD["items"][0], "title": "Invented title"}], "no_relevant_evidence": False}
    with pytest.raises(ModelSchemaError, match="title"):
        validate_structured(nested, Batch)


def test_schema_error_message_does_not_echo_input_values():
    secretish = "SOURCE TEXT THAT SHOULD NOT BE ECHOED"
    with pytest.raises(ModelSchemaError) as info:
        validate_structured({"items": secretish, "no_relevant_evidence": False}, Batch)
    assert secretish not in info.value.message


def test_parse_json_text_invalid():
    with pytest.raises(ModelResponseParseError) as info:
        parse_json_text("{not json")
    assert info.value.kind == "invalid_output_json"


def test_strict_json_schema_subset():
    schema = strict_json_schema(Batch)
    finding = schema["$defs"]["Finding"]
    assert schema["additionalProperties"] is False and finding["additionalProperties"] is False
    assert set(finding["required"]) == {"record_id", "claim", "confidence", "notes"}   # all required, notes nullable
    assert "default" not in finding["properties"]["notes"]
    assert "title" not in schema


def test_strict_json_schema_keeps_field_named_title():
    class HasTitle(StrictModel):
        title: str
        default: int

    schema = strict_json_schema(HasTitle)
    assert set(schema["properties"]) == {"title", "default"}
    assert schema["required"] == ["title", "default"]


@pytest.mark.parametrize("payload", [None, [], "text", 42])
def test_non_object_payload(payload):
    with pytest.raises(ModelResponseParseError) as info:
        extract_output_text(payload)
    assert info.value.kind == "malformed_payload"


def test_output_not_a_list():
    with pytest.raises(ModelResponseParseError):
        extract_output_text(responses_payload(output={"oops": 1}))


def test_provider_error_object():
    with pytest.raises(ModelResponseParseError) as info:
        extract_output_text(responses_payload(error={"message": "model overloaded"}))
    assert info.value.kind == "provider_error" and "overloaded" in info.value.message


def test_in_progress_without_text():
    with pytest.raises(ModelResponseParseError) as info:
        extract_output_text(responses_payload(None, status="in_progress"))
    assert info.value.kind == "missing_output_text"


def test_unexpected_status_with_text():
    with pytest.raises(ModelResponseParseError) as info:
        extract_output_text(responses_payload("x", status="failed"))
    assert info.value.kind == "unexpected_status"


def test_incomplete_without_details():
    with pytest.raises(ModelIncompleteError) as info:
        extract_output_text(responses_payload(None, status="incomplete"))
    assert info.value.reason is None


def test_refusal_text_variant():
    payload = responses_payload(None, extra_output=[{"type": "message", "content": [{"type": "refusal",
                                                                                      "text": "no"}]}])
    with pytest.raises(ModelRefusalError):
        extract_output_text(payload)


def test_non_assistant_and_junk_items_ignored():
    payload = responses_payload("ok", extra_output=[
        "junk", {"type": "message", "role": "user", "content": [{"type": "output_text", "text": "NO"}]},
        {"type": "message", "content": [None, {"type": "output_text", "text": 5}]}])
    assert extract_output_text(payload) == "ok"


def test_usage_fallback_names_and_invalid_values():
    u = parse_usage({"usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 20,
                               "completion_tokens_details": {"reasoning_tokens": 6}}})
    assert (u.input_tokens, u.output_tokens, u.reasoning_tokens) == (10, 4, 6)
    assert u.billable_output_tokens == 10
    bad = parse_usage({"usage": {"input_tokens": "12", "output_tokens": -1, "total_tokens": True,
                                 "cost_in_nano_usd": 2_000_000_000}})
    assert (bad.input_tokens, bad.output_tokens, bad.total_tokens) == (None, None, None)
    assert bad.cost_usd_reported == Decimal(2) and bad.cost_source == "reported_nano" and bad.reported is True
    assert parse_usage({"usage": None}).reported is False
    assert parse_usage("x").reported is False


def test_cost_ticks_preferred_over_nano():
    u = parse_usage({"usage": {"cost_in_usd_ticks": 10_000_000_000, "cost_in_nano_usd": 5}})
    assert u.cost_usd_reported == Decimal(1) and u.cost_source == "reported_ticks"


def test_exact_tick_conversion():
    u = parse_usage({"usage": {"cost_in_usd_ticks": 12345678901}})
    assert u.cost_usd_reported == Decimal("1.2345678901")
    assert str(u.cost_usd_reported) == "1.2345678901"            # exact, no float rounding
    assert parse_usage({"usage": {"cost_in_usd_ticks": 1}}).cost_usd_reported == Decimal("1E-10")


def test_exact_nano_conversion():
    u = parse_usage({"usage": {"cost_in_nano_usd": 1234567891}})
    assert u.cost_usd_reported == Decimal("1.234567891") and u.cost_source == "reported_nano"


@pytest.mark.parametrize("raw", [-5, 1.5, "abc", True, None])
def test_invalid_cost_values_ignored(raw):
    assert parse_usage({"usage": {"cost_in_usd_ticks": raw}}).cost_usd_reported is None


def test_integral_float_or_string_ticks_accepted():
    assert parse_usage({"usage": {"cost_in_usd_ticks": 2e10}}).cost_usd_reported == Decimal(2)
    assert parse_usage({"usage": {"cost_in_usd_ticks": "30000000000"}}).cost_usd_reported == Decimal(3)
