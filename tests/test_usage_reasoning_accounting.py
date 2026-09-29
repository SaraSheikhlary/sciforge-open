"""Output-side token accounting with reasoning tokens and provider-reported cost (offline: MockTransport)."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import httpx

from conftest import json_response, model_env, responses_payload
from sciforge.config import ModelSettings
from sciforge.llm.budget import BudgetLimits, BudgetTracker, PriceTable
from sciforge.llm.client import ModelMessage, ModelRequest, ModelUsage
from sciforge.llm.parsing import parse_usage
from sciforge.llm.xai import XAIClient

PRICES = PriceTable(input_per_mtok=2.0, output_per_mtok=10.0)
SECRET_REASONING = "PRIVATE-CHAIN-OF-THOUGHT-should-never-be-stored"
REPO = Path(__file__).resolve().parents[1]


def req(max_out=2000):
    return ModelRequest(messages=(ModelMessage("user", "x" * 300),), max_output_tokens=max_out, stage="evidence")


def realistic_usage(**over):
    usage = {"input_tokens": 1200, "output_tokens": 300, "total_tokens": 2100,
             "input_tokens_details": {"cached_tokens": 100},
             "output_tokens_details": {"reasoning_tokens": 600}}
    usage.update(over)
    return usage


def test_xai_response_with_reasoning_tokens_is_parsed_and_reasoning_content_dropped():
    payload = responses_payload('{"answer": "ok"}', usage=realistic_usage(cost_in_usd_ticks=123_456_789),
                                extra_output=[{"type": "reasoning", "id": "rs_1",
                                               "summary": [{"type": "summary_text", "text": SECRET_REASONING}],
                                               "encrypted_content": SECRET_REASONING}])
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return json_response(payload)

    settings = ModelSettings.from_env(model_env())
    client = XAIClient(settings, http=httpx.Client(transport=httpx.MockTransport(handler)))
    response = client.complete(ModelRequest(messages=(ModelMessage("user", "q" * 30),), max_output_tokens=500))
    assert sent[0]["store"] is False and sent[0]["max_output_tokens"] == 500
    u = response.usage
    assert (u.output_tokens, u.reasoning_tokens, u.total_tokens) == (300, 600, 2100)
    assert u.billable_output_tokens == 900                     # total - input: reasoning charged once
    assert u.reasoning_included_in_output is False
    assert u.cost_usd_reported == Decimal("0.0123456789") and u.cost_source == "reported_ticks"
    d = u.to_dict()
    assert d["output_side_tokens"] == 900 and d["cost_reported_status"] == "reported"
    dumped = json.dumps(response.to_dict()) + response.text
    assert SECRET_REASONING not in dumped


def test_budget_records_reasoning_reported_cost_and_output_side_tokens():
    t = BudgetTracker(BudgetLimits(max_spend_usd=None), PRICES)
    acc = t.record(t.reserve(req()), parse_usage({"usage": realistic_usage(cost_in_usd_ticks=10_000_000_000)}))
    assert acc["output_tokens_reported"] == 300 and acc["reasoning_tokens_reported"] == 600
    assert acc["total_tokens_reported"] == 2100 and acc["output_side_tokens_charged"] == 900
    assert acc["reported_cost_usd"] == "1" and acc["reported_cost_status"] == "reported"
    assert acc["cost_source"] == "reported_ticks" and acc["cost_usd"] == "1"
    # price estimate covers input + output-side (incl. reasoning) tokens: 1200*2 + 900*10 per million
    assert Decimal(acc["price_estimate_usd"]) == Decimal("0.0114")
    s = t.summary()
    assert s["used"]["reasoning_tokens"] == 600 and s["used"]["output_tokens_reported"] == 300
    assert s["used"]["output_side_tokens_charged"] == 900 and s["used"]["reported_cost_usd"] == "1"
    assert "SCIFORGE_MAX_SPEND_USD is the hard financial guard" in s["financial_guard"]
    assert "not a guaranteed ceiling" in s["financial_guard"]


def test_cost_unavailable_is_recorded_as_unavailable_and_estimated_conservatively():
    t = BudgetTracker(BudgetLimits(max_spend_usd=None), PRICES)
    acc = t.record(t.reserve(req()), parse_usage({"usage": realistic_usage()}))
    assert acc["reported_cost_usd"] is None and acc["reported_cost_status"] == "unavailable"
    assert acc["cost_source"] == "price_estimate" and Decimal(acc["cost_usd"]) == Decimal("0.0114")
    assert t.summary()["used"]["reported_cost_usd"] is None


def test_reasoning_reported_separately_is_always_added_once():
    # xAI reports reasoning separately: output + reasoning is charged even if the total looks "included"
    u = ModelUsage(input_tokens=1000, output_tokens=500, reasoning_tokens=400, total_tokens=1500)
    assert u.billable_output_tokens == 900
    # xAI's documented shape (total = input + output + reasoning): exactly output + reasoning, no double count
    u = ModelUsage(input_tokens=1000, output_tokens=500, reasoning_tokens=400, total_tokens=1900)
    assert u.billable_output_tokens == 900 and u.reasoning_included_in_output is False


def test_reasoning_larger_than_output_is_always_charged_on_top():
    # a total that ignores reasoning cannot hide reasoning > output: charge output + reasoning (more conservative)
    u = ModelUsage(input_tokens=1000, output_tokens=100, reasoning_tokens=700, total_tokens=1100)
    assert u.billable_output_tokens == 800 and u.reasoning_included_in_output is False


def test_no_total_charges_output_plus_reasoning_and_no_usage_charges_worst_case():
    assert ModelUsage(input_tokens=10, output_tokens=100, reasoning_tokens=50).billable_output_tokens == 150
    assert ModelUsage(input_tokens=10, output_tokens=100).billable_output_tokens == 100
    t = BudgetTracker(BudgetLimits(max_spend_usd=None), PRICES)
    acc = t.record(t.reserve(req(max_out=700)), None)
    assert acc["output_side_tokens_charged"] == 700 and acc["cost_source"] == "estimated_worst_case"
    assert acc["reported_cost_status"] == "unavailable" and acc["reasoning_tokens_reported"] is None


def test_malformed_usage_fields_never_fabricate_cost_or_tokens():
    u = parse_usage({"usage": {"input_tokens": "12", "output_tokens": -3, "cost_in_usd_ticks": "abc",
                               "output_tokens_details": {"reasoning_tokens": None}}})
    assert u.cost_usd_reported is None and u.reasoning_tokens is None
    assert u.to_dict()["cost_reported_status"] == "unavailable"


def test_docs_do_not_claim_output_setting_is_a_hard_ceiling():
    texts = [(REPO / p).read_text(encoding="utf-8") for p in ("README.md", "docs/web-app.md", ".env.example",
                                                                "docs/v0.3-model-layer.md")]
    for text in texts:
        low = text.lower()
        assert "guaranteed hard ceiling" not in low.replace("not a guaranteed hard ceiling", "")
    assert any("SCIFORGE_MAX_SPEND_USD" in t and "hard financial guard" in t for t in texts)
