"""Every API attempt (retries included) is budgeted, recorded, and audited.

Offline only: httpx.MockTransport or FakeModelClient; sleep is patched.
"""

import json
from datetime import datetime, timezone
from decimal import Decimal

import httpx
import pytest

from conftest import FAKE_XAI_KEY, Sleeper, json_response, model_env, responses_payload
from sciforge.config import ModelSettings
from sciforge.llm.audit import ModelCallAudit
from sciforge.llm.budget import (
    BudgetLimits,
    BudgetTracker,
    PriceTable,
    RetryPolicy,
    budgeted_call,
    estimate_input_tokens,
)
from sciforge.llm.client import (
    BudgetExhausted,
    ModelHTTPError,
    ModelMessage,
    ModelRateLimited,
    ModelRequest,
    ModelTimeout,
)
from sciforge.llm.fake import FakeModelClient
from sciforge.llm.xai import XAIClient

NOW = datetime(2026, 9, 28, 12, 0, 0, tzinfo=timezone.utc)
SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"],
          "additionalProperties": False}
PRICES = PriceTable(input_per_mtok=2.0, output_per_mtok=10.0)


def make_request(chars=300, max_out=2000):
    return ModelRequest(messages=(ModelMessage("user", "q" * chars),), max_output_tokens=max_out,
                        schema_name="answer", json_schema=SCHEMA, stage="test_stage")


class Harness:
    def __init__(self, handler, *, limits=None, prices=None, max_retries=2, backoff=0.5, store_prompts=True):
        self.http_requests: list[httpx.Request] = []
        self.sleeper = Sleeper()

        def recording(request):
            self.http_requests.append(request)
            return handler(request)

        settings = ModelSettings.from_env(model_env(XAI_BASE_URL="https://api.x.ai.test"))
        self.client = XAIClient(settings, http=httpx.Client(transport=httpx.MockTransport(recording)),
                                now=lambda: NOW)
        self.tracker = BudgetTracker(limits or BudgetLimits(max_spend_usd=None), prices or PriceTable())
        self.audit = ModelCallAudit(store_prompts=store_prompts, secrets=settings.secret_values(),
                                    now=lambda: "2026-09-28T12:00:00Z")
        self.retry = RetryPolicy(max_retries=max_retries, backoff_seconds=backoff)

    def call(self, request=None, **kw):
        return budgeted_call(self.client, request or make_request(), self.tracker, retry=self.retry,
                             audit=self.audit, sleep=self.sleeper, now=lambda: NOW, **kw)

    def attempts(self):
        return [e for e in self.audit.entries if e["entry_type"] == "attempt"]


def seq(*items):
    items = list(items)

    def handler(request):
        item = items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return handler


def ok(usage=None):
    return json_response(responses_payload('{"answer": "ok"}', usage=usage))


def test_successful_first_attempt():
    h = Harness(seq(ok()))
    resp = h.call()
    assert resp.attempts == 1 and resp.parsed == {"answer": "ok"}
    assert h.tracker.attempts == 1 and h.tracker.failed_attempts == 0 and h.tracker.logical_calls == 1
    (entry,) = h.audit.entries
    assert entry["entry_type"] == "attempt" and entry["attempt"] == 1 and entry["call_id"] == 1
    assert entry["stage"] == "test_stage" and entry["timestamp"] == "2026-09-28T12:00:00Z"
    assert entry["outcome"] == "success" and entry["http_status"] == 200
    assert entry["usage"]["input_tokens"] == 120 and entry["cost_source"] == "unpriced"
    assert h.sleeper.calls == []


def test_failed_first_attempt_then_successful_retry():
    h = Harness(seq(httpx.Response(503), ok()))
    resp = h.call()
    assert resp.attempts == 2 and len(h.http_requests) == 2
    assert h.tracker.attempts == 2 and h.tracker.failed_attempts == 1
    first, second = h.attempts()
    assert (first["attempt"], first["outcome"], first["http_status"]) == (1, "http_error", 503)
    assert first["retry_reason"] == "http_5xx" and first["will_retry"] is True and first["backoff_s"] == 0.5
    assert first["retry_after_s"] is None
    assert (second["attempt"], second["outcome"]) == (2, "success")
    assert h.sleeper.calls == [0.5]


def test_multiple_retries_all_counted():
    h = Harness(seq(httpx.Response(500), httpx.ConnectError("boom"), httpx.Response(502), ok()), max_retries=3)
    resp = h.call()
    assert resp.attempts == 4 and h.tracker.attempts == 4 and h.tracker.failed_attempts == 3
    outcomes = [e["outcome"] for e in h.attempts()]
    assert outcomes == ["http_error", "connection_error", "http_error", "success"]
    assert [e["backoff_s"] for e in h.attempts()[:3]] == [0.5, 1.0, 2.0]
    assert h.sleeper.calls == [0.5, 1.0, 2.0]


def test_retry_blocked_by_attempt_budget():
    h = Harness(lambda r: httpx.Response(503), limits=BudgetLimits(max_attempts=2, max_spend_usd=None),
                max_retries=5)
    with pytest.raises(BudgetExhausted) as info:
        h.call()
    stop = info.value
    assert stop.limit == "max_attempts"
    assert isinstance(stop.last_error, ModelHTTPError) and stop.__cause__ is stop.last_error
    assert stop.last_error.http_status == 503 and stop.last_error.attempts == 2
    assert len(h.http_requests) == 2 and h.tracker.attempts == 2          # no third attempt sent
    first, second, blocked = h.audit.entries
    assert first["will_retry"] is True and second["will_retry"] is False
    assert second["retry_blocked_by"] == "max_attempts" and second["backoff_s"] is None
    assert blocked["entry_type"] == "budget_stop" and blocked["attempt"] == 3
    assert blocked["limit"] == "max_attempts" and blocked["last_error"]["http_status"] == 503
    assert h.sleeper.calls == [0.5]                                        # no sleep before the blocked retry
    with pytest.raises(BudgetExhausted):                                   # sticky: later calls refused too
        h.call()
    assert len(h.http_requests) == 2


EST_IN = estimate_input_tokens(make_request())                 # 145 (messages + schema, chars/3 + overhead)
WORST = PRICES.cost(EST_IN, 2000)                              # exact Decimal, e.g. 0.02029


def test_worst_case_constants():
    assert EST_IN == 145 and WORST == Decimal("0.02029")


def test_retry_blocked_by_spend_budget():
    # 5xx without usage is charged the full worst case, so a $0.05 cap allows exactly two attempts.
    h = Harness(lambda r: httpx.Response(500), limits=BudgetLimits(max_spend_usd=Decimal("0.05")), prices=PRICES,
                max_retries=5)
    with pytest.raises(BudgetExhausted) as info:
        h.call()
    assert info.value.limit == "max_spend_usd" and isinstance(info.value.last_error, ModelHTTPError)
    assert len(h.http_requests) == 2
    assert h.tracker.spend_usd == 2 * WORST <= Decimal("0.05")
    assert [e["cost_source"] for e in h.attempts()] == ["estimated_worst_case"] * 2
    assert [e["cost_usd"] for e in h.attempts()] == [str(WORST)] * 2
    assert h.audit.entries[-1]["limit"] == "max_spend_usd"


def test_first_attempt_blocked_before_any_request():
    h = Harness(seq(ok()), limits=BudgetLimits(max_input_tokens=1000, max_spend_usd=None))
    with pytest.raises(BudgetExhausted) as info:
        h.call(make_request(chars=6000))
    assert info.value.limit == "max_input_tokens" and info.value.last_error is None
    assert h.http_requests == [] and h.tracker.attempts == 0
    (stop,) = h.audit.entries
    assert stop["entry_type"] == "budget_stop" and stop["attempt"] == 1


def test_timeout_attempt_recorded():
    h = Harness(seq(httpx.ReadTimeout("slow"), ok()), limits=BudgetLimits(max_spend_usd=10), prices=PRICES)
    h.call()
    timeout_entry = h.attempts()[0]
    assert timeout_entry["outcome"] == "timeout" and timeout_entry["error_type"] == "timeout"
    assert timeout_entry["http_status"] is None and timeout_entry["retry_reason"] == "timeout"
    assert timeout_entry["usage"] is None and timeout_entry["will_retry"] is True
    assert timeout_entry["cost_source"] == "estimated_worst_case" and timeout_entry["cost_usd"] == str(WORST)
    assert timeout_entry["accounting"]["input_tokens_charged"] == EST_IN
    assert timeout_entry["accounting"]["output_tokens_charged"] == 2000
    assert timeout_entry["backoff_s"] == 0.5
    assert h.attempts()[1]["cost_source"] == "price_estimate"


def test_429_attempt_recorded_with_retry_after():
    h = Harness(seq(httpx.Response(429, headers={"Retry-After": "7"}, json={"error": {"message": "slow"}}), ok()),
                limits=BudgetLimits(max_spend_usd=10), prices=PRICES)
    h.call()
    e = h.attempts()[0]
    assert (e["outcome"], e["http_status"], e["error_type"]) == ("http_error", 429, "rate_limited")
    assert e["retry_reason"] == "rate_limited" and e["retry_after_s"] == 7.0 and e["backoff_s"] == 7.0
    assert e["cost_source"] == "estimated_worst_case"     # conservative: rejection not proven free
    assert h.sleeper.calls == [7.0] and h.tracker.attempts == 2


def test_exact_tick_cost_is_recorded_spend():
    usage = {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110, "cost_in_usd_ticks": 12345678901}
    h = Harness(seq(ok(usage)), limits=BudgetLimits(max_spend_usd=15), prices=PRICES)
    resp = h.call()
    assert resp.usage.cost_usd_reported == Decimal("1.2345678901")
    (e,) = h.attempts()
    assert e["cost_usd"] == "1.2345678901" and e["cost_source"] == "reported_ticks"
    assert e["accounting"]["price_estimate_usd"] == "0.0003"         # estimate kept, not recorded as spend
    assert h.tracker.spend_usd == Decimal("1.2345678901")
    assert h.tracker.summary()["used"]["spend_by_cost_source_usd"]["reported_ticks"] == "1.2345678901"


def test_exact_nano_cost_is_recorded_spend():
    usage = {"input_tokens": 100, "output_tokens": 10, "cost_in_nano_usd": 1234567891}
    h = Harness(seq(ok(usage)), limits=BudgetLimits(max_spend_usd=15), prices=PRICES)
    h.call()
    (e,) = h.attempts()
    assert e["cost_usd"] == "1.234567891" and e["cost_source"] == "reported_nano"


def test_ticks_sum_exactly_without_float_error():
    usage = {"input_tokens": 1, "output_tokens": 1, "cost_in_usd_ticks": 1_000_000_000}   # $0.1 each
    h = Harness(lambda r: ok(usage), limits=BudgetLimits(max_spend_usd=None))
    for _ in range(3):
        h.call()
    assert h.tracker.spend_usd == Decimal("0.3")                  # 0.1 + 0.1 + 0.1 exactly


def test_reported_cost_reaching_cap_stops_further_calls():
    usage = {"input_tokens": 1, "output_tokens": 1, "cost_in_usd_ticks": 10_000_000_000}  # $1 reported
    h = Harness(lambda r: ok(usage), limits=BudgetLimits(max_spend_usd=1), prices=PRICES)
    h.call()
    assert h.tracker.exhausted_by == "max_spend_usd"
    with pytest.raises(BudgetExhausted):
        h.call()
    assert len(h.http_requests) == 1


def test_non_retryable_error_not_retried_even_with_budget():
    h = Harness(lambda r: httpx.Response(400, json={"error": {"message": "bad"}}))
    with pytest.raises(ModelHTTPError):
        h.call()
    (e,) = h.audit.entries
    assert e["outcome"] == "http_error" and e["retryable"] is False and e["will_retry"] is False
    assert h.tracker.attempts == 1


def test_fake_client_retry_flow_and_keep_attempts():
    fake = FakeModelClient([ModelRateLimited("429", http_status=429, retry_after_s=3.0), {"answer": "x"},
                            ModelTimeout("t")])
    tracker = BudgetTracker(BudgetLimits(max_attempts=4, max_spend_usd=None))
    slept: list[float] = []
    resp = budgeted_call(fake, make_request(), tracker, retry=RetryPolicy(2, 1.0), sleep=slept.append)
    assert resp.attempts == 2 and slept == [3.0] and tracker.attempts == 2
    # 2 of 4 used, 1 kept for a later stage → only one more attempt allowed; the timeout retry is blocked
    with pytest.raises(BudgetExhausted) as info:
        budgeted_call(fake, make_request(), tracker, retry=RetryPolicy(2, 1.0), sleep=slept.append,
                      keep_attempts=1)
    assert isinstance(info.value.last_error, ModelTimeout) and tracker.attempts == 3
    assert tracker.exhausted_by is None and tracker.remaining_attempts == 1   # kept attempt still usable


def test_audit_never_contains_key_across_retries():
    h = Harness(seq(httpx.Response(500, json={"error": {"message": f"Bearer {FAKE_XAI_KEY}"}}), ok()))
    h.call()
    dumped = json.dumps(h.audit.entries)
    assert FAKE_XAI_KEY not in dumped and "Authorization" not in dumped
