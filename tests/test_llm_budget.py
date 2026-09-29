"""BudgetTracker (decision D8): per-attempt pre-checks, exact spend, fail-closed prices."""

from decimal import Decimal

import pytest

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
    ModelAuthError,
    ModelConfigError,
    ModelIncompleteError,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelUsage,
)
from sciforge.llm.fake import FakeModelClient

NO_CAP = BudgetLimits(max_spend_usd=None)
PRICES = PriceTable(input_per_mtok=2.0, output_per_mtok=10.0)
NO_SLEEP = {"sleep": lambda s: None}


def req(chars=300, max_out=2000):
    return ModelRequest(messages=(ModelMessage("user", "x" * chars),), max_output_tokens=max_out)


def usage(inp=100, out=50, **kw):
    return ModelUsage(input_tokens=inp, output_tokens=out, **kw)


def test_d8_defaults():
    t = BudgetTracker(BudgetLimits(), PRICES)
    assert t.summary()["limits"] == {"max_attempts": 30, "max_sources": 10, "max_input_tokens": 200_000,
                                     "max_output_tokens_per_call": 2_000, "max_spend_usd": "15"}
    assert BudgetLimits().max_spend_usd == Decimal("15")


def test_estimate_is_pessimistic():
    assert estimate_input_tokens(req(chars=300)) == 100 + 8        # chars/3 + per-message overhead


def test_max_attempts_stops_before_exceeding_and_is_sticky():
    t = BudgetTracker(BudgetLimits(max_attempts=2, max_spend_usd=None))
    for _ in range(2):
        t.record(t.reserve(req()), usage())
    assert t.exhausted_by == "max_attempts"                        # reached → sticky
    with pytest.raises(BudgetExhausted) as info:
        t.reserve(req())
    assert info.value.limit == "max_attempts" and t.attempts == 2
    assert t.summary()["remaining"]["attempts"] == 0


def test_keep_attempts_reserves_for_later_stages():
    t = BudgetTracker(BudgetLimits(max_attempts=3, max_spend_usd=None))
    t.record(t.reserve(req(), keep_attempts=2), usage())
    with pytest.raises(BudgetExhausted):
        t.reserve(req(), keep_attempts=2)
    assert t.exhausted_by is None                                   # keep-refusal is not sticky
    t.record(t.reserve(req()), usage())                             # later stage uses the kept attempts
    t.record(t.reserve(req()), usage())
    assert t.attempts == 3 and t.exhausted_by == "max_attempts"


def test_pending_reservations_count():
    t = BudgetTracker(BudgetLimits(max_attempts=2, max_spend_usd=None))
    r = t.reserve(req())
    t.reserve(req())
    t.release(r)
    t.reserve(req())
    assert len(t._pending) == 2


def test_input_tokens_stop_before_exceeding():
    t = BudgetTracker(BudgetLimits(max_input_tokens=1000, max_spend_usd=None))
    t.record(t.reserve(req(chars=1500)), usage(inp=600))           # est 508 fits
    with pytest.raises(BudgetExhausted) as info:
        t.reserve(req(chars=1500))                                  # 600 + 508 > 1000: refused BEFORE the call
    assert info.value.limit == "max_input_tokens" and t.input_tokens == 600


def test_recorded_usage_reaching_limit_stops_further_attempts():
    t = BudgetTracker(BudgetLimits(max_input_tokens=1000, max_spend_usd=None))
    t.record(t.reserve(req(chars=30)), usage(inp=1000))             # provider reported more than estimated
    assert t.exhausted_by == "max_input_tokens"
    with pytest.raises(BudgetExhausted):
        t.reserve(req(chars=3))


def test_output_tokens_capped_per_call():
    t = BudgetTracker(BudgetLimits(max_output_tokens_per_call=2000, max_spend_usd=None))
    res = t.reserve(req(max_out=10_000))
    assert res.max_output_tokens == 2000 and res.request.max_output_tokens == 2000
    assert t.reserve(req(max_out=300)).request.max_output_tokens == 300


def test_spend_worst_case_precheck_with_prices():
    # worst case per attempt: 108 in * $2/M + 2000 out * $10/M = $0.020216 (exact Decimal)
    t = BudgetTracker(BudgetLimits(max_spend_usd=0.05), PRICES)
    r = t.reserve(req())
    assert r.worst_case_cost_usd == Decimal("0.020216")
    t.record(r, usage(inp=100, out=2000))                           # price_estimate $0.0202
    t.record(t.reserve(req()), usage(inp=100, out=2000))            # $0.0404
    with pytest.raises(BudgetExhausted) as info:
        t.reserve(req())                                            # 0.0404 + 0.020216 > 0.05
    assert info.value.limit == "max_spend_usd"
    assert t.spend_usd == Decimal("0.0404")


def test_spend_never_exceeds_cap_over_many_attempts():
    t = BudgetTracker(BudgetLimits(max_attempts=500, max_input_tokens=10_000_000, max_spend_usd=1), PRICES)
    with pytest.raises(BudgetExhausted):
        while True:
            t.record(t.reserve(req(chars=3000, max_out=2000)), usage(inp=1000, out=2000))
    assert t.spend_usd <= Decimal(1) and t.attempts > 0


def test_fail_closed_when_cap_enabled_and_prices_unset():
    with pytest.raises(ModelConfigError, match="SCIFORGE_MAX_SPEND_USD=none"):
        BudgetTracker(BudgetLimits(max_spend_usd=15.0), PriceTable())
    with pytest.raises(ModelConfigError):
        BudgetTracker(BudgetLimits(), PriceTable(input_per_mtok=1.0))


def test_cost_source_reported_ticks_is_recorded_actual():
    t = BudgetTracker(BudgetLimits(max_spend_usd=10), PRICES)
    acct = t.record(t.reserve(req()), usage(cost_usd_reported=Decimal("0.0000001"), cost_source="reported_ticks"))
    assert acct["cost_source"] == "reported_ticks" and acct["cost_usd"] == "0.0000001"
    assert acct["price_estimate_usd"] == "0.0007"                   # estimate kept for information only
    assert t.spend_usd == Decimal("0.0000001")                      # reported, even though estimate is larger


def test_reported_cost_greater_than_estimate_recorded_exactly():
    t = BudgetTracker(BudgetLimits(max_spend_usd=10), PRICES)
    acct = t.record(t.reserve(req()), usage(cost_usd_reported=Decimal("3"), cost_source="reported_nano"))
    assert acct["cost_source"] == "reported_nano" and t.spend_usd == Decimal(3)


def test_price_estimate_when_usage_but_no_reported_cost():
    t = BudgetTracker(BudgetLimits(max_spend_usd=10), PRICES)
    acct = t.record(t.reserve(req()), usage(inp=1000, out=1000))
    assert acct["cost_source"] == "price_estimate" and acct["cost_usd"] == "0.012"


def test_missing_usage_charged_worst_case_tokens_and_cost():
    t = BudgetTracker(BudgetLimits(max_spend_usd=10), PRICES)
    r = t.reserve(req(chars=300, max_out=700))
    acct = t.record(r, ModelUsage(reported=False))
    assert acct["input_tokens_charged"] == 108 and acct["output_tokens_charged"] == 700
    assert acct["cost_source"] == "estimated_worst_case" and Decimal(acct["cost_usd"]) == r.worst_case_cost_usd


def test_failed_attempt_without_usage_charged_worst_case():
    t = BudgetTracker(BudgetLimits(max_spend_usd=10), PRICES)
    r = t.reserve(req())
    acct = t.record(r, None, failed=True)
    assert acct["cost_source"] == "estimated_worst_case" and t.failed_attempts == 1
    assert t.input_tokens == 108 and t.output_tokens == 2000 and t.spend_usd == Decimal("0.020216")


def test_unpriced_when_cap_disabled_and_no_prices():
    t = BudgetTracker(NO_CAP, PriceTable())
    acct = t.record(t.reserve(req()), usage())
    assert acct["cost_source"] == "unpriced" and acct["cost_usd"] is None
    assert t.summary()["used"]["unpriced_attempts"] == 1
    acct2 = t.record(t.reserve(req()), usage(cost_usd_reported=Decimal("0.5"), cost_source="reported_ticks"))
    assert acct2["cost_source"] == "reported_ticks" and t.summary()["used"]["spend_usd"] == "0.5"


def test_reasoning_tokens_charged_as_output():
    t = BudgetTracker(NO_CAP, PRICES)
    t.record(t.reserve(req()), ModelUsage(input_tokens=32, output_tokens=9, reasoning_tokens=110, total_tokens=151))
    assert t.output_tokens == 119 and t.reasoning_tokens == 110


def test_sources_limit_does_not_stop_attempts():
    t = BudgetTracker(BudgetLimits(max_sources=10, max_spend_usd=None))
    assert t.admit_sources(7) == 7 and t.admit_sources(7) == 3 and t.admit_sources(1) == 0
    assert t.sources == 10 and t.summary()["sources_limited"] is True
    assert t.exhausted_by is None
    t.reserve(req())


def test_record_twice_rejected_and_invalid_limits():
    t = BudgetTracker(NO_CAP)
    r = t.reserve(req())
    t.record(r, usage())
    with pytest.raises(ValueError):
        t.record(r, usage())
    with pytest.raises(ValueError):
        BudgetLimits(max_attempts=0)
    with pytest.raises(ValueError):
        BudgetLimits(max_spend_usd=0)
    with pytest.raises(ValueError):
        PriceTable(input_per_mtok=-1)
    assert RetryPolicy(2, 0.5).delay(1, None) == 1.0 and RetryPolicy().delay(0, 7.0) == 7.0


def test_budgeted_call_success_and_budget_stop_before_client_called():
    fake = FakeModelClient([{"a": 1}, {"a": 2}])
    t = BudgetTracker(BudgetLimits(max_attempts=1, max_spend_usd=None))
    budgeted_call(fake, req(max_out=5000), t, **NO_SLEEP)
    assert fake.requests[0].max_output_tokens == 2000              # capped request actually sent
    with pytest.raises(BudgetExhausted):
        budgeted_call(fake, req(), t, **NO_SLEEP)
    assert len(fake.requests) == 1 and fake.remaining == 1         # client never called


def test_budgeted_call_non_retryable_failure_counted_once():
    fake = FakeModelClient([ModelAuthError("HTTP 401", http_status=401), {"never": 1}])
    t = BudgetTracker(NO_CAP, PRICES)
    with pytest.raises(ModelAuthError):
        budgeted_call(fake, req(), t, **NO_SLEEP)
    assert t.attempts == 1 and t.failed_attempts == 1 and len(fake.requests) == 1
    assert t.input_tokens == 108 and t.output_tokens == 2000       # worst case (no usage)


def test_budgeted_call_error_with_usage_is_billed_from_usage():
    err = ModelIncompleteError("incomplete", reason="max_output_tokens", usage=usage(inp=90, out=2000))
    t = BudgetTracker(NO_CAP, PRICES)
    with pytest.raises(ModelIncompleteError):
        budgeted_call(FakeModelClient([err]), req(), t, **NO_SLEEP)
    assert t.input_tokens == 90 and t.output_tokens == 2000 and t.attempts == 1


def test_budgeted_call_unexpected_exception_charged_conservatively():
    def boom(request):
        raise RuntimeError("bug")

    t = BudgetTracker(NO_CAP)
    with pytest.raises(RuntimeError):
        budgeted_call(FakeModelClient([boom]), req(), t, **NO_SLEEP)
    assert t.attempts == 1 and not t._pending


def test_summary_shape():
    t = BudgetTracker(BudgetLimits(max_spend_usd=15.0), PRICES)
    t.record(t.reserve(req()), usage())
    s = t.summary()
    assert set(s) == {"limits", "used", "remaining", "prices_configured", "spend_cap_enabled",
                      "sources_limited", "exhausted_by"}
    assert s["used"]["attempts"] == 1 and s["remaining"]["attempts"] == 29
    assert s["remaining"]["spend_usd"] == "14.9993"
    assert s["used"]["spend_by_cost_source_usd"]["price_estimate"] == "0.0007"


def test_fake_response_object_usage_recorded():
    resp = ModelResponse(text="{}", parsed={}, usage=usage(inp=7, out=3), model="m")
    t = BudgetTracker(NO_CAP)
    out = budgeted_call(FakeModelClient([resp]), req(), t, **NO_SLEEP)
    assert t.input_tokens == 7 and t.output_tokens == 3 and out.attempts == 1
