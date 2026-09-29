"""Model attempt budget default (15 per investigation) and spend-cap configuration. Offline only."""

from __future__ import annotations

import pytest

from conftest import model_env
from sciforge.config import DEFAULT_MAX_SPEND_USD, DEFAULT_MODEL_MAX_ATTEMPTS, ModelSettings
from sciforge.llm.budget import BudgetLimits, BudgetTracker, RetryPolicy, budgeted_call
from sciforge.llm.client import BudgetExhausted, ModelConfigError, ModelMessage, ModelRequest
from sciforge.llm.fake import FakeModelClient

PRICES = {"SCIFORGE_PRICE_INPUT_PER_MTOK": "1", "SCIFORGE_PRICE_OUTPUT_PER_MTOK": "2"}


def test_default_attempt_limit_is_15():
    assert DEFAULT_MODEL_MAX_ATTEMPTS == 15 and BudgetLimits().max_attempts == 15
    s = ModelSettings.from_env(model_env())
    assert s.max_attempts == 15 and s.budget_limits().max_attempts == 15


@pytest.mark.parametrize("raw, expected", [("5", 5), ("30", 30), ("1", 1)])
def test_attempt_limit_override_via_env(raw, expected):
    assert ModelSettings.from_env(model_env(SCIFORGE_MODEL_MAX_ATTEMPTS=raw)).max_attempts == expected


def test_sixteenth_attempt_is_refused_by_default():
    s = ModelSettings.from_env(model_env())
    tracker = BudgetTracker(s.budget_limits(), s.price_table())
    client = FakeModelClient([{"ok": True}] * 20)
    req = ModelRequest(messages=(ModelMessage("user", "q"),), max_output_tokens=100, stage="t")
    for _ in range(15):
        budgeted_call(client, req, tracker, retry=RetryPolicy(max_retries=0), sleep=lambda s: None)
    with pytest.raises(BudgetExhausted) as info:
        budgeted_call(client, req, tracker, retry=RetryPolicy(max_retries=0), sleep=lambda s: None)
    assert info.value.limit == "max_attempts" and tracker.attempts == 15 and len(client.requests) == 15


def test_spend_cap_default_unchanged_and_two_dollar_cap_via_env():
    assert DEFAULT_MAX_SPEND_USD == 15.0
    assert ModelSettings.from_env(model_env(SCIFORGE_MAX_SPEND_USD=None, **PRICES)).max_spend_usd == 15.0
    two = ModelSettings.from_env(model_env(SCIFORGE_MAX_SPEND_USD="2", **PRICES))
    assert two.max_spend_usd == 2.0 and two.budget_limits().max_spend_usd == 2
    with pytest.raises(ModelConfigError):                       # cap on without prices still fails closed
        ModelSettings.from_env(model_env(SCIFORGE_MAX_SPEND_USD="2"))
