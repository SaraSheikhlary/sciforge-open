"""xAI request timeout (SCIFORGE_MODEL_TIMEOUT_SECONDS, default 120 s). Offline: httpx.MockTransport only."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import httpx
import pytest

from conftest import FAKE_XAI_KEY, Sleeper, json_response, model_env, responses_payload
from sciforge.config import (
    DEFAULT_MODEL_TIMEOUT_SECONDS,
    MAX_MODEL_TIMEOUT_SECONDS,
    ModelSettings,
    parse_model_timeout,
)
from sciforge.llm.audit import ModelCallAudit
from sciforge.llm.budget import BudgetLimits, BudgetTracker, PriceTable, RetryPolicy, budgeted_call
from sciforge.llm.client import ModelMessage, ModelRequest, ModelTimeout
from sciforge.llm.xai import XAIClient

NOW = datetime(2026, 9, 28, 12, 0, 0, tzinfo=timezone.utc)
SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"],
          "additionalProperties": False}


def request() -> ModelRequest:
    return ModelRequest(messages=(ModelMessage("user", "q" * 300),), max_output_tokens=500, schema_name="answer",
                        json_schema=SCHEMA, stage="test_stage")


# ------------------------------------------------------------------ parsing


def test_default_is_120_seconds():
    assert DEFAULT_MODEL_TIMEOUT_SECONDS == 120.0 and MAX_MODEL_TIMEOUT_SECONDS == 600.0
    assert ModelSettings.from_env(model_env()).timeout_seconds == 120.0
    assert ModelSettings(api_key=FAKE_XAI_KEY, model="m", max_spend_usd=None).timeout_seconds == 120.0
    assert parse_model_timeout({}) == 120.0


@pytest.mark.parametrize("raw, expected", [("30", 30.0), ("90.5", 90.5), (" 45 ", 45.0), ("1", 1.0),
                                           ("0.5", 0.5), ("600", 600.0), ("2e2", 200.0)])
def test_custom_value(raw, expected):
    assert ModelSettings.from_env(model_env(SCIFORGE_MODEL_TIMEOUT_SECONDS=raw)).timeout_seconds == expected


@pytest.mark.parametrize("raw", ["", "   ", "abc", "0", "-5", "-0.1", "nan", "NaN", "inf", "-inf", "Infinity",
                                 "600.01", "601", "1e9", "30s", "1,5"])
def test_invalid_blank_nonpositive_nonfinite_or_too_large_falls_back_to_default(raw):
    s = ModelSettings.from_env(model_env(SCIFORGE_MODEL_TIMEOUT_SECONDS=raw))    # never raises
    assert s.timeout_seconds == 120.0


# ------------------------------------------------------------------ wiring into XAIClient / httpx


def test_timeout_is_passed_to_every_httpx_request():
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req.extensions["timeout"])
        return json_response(responses_payload('{"answer": "ok"}'))

    settings = ModelSettings.from_env(model_env(SCIFORGE_MODEL_TIMEOUT_SECONDS="42"))
    client = XAIClient(settings, http=httpx.Client(transport=httpx.MockTransport(handler)))
    client.complete(request())
    assert seen and all(v == 42.0 for v in seen[0].values())


def test_default_http_client_uses_the_setting():
    settings = ModelSettings.from_env(model_env(SCIFORGE_MODEL_TIMEOUT_SECONDS="75"))
    client = XAIClient(settings)                     # constructing an httpx.Client opens no connection
    try:
        t = client._http.timeout
        assert (t.connect, t.read, t.write, t.pool) == (75.0, 75.0, 75.0, 75.0)
    finally:
        client.close()
    default = XAIClient(ModelSettings.from_env(model_env()))
    try:
        assert default._http.timeout.read == 120.0
    finally:
        default.close()


# ------------------------------------------------------------------ timeouts still follow retry + accounting


def test_timeouts_count_as_attempts_and_follow_retry_and_accounting(caplog):
    caplog.set_level(logging.DEBUG, logger="sciforge")
    http_requests = []

    def handler(req: httpx.Request) -> httpx.Response:
        http_requests.append(req)
        raise httpx.ReadTimeout(f"slow; Authorization: Bearer {FAKE_XAI_KEY}")

    settings = ModelSettings.from_env(model_env(SCIFORGE_MODEL_TIMEOUT_SECONDS="7"))
    client = XAIClient(settings, http=httpx.Client(transport=httpx.MockTransport(handler)), now=lambda: NOW)
    tracker = BudgetTracker(BudgetLimits(max_spend_usd=10), PriceTable(input_per_mtok=2.0, output_per_mtok=10.0))
    audit = ModelCallAudit(store_prompts=True, secrets=settings.secret_values(), now=lambda: "2026-09-28T12:00:00Z")
    sleeper = Sleeper()
    with pytest.raises(ModelTimeout) as info:
        budgeted_call(client, request(), tracker, retry=RetryPolicy(max_retries=2, backoff_seconds=0.5), audit=audit,
                      sleep=sleeper, now=lambda: NOW)
    assert "timed out after 7s" in str(info.value)
    assert len(http_requests) == 3 and tracker.attempts == 3               # 1 attempt + 2 retries, all counted
    assert sleeper.calls == [0.5, 1.0]                                      # unchanged exponential backoff
    attempts = [e for e in audit.entries if e["entry_type"] == "attempt"]
    assert [e["outcome"] for e in attempts] == ["timeout"] * 3
    assert [e["will_retry"] for e in attempts] == [True, True, False]
    assert all(e["cost_source"] == "estimated_worst_case" for e in attempts)
    summary = tracker.summary()
    assert summary["used"]["failed_attempts"] == 3 and float(summary["used"]["spend_usd"]) > 0
    blob = json.dumps(audit.entries, default=str) + str(info.value) + repr(info.value) + caplog.text
    assert FAKE_XAI_KEY not in blob and "xai-FAKE" not in blob
    for req in http_requests:
        assert json.loads(req.content)["store"] is False                   # store=false kept


def test_timeout_then_success_is_accounted_like_before():
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectTimeout("connect slow")
        return json_response(responses_payload('{"answer": "ok"}'))

    settings = ModelSettings.from_env(model_env())
    client = XAIClient(settings, http=httpx.Client(transport=httpx.MockTransport(handler)), now=lambda: NOW)
    tracker = BudgetTracker(BudgetLimits(max_spend_usd=None))
    response = budgeted_call(client, request(), tracker, retry=RetryPolicy(max_retries=2, backoff_seconds=0.5),
                             sleep=Sleeper(), now=lambda: NOW)
    assert response.parsed == {"answer": "ok"} and tracker.attempts == 2
    assert tracker.summary()["used"]["failed_attempts"] == 1
