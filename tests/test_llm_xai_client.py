"""XAIClient against httpx.MockTransport only (never the real API; D7)."""

import json
import logging
from datetime import datetime, timezone
from decimal import Decimal

import httpx
import pytest

from conftest import FAKE_XAI_KEY, FAKE_XAI_MODEL, TEST_XAI_BASE, Sleeper, json_response, model_env, responses_payload
from sciforge.config import ModelSettings
from sciforge.llm.audit import ModelCallAudit
from sciforge.llm.budget import BudgetLimits, BudgetTracker, PriceTable, budgeted_call
from sciforge.llm.client import (
    ModelAuthError,
    ModelClient,
    ModelConnectionError,
    ModelError,
    ModelHTTPError,
    ModelIncompleteError,
    ModelMessage,
    ModelRateLimited,
    ModelRefusalError,
    ModelRequest,
    ModelResponseParseError,
    ModelTimeout,
)
from sciforge.llm.xai import XAIClient, build_request_body

SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"],
          "additionalProperties": False}
NOW = datetime(2026, 9, 28, 12, 0, 0, tzinfo=timezone.utc)


def make_request(**kw):
    base = dict(messages=(ModelMessage("system", "Be precise."), ModelMessage("user", "Question?")),
                max_output_tokens=500, schema_name="answer", json_schema=SCHEMA)
    base.update(kw)
    return ModelRequest(**base)


class Rig:
    def __init__(self, handler, **env):
        self.requests: list[httpx.Request] = []
        self.sleeper = Sleeper()

        def recording(request):
            self.requests.append(request)
            return handler(request)

        env.setdefault("XAI_BASE_URL", TEST_XAI_BASE)
        env.setdefault("SCIFORGE_BACKOFF_SECONDS", "0.5")
        self.settings = ModelSettings.from_env(model_env(**env))
        self.client = XAIClient(self.settings, http=httpx.Client(transport=httpx.MockTransport(recording)),
                                sleep=self.sleeper, now=lambda: NOW)

    def body(self, i=0):
        return json.loads(self.requests[i].content)


def ok_handler(request):
    return json_response(responses_payload())


def sequence(*responses):
    items = list(responses)

    def handler(request):
        item = items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return handler


# ------------------------------------------------------------------ request shape

def test_request_shape_endpoint_headers_and_body():
    rig = Rig(ok_handler)
    resp = rig.client.complete(make_request(instructions="System instructions."))
    req = rig.requests[0]
    assert req.method == "POST"
    assert str(req.url) == f"{TEST_XAI_BASE}/v1/responses"
    assert req.headers["Authorization"] == f"Bearer {FAKE_XAI_KEY}"
    assert req.headers["Content-Type"] == "application/json"
    assert req.headers["User-Agent"].startswith("SciForge/")
    body = rig.body()
    assert body["model"] == FAKE_XAI_MODEL          # from config (XAI_MODEL), never hard-coded
    assert body["store"] is False
    assert body["input"] == [{"role": "system", "content": "Be precise."}, {"role": "user", "content": "Question?"}]
    assert body["instructions"] == "System instructions."
    assert body["max_output_tokens"] == 500
    assert body["temperature"] == 0.0
    assert body["text"] == {"format": {"type": "json_schema", "name": "answer", "schema": SCHEMA, "strict": True}}
    assert "previous_response_id" not in body and "tools" not in body
    assert "messages" not in body and "response_format" not in body   # not Chat Completions
    assert FAKE_XAI_KEY not in req.content.decode()
    assert resp.parsed == {"answer": "ok"} and resp.provider == "xai"
    assert resp.response_id == "resp_test_1" and resp.model == FAKE_XAI_MODEL and resp.http_status == 200


def test_default_base_url_is_api_x_ai():
    seen = []
    client = XAIClient(ModelSettings.from_env(model_env()), http=httpx.Client(transport=httpx.MockTransport(
        lambda r: seen.append(r) or json_response(responses_payload()))))
    client.complete(make_request())
    assert str(seen[0].url) == "https://api.x.ai/v1/responses"   # MockTransport only; no real request


def test_store_false_on_every_request_including_repairs_and_unstructured():
    rig = Rig(lambda r: json_response(responses_payload('{"answer": "x"}')))
    first = make_request()
    rig.client.complete(first)
    repair = make_request(messages=first.messages + (ModelMessage("assistant", "{bad"),
                                                     ModelMessage("user", "Return corrected JSON.")))
    rig.client.complete(repair)
    rig.client.complete(make_request(schema_name=None, json_schema=None))
    bodies = [rig.body(i) for i in range(3)]
    assert all(b["store"] is False for b in bodies)
    assert all("previous_response_id" not in b for b in bodies)
    assert len(bodies[1]["input"]) == 4                  # full local history resent
    assert "text" not in bodies[2]


def test_build_request_body_is_pure_and_keyless():
    body = build_request_body("m", make_request())
    assert body["store"] is False and body["model"] == "m"
    assert "Authorization" not in json.dumps(body)


def test_protocol_conformance_and_repr_hides_key():
    rig = Rig(ok_handler)
    assert isinstance(rig.client, ModelClient)
    assert FAKE_XAI_KEY not in repr(rig.client)
    assert rig.client.model == FAKE_XAI_MODEL


# ------------------------------------------------------------------ single attempt + budgeted retries

def run(rig, request=None, tracker=None, audit=None):
    """One logical call through the budgeted retry loop (the only place retries happen)."""
    tracker = tracker or BudgetTracker(BudgetLimits(max_spend_usd=None), PriceTable())
    return budgeted_call(rig.client, request or make_request(), tracker, retry=rig.settings.retry_policy(),
                         audit=audit, sleep=rig.sleeper, now=lambda: NOW)


def test_client_complete_never_retries_itself():
    rig = Rig(lambda r: httpx.Response(503))
    with pytest.raises(ModelHTTPError) as info:
        rig.client.complete(make_request())
    assert len(rig.requests) == 1 and rig.sleeper.calls == []
    assert info.value.retryable and info.value.retry_reason == "http_5xx"


def test_429_with_retry_after_seconds_then_success():
    rig = Rig(sequence(httpx.Response(429, headers={"Retry-After": "7"}), json_response(responses_payload())))
    resp = run(rig)
    assert resp.attempts == 2 and rig.sleeper.calls == [7.0]


def test_429_retry_after_http_date_and_clamp():
    rig = Rig(sequence(httpx.Response(429, headers={"Retry-After": "Mon, 28 Sep 2026 12:00:05 GMT"}),
                       httpx.Response(429, headers={"Retry-After": "999"}),
                       json_response(responses_payload())))
    run(rig)
    assert rig.sleeper.calls == [5.0, 30.0]


def test_429_exhausted_raises_rate_limited():
    rig = Rig(lambda r: httpx.Response(429, json={"error": {"message": "slow down"}}))
    tracker = BudgetTracker(BudgetLimits(max_spend_usd=None), PriceTable())
    with pytest.raises(ModelRateLimited) as info:
        run(rig, tracker=tracker)
    assert info.value.http_status == 429 and info.value.attempts == 3
    assert "slow down" in info.value.message
    assert rig.sleeper.calls == [0.5, 1.0]            # exponential backoff without Retry-After
    assert tracker.attempts == 3                      # every attempt counted


def test_5xx_retried_then_success():
    rig = Rig(sequence(httpx.Response(503), httpx.Response(500), json_response(responses_payload())))
    assert run(rig).attempts == 3
    assert len(rig.requests) == 3


def test_5xx_exhausted():
    rig = Rig(lambda r: httpx.Response(502), SCIFORGE_MAX_RETRIES="1")
    with pytest.raises(ModelHTTPError) as info:
        run(rig)
    assert info.value.http_status == 502 and len(rig.requests) == 2


def test_timeout_retried_then_raises_model_timeout():
    rig = Rig(lambda r: (_ for _ in ()).throw(httpx.ReadTimeout("timed out", request=r)))
    with pytest.raises(ModelTimeout) as info:
        run(rig)
    assert info.value.attempts == 3 and len(rig.requests) == 3


def test_timeout_then_success():
    rig = Rig(sequence(httpx.ConnectTimeout("t"), json_response(responses_payload())))
    assert run(rig).attempts == 2


def test_connection_error():
    rig = Rig(lambda r: (_ for _ in ()).throw(httpx.ConnectError("refused", request=r)), SCIFORGE_MAX_RETRIES="0")
    with pytest.raises(ModelConnectionError):
        run(rig)


@pytest.mark.parametrize("status", [401, 403])
def test_auth_errors_not_retried(status):
    rig = Rig(lambda r: httpx.Response(status, json={"error": {"message": f"bad key {FAKE_XAI_KEY}"}}))
    with pytest.raises(ModelAuthError) as info:
        run(rig)
    assert len(rig.requests) == 1 and rig.sleeper.calls == []
    assert info.value.http_status == status
    assert FAKE_XAI_KEY not in str(info.value)          # echoed key scrubbed


@pytest.mark.parametrize("status", [400, 404, 422])
def test_client_errors_not_retried(status):
    rig = Rig(lambda r: httpx.Response(status, json={"error": {"message": "invalid schema"}}))
    with pytest.raises(ModelHTTPError) as info:
        run(rig)
    assert not isinstance(info.value, (ModelAuthError, ModelRateLimited))
    assert len(rig.requests) == 1 and "invalid schema" in info.value.message


def test_redirect_not_followed():
    rig = Rig(lambda r: httpx.Response(307, headers={"Location": "https://evil.example/"}))
    with pytest.raises(ModelHTTPError, match="redirect"):
        rig.client.complete(make_request())
    assert len(rig.requests) == 1


# ------------------------------------------------------------------ response parsing

def test_malformed_json_body():
    rig = Rig(lambda r: httpx.Response(200, content=b"<html>oops</html>"))
    with pytest.raises(ModelResponseParseError) as info:
        rig.client.complete(make_request())
    assert info.value.kind == "invalid_json_envelope" and len(rig.requests) == 1


def test_missing_output_text():
    rig = Rig(lambda r: json_response(responses_payload(None)))
    with pytest.raises(ModelResponseParseError) as info:
        rig.client.complete(make_request())
    assert info.value.kind == "missing_output_text"
    assert info.value.usage is not None and info.value.usage.input_tokens == 120   # billed usage kept


def test_incomplete_response():
    payload = responses_payload('{"answer": "trunc', status="incomplete",
                                incomplete_details={"reason": "max_output_tokens"})
    rig = Rig(lambda r: json_response(payload))
    with pytest.raises(ModelIncompleteError) as info:
        rig.client.complete(make_request())
    assert info.value.reason == "max_output_tokens" and info.value.http_status == 200
    assert info.value.usage.output_tokens == 30


def test_refusal():
    payload = responses_payload(None, extra_output=[{"type": "message", "role": "assistant",
                                                     "content": [{"type": "refusal", "refusal": "I can't"}]}])
    rig = Rig(lambda r: json_response(payload))
    with pytest.raises(ModelRefusalError):
        rig.client.complete(make_request())


def test_output_not_json_when_schema_requested():
    rig = Rig(lambda r: json_response(responses_payload("Sure! Here you go.")))
    with pytest.raises(ModelResponseParseError) as info:
        rig.client.complete(make_request())
    assert info.value.kind == "invalid_output_json" and info.value.raw_text == "Sure! Here you go."


def test_unstructured_request_returns_text_without_parsing():
    rig = Rig(lambda r: json_response(responses_payload("plain text")))
    resp = rig.client.complete(make_request(schema_name=None, json_schema=None))
    assert resp.text == "plain text" and resp.parsed is None


def test_usage_parsing_with_reasoning_cached_and_cost():
    usage = {"input_tokens": 32, "output_tokens": 9, "total_tokens": 151,
             "input_tokens_details": {"cached_tokens": 8}, "output_tokens_details": {"reasoning_tokens": 110},
             "cost_in_usd_ticks": 25_000_000}
    rig = Rig(lambda r: json_response(responses_payload(usage=usage)))
    u = rig.client.complete(make_request()).usage
    assert (u.input_tokens, u.output_tokens, u.reasoning_tokens, u.cached_input_tokens, u.total_tokens) == \
        (32, 9, 110, 8, 151)
    assert u.cost_usd_reported == Decimal("0.0025") and u.cost_source == "reported_ticks"
    assert u.billable_output_tokens == 119             # reasoning included


def test_missing_usage_tolerated():
    payload = responses_payload()
    del payload["usage"]
    rig = Rig(lambda r: json_response(payload))
    u = rig.client.complete(make_request()).usage
    assert u.reported is False and u.input_tokens is None


def test_reasoning_items_ignored_and_multiple_text_parts_joined():
    payload = responses_payload(None, extra_output=[
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "thinking"}]},
        {"type": "message", "role": "assistant", "content": [
            {"type": "output_text", "text": '{"answer": '}, {"type": "output_text", "text": '"joined"}'}]}])
    rig = Rig(lambda r: json_response(payload))
    assert rig.client.complete(make_request()).parsed == {"answer": "joined"}


# ------------------------------------------------------------------ secrets in logs / audit

def test_no_authorization_or_key_in_logs_or_audit(caplog, tmp_path):
    rig = Rig(sequence(httpx.Response(500), json_response(responses_payload())))
    audit = ModelCallAudit(store_prompts=True, secrets=rig.settings.secret_values())
    tracker = BudgetTracker(BudgetLimits(max_spend_usd=None), PriceTable())
    with caplog.at_level(logging.DEBUG, logger="sciforge"):
        run(rig, tracker=tracker, audit=audit)
        rig2 = Rig(lambda r: httpx.Response(401, json={"error": {"message": f"Bearer {FAKE_XAI_KEY} rejected"}}))
        with pytest.raises(ModelAuthError):
            run(rig2, tracker=tracker, audit=audit)
    path = audit.write(tmp_path / "model_calls.json")
    text = path.read_text() + caplog.text
    assert FAKE_XAI_KEY not in text
    assert "Authorization" not in path.read_text()
    assert "Bearer" not in path.read_text() or "Bearer [REDACTED]" in path.read_text()
    assert len(audit.entries) == 3                    # 500 attempt + retry success + 401 attempt


def test_errors_never_contain_key_or_header():
    rig = Rig(lambda r: httpx.Response(400, json={"error": {"message": f"Authorization: Bearer {FAKE_XAI_KEY}"}}))
    with pytest.raises(ModelError) as info:
        rig.client.complete(make_request())
    assert FAKE_XAI_KEY not in str(info.value) and FAKE_XAI_KEY not in repr(info.value.to_dict())
