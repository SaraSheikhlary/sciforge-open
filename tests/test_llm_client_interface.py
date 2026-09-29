"""Model-client interface: dataclass invariants, error hierarchy, package API."""

import pytest

import sciforge.llm as llm
from sciforge.llm.client import (
    BudgetExhausted,
    ModelAuthError,
    ModelConfigError,
    ModelConnectionError,
    ModelError,
    ModelHTTPError,
    ModelIncompleteError,
    ModelMessage,
    ModelRateLimited,
    ModelRefusalError,
    ModelRequest,
    ModelResponse,
    ModelResponseParseError,
    ModelSchemaError,
    ModelTimeout,
    ModelUsage,
)


def test_message_validation():
    assert ModelMessage("user", "x").to_dict() == {"role": "user", "content": "x"}
    with pytest.raises(ValueError):
        ModelMessage("tool", "x")
    with pytest.raises(TypeError):
        ModelMessage("user", None)


def test_request_validation():
    m = (ModelMessage("user", "x"),)
    with pytest.raises(ValueError):
        ModelRequest(messages=(), max_output_tokens=10)
    with pytest.raises(ValueError):
        ModelRequest(messages=m, max_output_tokens=0)
    with pytest.raises(ValueError):
        ModelRequest(messages=m, max_output_tokens=True)
    with pytest.raises(ValueError):
        ModelRequest(messages=m, max_output_tokens=10, schema_name="only-name")
    with pytest.raises(ValueError):
        ModelRequest(messages=m, max_output_tokens=10, temperature=3)
    with pytest.raises(TypeError):
        ModelRequest(messages=({"role": "user", "content": "x"},), max_output_tokens=10)


def test_request_helpers():
    r = ModelRequest(messages=[ModelMessage("user", "abc")], max_output_tokens=10, instructions="12",
                     schema_name="s", json_schema={"type": "object"})
    assert isinstance(r.messages, tuple) and r.structured
    assert r.total_chars() == 3 + 2 + len('{"type":"object"}')
    r2 = r.with_max_output_tokens(5)
    assert r2.max_output_tokens == 5 and r2.messages == r.messages and r2.json_schema == r.json_schema


def test_usage_billable_output():
    assert ModelUsage().billable_output_tokens is None
    assert ModelUsage(output_tokens=10).billable_output_tokens == 10
    assert ModelUsage(output_tokens=10, reasoning_tokens=5).billable_output_tokens == 15
    assert ModelUsage(input_tokens=5, output_tokens=10, total_tokens=40).billable_output_tokens == 35
    assert ModelUsage(input_tokens=5, output_tokens=10, total_tokens=12).billable_output_tokens == 10


def test_response_to_dict_optional_text():
    r = ModelResponse(text="t", parsed=None, usage=ModelUsage(), model="m", latency_s=0.12345)
    assert r.to_dict()["text"] == "t" and "text" not in r.to_dict(include_text=False)
    assert r.to_dict()["latency_s"] == 0.123


def test_error_hierarchy():
    for cls in (ModelConfigError, ModelHTTPError, ModelTimeout, ModelConnectionError, ModelResponseParseError,
                ModelSchemaError, BudgetExhausted):
        assert issubclass(cls, ModelError)
    assert issubclass(ModelAuthError, ModelHTTPError) and issubclass(ModelRateLimited, ModelHTTPError)
    assert issubclass(ModelIncompleteError, ModelResponseParseError)
    assert issubclass(ModelRefusalError, ModelResponseParseError)
    assert not ModelConfigError("x").reached_provider
    assert not BudgetExhausted("max_model_calls", "d").reached_provider
    assert ModelHTTPError("x").reached_provider
    e = BudgetExhausted("max_spend_usd", "detail")
    assert e.limit == "max_spend_usd" and e.kind == "budget_exhausted" and "max_spend_usd" in str(e)
    assert ModelResponseParseError("m", kind="custom").kind == "custom"
    assert ModelAuthError("m", http_status=401).to_dict() == {"error_type": "auth_error", "message": "m",
                                                              "http_status": 401, "attempts": 1,
                                                              "retryable": False}


def test_retryable_defaults():
    assert ModelRateLimited("x", http_status=429).retryable
    assert ModelHTTPError("x", http_status=503).retryable
    assert ModelHTTPError("x", http_status=503).retry_reason == "http_error"
    assert not ModelHTTPError("x", http_status=400).retryable
    assert not ModelAuthError("x", http_status=401).retryable
    assert ModelTimeout("x").retryable and ModelConnectionError("x").retryable
    assert not ModelResponseParseError("x").retryable and not ModelSchemaError("x").retryable
    assert ModelTimeout("x").outcome == "timeout" and ModelSchemaError("x").outcome == "parse_error"
    assert ModelHTTPError("x", http_status=500, retry_after_s=3.0).retry_after_s == 3.0
    assert not ModelHTTPError("x", http_status=500, retryable=False).retryable


def test_usage_cost_coerced_to_decimal():
    from decimal import Decimal

    u = ModelUsage(cost_usd_reported=0.1)
    assert u.cost_usd_reported == Decimal("0.1") and u.cost_source == "reported"
    assert u.to_dict()["cost_usd_reported"] == "0.1"
    with pytest.raises(TypeError):
        ModelUsage(cost_usd_reported=True)


def test_package_exports_and_lazy_xai():
    for name in llm.__all__:
        assert getattr(llm, name) is not None
    from sciforge.llm.xai import XAIClient

    assert llm.XAIClient is XAIClient
    with pytest.raises(AttributeError):
        llm.DoesNotExist  # noqa: B018


def test_v02_modules_do_not_import_model_layer():
    import importlib
    import sys

    for mod in ("sciforge.pipeline", "sciforge.cli", "sciforge.verify"):
        importlib.import_module(mod)
    source = "".join(open(sys.modules[m].__file__).read() for m in ("sciforge.pipeline", "sciforge.cli"))
    assert "sciforge.llm" not in source and "ModelSettings" not in source
