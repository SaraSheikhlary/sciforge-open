"""Network/API failures: timeouts, connection errors, 5xx, 429, 404; retries and backoff."""

import httpx
import pytest

from conftest import FAKE_API_KEY, crossref_list_payload, json_response
from sciforge.http_utils import RateLimiter


def raising(exc_factory):
    def handler(request):
        raise exc_factory(request)
    return handler


def sequence(*responses):
    calls = iter(responses)

    def handler(request):
        item = next(calls)
        if isinstance(item, Exception):
            raise item
        return item
    return handler


@pytest.mark.parametrize("exc_factory, error_type", [
    (lambda r: httpx.ReadTimeout("read timed out", request=r), "timeout"),
    (lambda r: httpx.ConnectTimeout("connect timed out", request=r), "timeout"),
    (lambda r: httpx.ConnectError("connection refused", request=r), "connection_error"),
    (lambda r: httpx.RemoteProtocolError("peer closed", request=r), "transport_error"),
])
def test_transport_failures_are_retried_then_recorded(make_harness, sleeper, exc_factory, error_type):
    h = make_harness(raising(exc_factory))
    out = h.crossref.search("q", 3)
    assert out.status == "failed" and out.records == []
    assert len(h.requests) == 3  # 1 attempt + 2 retries
    assert sleeper.calls == [0.5, 1.0]  # exponential backoff
    err = h.run_log.errors[0]
    assert err.error_type == error_type and err.database == "crossref" and err.query == "q"
    assert err.http_status is None and "after 3 attempts" in err.message
    assert h.run_log.entries[0].attempts == 3


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_5xx_retried_then_recorded(make_harness, status):
    h = make_harness(lambda req: httpx.Response(status))
    out = h.pubmed.search("q", 3)
    assert out.status == "failed" and len(h.requests) == 3
    err = h.run_log.errors[0]
    assert err.error_type == "http_error" and err.http_status == status and err.stage == "search"


def test_5xx_then_success(make_harness, sleeper):
    h = make_harness(sequence(httpx.Response(503), json_response(crossref_list_payload([]))))
    out = h.crossref.search("q", 3)
    assert out.status == "ok" and h.run_log.errors == [] and len(h.requests) == 2
    assert h.run_log.entries[0].attempts == 2 and h.run_log.entries[0].status == "ok"


def test_429_uses_retry_after(make_harness, sleeper):
    h = make_harness(sequence(httpx.Response(429, headers={"Retry-After": "3"}),
                              httpx.Response(429, headers={"Retry-After": "999"}),
                              httpx.Response(429)))
    out = h.crossref.search("q", 3)
    assert out.status == "failed"
    assert sleeper.calls == [3.0, 30.0]  # capped at 30 s
    err = h.run_log.errors[0]
    assert err.error_type == "rate_limited" and err.http_status == 429


def test_404_not_retried(make_harness):
    h = make_harness(lambda req: httpx.Response(404))
    out = h.crossref.search("q", 3)
    assert out.status == "failed" and len(h.requests) == 1
    assert h.run_log.errors[0].error_type == "not_found" and h.run_log.errors[0].http_status == 404


def test_other_4xx_not_retried(make_harness):
    h = make_harness(lambda req: httpx.Response(400))
    h.pubmed.search("q", 3)
    assert len(h.requests) == 1 and h.run_log.errors[0].http_status == 400


def test_timeout_on_verification_is_lookup_failed(make_harness):
    h = make_harness(raising(lambda r: httpx.ReadTimeout("slow", request=r)))
    res = h.crossref.lookup("10.1/x")
    assert res.outcome == "lookup_failed" and res.error_type == "timeout"
    pm = h.pubmed.lookup_many(["1"])
    assert pm["1"].outcome == "lookup_failed"
    assert {e.stage for e in h.run_log.errors} == {"verification"}


def test_api_key_never_in_error_messages_or_log(make_harness):
    def handler(request):
        raise httpx.ConnectError(f"failed for {request.url}", request=request)
    h = make_harness(handler)
    h.pubmed.search("q", 3)
    dumped = str([e.model_dump() for e in h.run_log.entries]) + str([e.model_dump() for e in h.run_log.errors])
    assert "api_key" in str(h.requests[0].url)  # it was really sent...
    assert FAKE_API_KEY not in dumped  # ...but never logged


def test_rate_limiter_spacing():
    now = [0.0]
    sleeps = []

    def sleep(s):
        sleeps.append(round(s, 3))
        now[0] += s

    limiter = RateLimiter(0.34, sleep=sleep, clock=lambda: now[0])
    limiter.wait()
    limiter.wait()
    now[0] += 1.0
    limiter.wait()
    assert sleeps == [0.34]
