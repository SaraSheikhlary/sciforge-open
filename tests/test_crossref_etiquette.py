"""Crossref / NCBI etiquette: mailto, User-Agent, 429 + Retry-After, throttling, no concurrency."""

import threading
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import httpx
import pytest

from conftest import FAKE_EMAIL, Sleeper, api_handler, crossref_list_payload, crossref_work, crossref_work_payload, json_response, mock_client
from sciforge.config import Settings
from sciforge.crossref import CrossrefClient
from sciforge.http_utils import MAX_RETRY_AFTER_SECONDS, HttpFetcher, retry_after_seconds
from sciforge.logging_utils import RunLog
from sciforge.pipeline import run_investigation
from sciforge.pubmed import PubMedClient

NOW = datetime(2026, 9, 28, 23, 0, 0, tzinfo=timezone.utc)


class FakeClock:
    """Monotonic clock that only advances when sleep() is called."""

    def __init__(self):
        self.t = 100.0
        self.sleeps = []

    def clock(self):
        return self.t

    def sleep(self, seconds):
        self.sleeps.append(round(seconds, 6))
        self.t += seconds


def crossref_with(handler, settings, fake=None):
    fake = fake or FakeClock()
    requests = []

    def recording(request):
        requests.append(request)
        return handler(request)

    fetcher = HttpFetcher(mock_client(recording), settings, RunLog(settings.secret_values()),
                          sleep=fake.sleep, clock=fake.clock, now=lambda: NOW)
    return CrossrefClient(fetcher, settings), requests, fake


def ok_handler(request):
    if request.url.path == "/works":
        return json_response(crossref_list_payload([crossref_work()]))
    return json_response(crossref_work_payload(crossref_work()))


WITH_EMAIL = Settings(contact_email=FAKE_EMAIL, max_retries=2, backoff_seconds=1.0)
NO_EMAIL = Settings(max_retries=2, backoff_seconds=1.0)


# ------------------------------------------------------------ mailto and User-Agent

def test_mailto_on_search_and_verification_when_email_set():
    client, requests, _ = crossref_with(ok_handler, WITH_EMAIL)
    client.search("q", 3)
    client.lookup("10.1000/abc123")
    assert len(requests) == 2
    for req in requests:
        assert req.url.params["mailto"] == FAKE_EMAIL
        ua = req.headers["User-Agent"]
        assert ua == f"SciForge/0.4.0 (https://github.com/SaraSheikhlary/sciforge-open; mailto:{FAKE_EMAIL})"


def test_no_mailto_on_search_and_verification_when_email_unset():
    client, requests, _ = crossref_with(ok_handler, NO_EMAIL)
    client.search("q", 3)
    client.lookup("10.1000/abc123")
    for req in requests:
        assert "mailto" not in req.url.params
        assert req.headers["User-Agent"] == "SciForge/0.4.0 (https://github.com/SaraSheikhlary/sciforge-open)"


def test_pubmed_email_only_when_set(make_harness):
    for s, expected in ((WITH_EMAIL, FAKE_EMAIL), (NO_EMAIL, None)):
        h = make_harness(api_handler, s)
        h.pubmed.search("q", 2)
        h.pubmed.lookup_many(["111"])
        for req in h.requests:
            assert req.url.params.get("email") == expected
            assert req.url.params["tool"] == "sciforge"


def test_mailto_logged_redacted():
    client, _, _ = crossref_with(ok_handler, WITH_EMAIL)
    client.search("q", 3)
    entry = client.fetcher.run_log.entries[0]
    assert entry.params["mailto"] == "[REDACTED]" and FAKE_EMAIL not in entry.model_dump_json()


# ------------------------------------------------------------ 429 and Retry-After

def seq(*responses):
    it = iter(responses)
    return lambda request: next(it)


def test_429_retried_then_succeeds():
    client, requests, fake = crossref_with(
        seq(httpx.Response(429, headers={"Retry-After": "2"}), json_response(crossref_list_payload([]))), NO_EMAIL)
    out = client.search("q", 3)
    assert out.status == "ok" and len(requests) == 2
    assert 2.0 in fake.sleeps


def test_429_retry_after_seconds_is_respected_and_capped():
    client, requests, fake = crossref_with(
        seq(httpx.Response(429, headers={"Retry-After": "7"}), httpx.Response(429, headers={"Retry-After": "3600"}),
            httpx.Response(429, headers={"Retry-After": "1"})), NO_EMAIL)
    out = client.search("q", 3)
    assert out.status == "failed" and len(requests) == 3  # bounded: 1 + 2 retries
    backoff_sleeps = [s for s in fake.sleeps if s not in (0.2,)]
    assert backoff_sleeps[:2] == [7.0, MAX_RETRY_AFTER_SECONDS]
    err = client.fetcher.run_log.errors[0]
    assert err.error_type == "rate_limited" and err.http_status == 429


def test_429_retry_after_http_date_is_respected():
    in_10s = format_datetime(NOW + timedelta(seconds=10), usegmt=True)
    client, _, fake = crossref_with(
        seq(httpx.Response(429, headers={"Retry-After": in_10s}), json_response(crossref_list_payload([]))), NO_EMAIL)
    assert client.search("q", 3).status == "ok"
    assert 10.0 in fake.sleeps


def test_429_without_retry_after_uses_exponential_backoff():
    client, _, fake = crossref_with(seq(httpx.Response(429), httpx.Response(429), httpx.Response(429)), NO_EMAIL)
    client.search("q", 3)
    assert [s for s in fake.sleeps if s != 0.2][:2] == [1.0, 2.0]


@pytest.mark.parametrize("header, expected", [
    ("5", 5.0), (" 12 ", 12.0), ("0", 0.0), ("-3", 0.0), ("100000", MAX_RETRY_AFTER_SECONDS),
    (format_datetime(NOW + timedelta(seconds=20), usegmt=True), 20.0),
    (format_datetime(NOW + timedelta(days=1), usegmt=True), MAX_RETRY_AFTER_SECONDS),
    (format_datetime(NOW - timedelta(minutes=5), usegmt=True), 0.0),
    ("soon", None), ("", None), ("nan", None),
])
def test_retry_after_parsing(header, expected):
    assert retry_after_seconds(httpx.Response(429, headers={"Retry-After": header}), NOW) == expected


def test_retry_after_absent():
    assert retry_after_seconds(httpx.Response(429), NOW) is None


# ------------------------------------------------------------ throttling and sequential requests

def test_default_throttle_intervals():
    for s, pm, cr in ((Settings(), 0.34, 0.2), (Settings(ncbi_api_key="k" * 10, contact_email=FAKE_EMAIL), 0.11, 0.1)):
        fetcher = HttpFetcher(mock_client(api_handler), s, RunLog())
        assert PubMedClient(fetcher, s).limiter.min_interval == pm
        assert CrossrefClient(fetcher, s).limiter.min_interval == cr


def test_crossref_throttle_honoured_between_requests():
    client, requests, fake = crossref_with(ok_handler, NO_EMAIL)
    for doi in ("10.1/a", "10.1/b", "10.1/c"):
        client.lookup(doi)
    assert len(requests) == 3
    assert fake.sleeps == [0.2, 0.2]  # clock never advances on its own, so full interval each time


def test_throttle_applies_to_retries():
    s = Settings(max_retries=2, backoff_seconds=0.0)
    client, requests, fake = crossref_with(lambda r: httpx.Response(503), s)
    client.search("q", 3)
    assert len(requests) == 3
    assert [x for x in fake.sleeps if x > 0] == [0.2, 0.2]


def test_pubmed_throttle_honoured():
    fake = FakeClock()
    s = Settings()
    fetcher = HttpFetcher(mock_client(api_handler), s, RunLog(), sleep=fake.sleep, clock=fake.clock)
    PubMedClient(fetcher, s).search("q", 2)  # esearch + esummary
    assert fake.sleeps == [0.34]


def test_requests_are_sequential_no_concurrency(tmp_path):
    lock = threading.Lock()
    state = {"in_flight": 0, "max": 0, "threads": set()}

    def handler(request):
        with lock:
            state["in_flight"] += 1
            state["max"] = max(state["max"], state["in_flight"])
            state["threads"].add(threading.get_ident())
        try:
            return api_handler(request)
        finally:
            with lock:
                state["in_flight"] -= 1

    sleeper = Sleeper()
    threads_before = threading.active_count()
    result = run_investigation("q", output_dir=tmp_path, settings=Settings(), client=mock_client(handler),
                               sleep=sleeper)
    assert state["max"] == 1
    assert state["threads"] == {threading.get_ident()}  # all requests on the calling thread
    assert threading.active_count() == threads_before
    assert result.summary["errors_count"] == 0
    assert sleeper.calls, "throttle should have requested waits between requests"
