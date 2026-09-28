"""Shared fixtures. Every test runs with real network access disabled."""

from __future__ import annotations

import json
import socket
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from sciforge.config import Settings
from sciforge.crossref import CrossrefClient
from sciforge.http_utils import HttpFetcher, RateLimiter
from sciforge.logging_utils import RunLog
from sciforge.pubmed import PubMedClient

FAKE_API_KEY = "test-ncbi-key-0000000000"
FAKE_EMAIL = "tester@example.org"

Handler = Callable[[httpx.Request], httpx.Response]


class NetworkBlockedError(RuntimeError):
    """Raised when a test attempts a real network connection."""


@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test that tries to open a real connection."""

    def blocked(*args: Any, **kwargs: Any) -> Any:
        raise NetworkBlockedError("real network access is disabled in tests")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", blocked)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never see real credentials from the developer's environment."""
    for name in ("NCBI_API_KEY", "SCIFORGE_CONTACT_EMAIL", "SCIFORGE_TIMEOUT_SECONDS",
                 "SCIFORGE_MAX_RETRIES", "SCIFORGE_BACKOFF_SECONDS", "XAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def settings() -> Settings:
    return Settings(ncbi_api_key=FAKE_API_KEY, contact_email=FAKE_EMAIL, timeout_seconds=1.0,
                    max_retries=2, backoff_seconds=0.5)


class Sleeper:
    """Records requested sleeps instead of sleeping."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


@pytest.fixture
def sleeper() -> Sleeper:
    return Sleeper()


def mock_client(handler: Handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def json_response(payload: Any, status: int = 200) -> httpx.Response:
    return httpx.Response(status, content=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})


class Harness:
    """Bundles a mock client, fetcher, run log, and both API clients."""

    def __init__(self, handler: Handler, settings: Settings, sleeper: Sleeper) -> None:
        self.requests: list[httpx.Request] = []

        def recording(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return handler(request)

        self.client = mock_client(recording)
        self.run_log = RunLog(settings.secret_values())
        self.fetcher = HttpFetcher(self.client, settings, self.run_log, sleep=sleeper)
        no_throttle = RateLimiter(0, sleep=sleeper)
        self.pubmed = PubMedClient(self.fetcher, settings, limiter=no_throttle)
        self.crossref = CrossrefClient(self.fetcher, settings, limiter=no_throttle)


@pytest.fixture
def make_harness(settings: Settings, sleeper: Sleeper) -> Callable[..., Harness]:
    def factory(handler: Handler, custom_settings: Settings | None = None) -> Harness:
        return Harness(handler, custom_settings or settings, sleeper)

    return factory


# ---------------------------------------------------------------- sample payloads

def esearch_payload(ids: list[str], count: int | None = None) -> dict[str, Any]:
    return {"header": {"type": "esearch"},
            "esearchresult": {"count": str(count if count is not None else len(ids)), "retmax": str(len(ids)),
                              "idlist": ids}}


def esummary_doc(pmid: str, *, title: str = "Shear stress activates platelets.", doi: str | None = "10.1000/abc123",
                 pubdate: str = "2021 Mar 15", authors: list[str] | None = None,
                 journal: str = "Journal of Thrombosis") -> dict[str, Any]:
    articleids = [{"idtype": "pubmed", "idtypen": 1, "value": pmid}]
    if doi:
        articleids.append({"idtype": "doi", "idtypen": 3, "value": doi})
    return {
        "uid": pmid,
        "pubdate": pubdate,
        "sortpubdate": "2021/03/15 00:00",
        "source": "J Thromb",
        "fulljournalname": journal,
        "title": title,
        "authors": [{"name": n, "authtype": "Author"} for n in (authors or ["Smith JA", "Lee K"])],
        "articleids": articleids,
    }


def esummary_payload(docs: list[dict[str, Any]], missing: list[str] = ()) -> dict[str, Any]:
    result: dict[str, Any] = {"uids": [d["uid"] for d in docs] + list(missing)}
    for doc in docs:
        result[doc["uid"]] = doc
    for uid in missing:
        result[uid] = {"uid": uid, "error": "cannot get document summary"}
    return {"header": {"type": "esummary"}, "result": result}


def crossref_work(doi: str = "10.1000/ABC123", *, title: str = "Shear stress activates platelets",
                  year: int | None = 2021, authors: list[tuple[str, str]] | None = None,
                  journal: str = "Journal of Thrombosis") -> dict[str, Any]:
    work: dict[str, Any] = {
        "DOI": doi,
        "title": [title],
        "container-title": [journal],
        "author": [{"family": f, "given": g, "sequence": "first"} for f, g in (authors or [("Smith", "John A."), ("Lee", "Kim")])],
        "URL": f"http://dx.doi.org/{doi}",
    }
    if year is not None:
        work["issued"] = {"date-parts": [[year, 3, 15]]}
    return work


def crossref_list_payload(items: list[Any], total: int | None = None) -> dict[str, Any]:
    return {"status": "ok", "message-type": "work-list",
            "message": {"total-results": total if total is not None else len(items), "items": items}}


def crossref_work_payload(work: dict[str, Any]) -> dict[str, Any]:
    return {"status": "ok", "message-type": "work", "message": work}


# ---------------------------------------------------------------- full mocked API

def api_handler(request: httpx.Request) -> httpx.Response:
    """Mock of both APIs: 2 PubMed hits, 2 Crossref hits (one DOI shared), DOI lookups."""
    path = request.url.path
    if path.endswith("esearch.fcgi"):
        return json_response(esearch_payload(["111", "222"]))
    if path.endswith("esummary.fcgi"):
        return json_response(esummary_payload([
            esummary_doc("111", doi="10.1000/abc123"),
            esummary_doc("222", doi=None, title="Another platelet paper", authors=["Doe J"]),
        ]))
    if path == "/works":
        return json_response(crossref_list_payload([
            crossref_work("10.1000/ABC123"),
            crossref_work("10.3000/other", title="Unrelated coral study", authors=[("Reef", "A")]),
        ]))
    if path == "/works/10.1000/abc123":
        return json_response(crossref_work_payload(crossref_work("10.1000/ABC123")))
    if path == "/works/10.3000/other":
        return json_response(crossref_work_payload(
            crossref_work("10.3000/other", title="Unrelated coral study", authors=[("Reef", "A")])))
    return httpx.Response(500)


@pytest.fixture
def fast_sleep(monkeypatch: pytest.MonkeyPatch) -> Sleeper:
    """Replace time.sleep (used by the pipeline's throttling/backoff) with a recorder."""
    import time

    recorder = Sleeper()
    monkeypatch.setattr(time, "sleep", recorder)
    return recorder
