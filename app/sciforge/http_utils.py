"""HTTP fetching with throttling, bounded retries, and structured failures.

``HttpFetcher.get_json`` never raises for network, HTTP, or JSON problems; it
returns a :class:`FetchResult` describing what happened and logs the request
(with secrets redacted) to the run log.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from sciforge.config import Settings
from sciforge.logging_utils import RunLog, iso_utc, redact_params, redact_url, utc_now
from sciforge.models import ErrorEntry, RequestLogEntry

MAX_RETRY_AFTER_SECONDS = 30.0

Sleep = Callable[[float], None]
Clock = Callable[[], float]


class MalformedResponseError(ValueError):
    """The response JSON did not have the expected structure."""


class RateLimiter:
    """Enforces a minimum interval between consecutive requests."""

    def __init__(self, min_interval: float, sleep: Sleep = time.sleep, clock: Clock = time.monotonic) -> None:
        self.min_interval = max(0.0, min_interval)
        self.sleep = sleep
        self._clock = clock
        self._last: float | None = None

    def wait(self) -> None:
        """Sleep just long enough to respect ``min_interval``."""
        if self._last is not None and self.min_interval > 0:
            elapsed = self._clock() - self._last
            if elapsed < self.min_interval:
                self.sleep(self.min_interval - elapsed)
        self._last = self._clock()


@dataclass
class FetchResult:
    """Outcome of one logical request (after retries)."""

    ok: bool
    http_status: int | None
    data: Any = None
    error_type: str | None = None
    message: str | None = None
    attempts: int = 1
    entry: RequestLogEntry | None = None

    @property
    def not_found(self) -> bool:
        """True for an HTTP 404 response."""
        return self.http_status == 404

    def to_error(self, database: str, stage: str, query: str | None) -> ErrorEntry:
        """Convert a failed result to an :class:`ErrorEntry`."""
        return ErrorEntry(
            database=database,
            stage=stage,
            query=query,
            timestamp=self.entry.timestamp if self.entry else iso_utc(),
            error_type=self.error_type or "unknown_error",
            http_status=self.http_status,
            message=self.message or "request failed",
            url=self.entry.url if self.entry else None,
        )


def retry_after_seconds(response: httpx.Response, now: datetime | None = None) -> float | None:
    """Delay requested by a ``Retry-After`` header, clamped to [0, 30] seconds.

    Supports both forms from RFC 9110: delay-seconds (``"120"``) and an
    HTTP-date (``"Wed, 21 Oct 2015 07:28:00 GMT"``). Returns None when the
    header is absent or unparseable (the caller then uses exponential backoff).
    """
    raw = response.headers.get("Retry-After")
    if raw is None or not raw.strip():
        return None
    raw = raw.strip()
    try:
        seconds = float(raw)
    except ValueError:
        try:
            when = parsedate_to_datetime(raw)
        except (TypeError, ValueError, IndexError):
            return None
        if when is None:
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        seconds = (when - (now or utc_now())).total_seconds()
    if seconds != seconds:  # NaN
        return None
    return min(max(seconds, 0.0), MAX_RETRY_AFTER_SECONDS)


def is_retryable_status(status: int) -> bool:
    """True for HTTP statuses that are retried: 429 and every 5xx."""
    return status == 429 or status >= 500


def compute_backoff(
    base_seconds: float,
    attempt: int,
    response: httpx.Response | None = None,
    now: datetime | None = None,
) -> float:
    """Delay before retry number ``attempt + 1``.

    Uses the response's ``Retry-After`` (clamped to [0, 30] s) when present,
    otherwise ``base_seconds * 2**attempt``. Shared by :class:`HttpFetcher`
    and the v0.3 model client so both follow identical retry semantics.
    """
    if response is not None:
        retry_after = retry_after_seconds(response, now)
        if retry_after is not None:
            return retry_after
    return base_seconds * (2**attempt)


class HttpFetcher:
    """Performs GET requests returning JSON, with retries and logging."""

    def __init__(
        self,
        client: httpx.Client,
        settings: Settings,
        run_log: RunLog,
        *,
        sleep: Sleep = time.sleep,
        now: Callable[[], datetime] = utc_now,
        clock: Clock = time.monotonic,
    ) -> None:
        self.client = client
        self.settings = settings
        self.run_log = run_log
        self.sleep = sleep
        self.now = now
        self.clock = clock

    def make_limiter(self, min_interval: float) -> RateLimiter:
        """A rate limiter sharing this fetcher's sleep and clock functions."""
        return RateLimiter(min_interval, sleep=self.sleep, clock=self.clock)

    def _backoff(self, attempt: int, response: httpx.Response | None = None) -> float:
        now = self.now() if response is not None else None
        return compute_backoff(self.settings.backoff_seconds, attempt, response, now)

    def _attempt(self, url: str, params: Mapping[str, Any]) -> tuple[FetchResult, bool, httpx.Response | None]:
        """One attempt. Returns (result, retryable, response)."""
        try:
            response = self.client.get(
                url,
                params=dict(params),
                headers={"User-Agent": self.settings.user_agent, "Accept": "application/json"},
                timeout=self.settings.timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            return FetchResult(False, None, error_type="timeout", message=f"request timed out ({type(exc).__name__})"), True, None
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            return FetchResult(False, None, error_type="connection_error", message=f"connection failed ({type(exc).__name__}: {exc})"), True, None
        except httpx.TransportError as exc:
            return FetchResult(False, None, error_type="transport_error", message=f"transport error ({type(exc).__name__}: {exc})"), True, None
        except httpx.HTTPError as exc:
            return FetchResult(False, None, error_type="request_error", message=f"request error ({type(exc).__name__}: {exc})"), False, None
        except Exception as exc:  # noqa: BLE001 - a run must never crash on one request
            return FetchResult(False, None, error_type="unexpected_error", message=f"unexpected error ({type(exc).__name__}: {exc})"), False, None

        status = response.status_code
        if status == 429:
            return FetchResult(False, status, error_type="rate_limited", message="HTTP 429 Too Many Requests"), True, response
        if status >= 500:
            return FetchResult(False, status, error_type="http_error", message=f"HTTP {status} server error"), True, response
        if status == 404:
            return FetchResult(False, status, error_type="not_found", message="HTTP 404 Not Found"), False, response
        if status >= 400:
            return FetchResult(False, status, error_type="http_error", message=f"HTTP {status} client error"), False, response
        if status >= 300:
            return FetchResult(False, status, error_type="http_error", message=f"unexpected HTTP {status} redirect"), False, response
        try:
            data = response.json()
        except ValueError:
            return FetchResult(False, status, error_type="invalid_json", message="response body is not valid JSON"), False, response
        return FetchResult(True, status, data=data), False, response

    def get_json(
        self,
        *,
        database: str,
        stage: str,
        url: str,
        params: Mapping[str, Any] | None = None,
        query: str | None = None,
        limiter: RateLimiter | None = None,
    ) -> FetchResult:
        """GET ``url`` and decode JSON, retrying transient failures.

        Retries (up to ``settings.max_retries``) on 429, 5xx, timeouts, and
        connection/transport errors, sleeping ``backoff * 2**attempt`` seconds.
        When a retried response carries ``Retry-After`` (seconds or HTTP-date)
        that delay is used instead, clamped to [0, 30] s. 404 and other 4xx
        are not retried. Requests are strictly sequential (no concurrency).
        """
        params = dict(params or {})
        started = iso_utc()
        attempts = 0
        result: FetchResult
        while True:
            if limiter is not None:
                limiter.wait()
            result, retryable, response = self._attempt(url, params)
            attempts += 1
            if result.ok or not retryable or attempts > self.settings.max_retries:
                break
            self.sleep(self._backoff(attempts - 1, response))

        result.attempts = attempts
        if result.message:
            result.message = self.run_log.scrub(result.message)
        if attempts > 1 and not result.ok:
            result.message = f"{result.message} (after {attempts} attempts)"
        status = "ok" if result.ok else ("not_found" if result.not_found else "error")
        entry = RequestLogEntry(
            database=database,
            stage=stage,
            query=query,
            url=redact_url(url),
            params=redact_params(params),
            timestamp=started,
            status=status,
            http_status=result.http_status,
            attempts=attempts,
            error_type=result.error_type,
            message=result.message,
        )
        result.entry = self.run_log.add_entry(entry)
        return result
