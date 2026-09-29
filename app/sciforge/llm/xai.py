"""xAI Responses API client (``POST {XAI_BASE_URL}/v1/responses``; decision D1).

* ``store`` is always ``false`` (no server-side storage of prompts/outputs).
* ``previous_response_id`` is never sent: every request carries the full local
  message history, so no server-side conversation state is relied upon.
* Structured output uses ``text.format`` = ``{"type": "json_schema", "name",
  "schema", "strict": true}``.
* ``complete()`` makes exactly ONE HTTP attempt. Retries (429, 5xx, timeouts,
  transport errors; never 401/403/other 4xx) are driven by
  ``sciforge.llm.budget.budgeted_call`` so that EVERY attempt is pre-checked,
  reserved and recorded against the investigation budget. Backoff semantics
  match v0.2 (``backoff * 2**n`` or ``Retry-After`` clamped to [0, 30] s).
* The API key is only placed in the per-request ``Authorization`` header; it is
  never logged, returned, or included in exception messages.

This module performs no network I/O at import time, and tests only ever use it
with ``httpx.MockTransport``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

import httpx

from sciforge.config import ModelSettings
from sciforge.http_utils import is_retryable_status, retry_after_seconds
from sciforge.llm.client import (
    ModelAuthError,
    ModelConnectionError,
    ModelError,
    ModelHTTPError,
    ModelRateLimited,
    ModelRequest,
    ModelResponse,
    ModelResponseParseError,
    ModelTimeout,
    _Timer,
)
from sciforge.llm.parsing import extract_output_text, parse_json_text, parse_usage
from sciforge.logging_utils import get_logger, redact_text, utc_now

_MAX_ERROR_SNIPPET = 200


def build_request_body(model: str, request: ModelRequest) -> dict[str, Any]:
    """The exact JSON body sent to ``/v1/responses`` (no credentials inside)."""
    body: dict[str, Any] = {
        "model": model,
        "input": [m.to_dict() for m in request.messages],
        "max_output_tokens": request.max_output_tokens,
        "temperature": request.temperature,
        "store": False,  # D1: never store on the provider side
    }
    if request.instructions:
        body["instructions"] = request.instructions
    if request.json_schema is not None:
        body["text"] = {
            "format": {
                "type": "json_schema",
                "name": request.schema_name,
                "schema": request.json_schema,
                "strict": True,
            }
        }
    return body


class XAIClient:
    """:class:`~sciforge.llm.client.ModelClient` for the xAI Responses API."""

    name = "xai"

    def __init__(
        self,
        settings: ModelSettings,
        *,
        http: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = utc_now,
    ) -> None:
        self._settings = settings
        # SCIFORGE_MODEL_TIMEOUT_SECONDS (default 120 s) is the client default AND is passed on every request.
        self._http = http if http is not None else httpx.Client(follow_redirects=False,
                                                                timeout=settings.timeout_seconds)
        self._sleep = sleep
        self._clock = clock
        self._now = now
        self._secrets = settings.secret_values()

    def __repr__(self) -> str:  # never expose the key
        return f"XAIClient(model={self.model!r}, url={self.url!r})"

    @property
    def model(self) -> str:
        return self._settings.model

    @property
    def url(self) -> str:
        return self._settings.responses_url

    def _scrub(self, text: str) -> str:
        return redact_text(text, self._secrets)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._settings.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": self._settings.user_agent,
        }

    def _error_detail(self, response: httpx.Response) -> str:
        """Short, scrubbed provider error message (if any) for exception text."""
        try:
            data = response.json()
        except ValueError:
            return ""
        message: Any = None
        if isinstance(data, dict):
            err = data.get("error")
            message = err.get("message") if isinstance(err, dict) else (err or data.get("message"))
        if not isinstance(message, str) or not message.strip():
            return ""
        return ": " + self._scrub(message.strip())[:_MAX_ERROR_SNIPPET]

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Exactly ONE HTTP attempt (no internal retries). Raises ModelError.

        Retryable failures (429, 5xx, timeout, connection error) are raised
        with ``retryable=True`` and, for 429/5xx, ``retry_after_s`` from the
        ``Retry-After`` header (seconds or HTTP-date, clamped to [0, 30] s).
        :func:`sciforge.llm.budget.budgeted_call` decides whether to retry, so
        every attempt is reserved and recorded against the budget.
        """
        body = build_request_body(self.model, request)
        timer = _Timer(self._clock)
        log = get_logger()
        try:
            response = self._http.post(
                self.url, json=body, headers=self._headers(),
                timeout=self._settings.timeout_seconds, follow_redirects=False,
            )
        except httpx.TimeoutException:
            raise ModelTimeout(f"request timed out after {self._settings.timeout_seconds:g}s",
                               retry_reason="timeout") from None
        except httpx.TransportError as exc:
            raise ModelConnectionError(self._scrub(f"connection error ({type(exc).__name__})"),
                                       retry_reason="connection_error") from None
        status = response.status_code
        log.debug("POST %s -> HTTP %s", self.url, status)
        if 200 <= status < 300:
            return self._parse_success(response, request, timer)
        detail = self._error_detail(response)
        usage = parse_usage(self._json_or_none(response))
        usage = usage if usage.reported else None
        if status in (401, 403):
            raise ModelAuthError(f"HTTP {status} authentication/authorization failed{detail}",
                                 http_status=status, usage=usage)
        if status == 429 or is_retryable_status(status):
            retry_after = retry_after_seconds(response, self._now())
            cls = ModelRateLimited if status == 429 else ModelHTTPError
            label = "Too Many Requests" if status == 429 else "server error"
            raise cls(f"HTTP {status} {label}{detail}", http_status=status, usage=usage, retryable=True,
                      retry_after_s=retry_after, retry_reason="rate_limited" if status == 429 else "http_5xx")
        if 300 <= status < 400:
            raise ModelHTTPError(f"unexpected HTTP {status} redirect (not followed)", http_status=status)
        raise ModelHTTPError(f"HTTP {status} client error{detail}", http_status=status, usage=usage)

    @staticmethod
    def _json_or_none(response: httpx.Response) -> Any:
        try:
            return response.json()
        except ValueError:
            return None

    def _parse_success(self, response: httpx.Response, request: ModelRequest, timer: _Timer) -> ModelResponse:
        status = response.status_code
        try:
            payload = response.json()
        except ValueError:
            raise ModelResponseParseError("response body is not valid JSON", kind="invalid_json_envelope",
                                          http_status=status) from None
        try:
            text = extract_output_text(payload)
            usage = parse_usage(payload)
            parsed = parse_json_text(text, usage=usage) if request.structured else None
        except ModelError as exc:
            exc.http_status = status
            exc.message = self._scrub(exc.message)
            exc.args = (exc.message,)
            raise
        except Exception as exc:  # defensive: never leak a raw KeyError/TypeError
            raise ModelResponseParseError(f"could not interpret response ({type(exc).__name__})",
                                          kind="malformed_payload", http_status=status,
                                          usage=parse_usage(payload)) from None
        model = payload.get("model")
        response_id = payload.get("id")
        status_field = payload.get("status")
        return ModelResponse(
            text=text,
            parsed=parsed,
            usage=usage,
            model=model if isinstance(model, str) else None,
            response_id=response_id if isinstance(response_id, str) else None,
            status=status_field if isinstance(status_field, str) else None,
            http_status=status,
            latency_s=timer.elapsed(),
            attempts=1,
            provider=self.name,
        )

    def close(self) -> None:
        self._http.close()
