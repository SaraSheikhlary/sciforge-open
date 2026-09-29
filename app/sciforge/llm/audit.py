"""Local, secret-redacted audit trail of model calls (``model_calls.json``; D5).

There is one ``attempt`` entry per API attempt (retries included) and one
``budget_stop`` entry whenever an attempt is refused by the budget. Each entry
records metadata, usage, cost accounting (``cost_usd`` as an exact decimal
string plus ``cost_source``) and SHA-256 hashes of the request and response. The request messages/instructions and the response text
are included only when ``store_prompts`` is True (``SCIFORGE_STORE_PROMPTS``,
default true). Request headers — and therefore the Authorization header and
API key — are never part of an entry; every string is additionally scrubbed of
known secret values and Bearer/Authorization patterns before it is stored.
Only the reasoning-effort *setting* is recorded; hidden reasoning returned by
the provider (reasoning output items, encrypted reasoning, summaries) never
reaches an entry, because clients only return ``output_text`` and usage.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from sciforge.llm.client import BudgetExhausted, ModelError, ModelRequest, ModelResponse
from sciforge.logging_utils import iso_utc, redact_text


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def request_fingerprint(request: ModelRequest) -> str:
    """SHA-256 of the canonical request content (no headers, no key)."""
    canonical = json.dumps(
        {
            "messages": [m.to_dict() for m in request.messages],
            "instructions": request.instructions,
            "schema_name": request.schema_name,
            "json_schema": request.json_schema,
            "max_output_tokens": request.max_output_tokens,
            "temperature": request.temperature,
            # only when set, so fingerprints of requests without a reasoning effort are unchanged
            **({"reasoning_effort": request.reasoning_effort} if request.reasoning_effort is not None else {}),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return _sha256(canonical)


class ModelCallAudit:
    """Accumulates audit entries for one investigation."""

    def __init__(self, *, store_prompts: bool = True, secrets: Iterable[str | None] = (),
                 now: Callable[[], str] = iso_utc) -> None:
        self.store_prompts = store_prompts
        self._secrets = [s for s in secrets if s]
        self._now = now
        self.entries: list[dict[str, Any]] = []

    def _scrub(self, value: Any) -> Any:
        if isinstance(value, str):
            return redact_text(value, self._secrets)
        if isinstance(value, dict):
            return {k: self._scrub(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._scrub(v) for v in value]
        return value

    def _base(self, request: ModelRequest, *, kind: str, call_id: int | None, attempt: int | None,
              timestamp: str | None, provider: str | None, model: str | None) -> dict[str, Any]:
        return {
            "index": len(self.entries) + 1,
            "entry_type": kind,
            "timestamp": timestamp or self._now(),
            "provider": provider,
            "model_requested": model,
            "stage": request.stage,
            "call_id": call_id,
            "attempt": attempt,
            "schema_name": request.schema_name,
            "max_output_tokens": request.max_output_tokens,
            "temperature": request.temperature,
            "reasoning_effort": request.reasoning_effort,   # the setting only; reasoning content is never stored
            "request_sha256": request_fingerprint(request),
            "prompts_stored": self.store_prompts,
        }

    def add_attempt(
        self,
        request: ModelRequest,
        *,
        call_id: int | None = None,
        attempt: int = 1,
        timestamp: str | None = None,
        latency_s: float | None = None,
        response: ModelResponse | None = None,
        error: BaseException | None = None,
        accounting: dict[str, Any] | None = None,
        provider: str | None = None,
        model: str | None = None,
        will_retry: bool = False,
        backoff_s: float | None = None,
        retry_blocked_by: str | None = None,
    ) -> dict[str, Any]:
        """Append (and return) the entry for ONE API attempt (success or failure).

        Records attempt number, stage, timestamp, outcome (``success`` /
        ``http_error`` / ``timeout`` / ``connection_error`` / ``parse_error`` /
        ``error``), HTTP status, retry reason, Retry-After and backoff applied,
        usage, and the recorded cost with its ``cost_source``.
        """
        entry = self._base(request, kind="attempt", call_id=call_id, attempt=attempt, timestamp=timestamp,
                           provider=provider or (response.provider if response else None), model=model)
        acct = accounting or {}
        entry.update({
            "outcome": "success" if response is not None else getattr(error, "outcome", "error"),
            "latency_s": None if latency_s is None else round(max(0.0, latency_s), 3),
            "cost_usd": acct.get("cost_usd"),
            "cost_source": acct.get("cost_source"),
            "accounting": accounting,
        })
        if response is not None:
            entry.update({
                "http_status": response.http_status,
                "model_reported": response.model,
                "response_id": response.response_id,
                "status": response.status,
                "usage": response.usage.to_dict(),
                "response_sha256": _sha256(response.text),
            })
        if error is not None:
            model_error = error if isinstance(error, ModelError) else None
            entry.update({
                "error_type": model_error.kind if model_error else type(error).__name__,
                "error_message": model_error.message if model_error else "unexpected error",
                "http_status": model_error.http_status if model_error else None,
                "retryable": model_error.retryable if model_error else False,
                "retry_reason": model_error.retry_reason if model_error else None,
                "retry_after_s": model_error.retry_after_s if model_error else None,
                "will_retry": will_retry,
                "backoff_s": backoff_s,
                "retry_blocked_by": retry_blocked_by,
                "usage": model_error.usage.to_dict() if model_error and model_error.usage else None,
                "response_sha256": _sha256(model_error.raw_text) if model_error and model_error.raw_text
                else None,
            })
        if self.store_prompts:
            entry["request"] = {
                "instructions": request.instructions,
                "messages": [m.to_dict() for m in request.messages],
            }
            if response is not None:
                entry["response_text"] = response.text
            elif isinstance(error, ModelError) and error.raw_text:
                entry["response_text"] = error.raw_text
        return self._append(entry)

    def add_budget_stop(
        self,
        request: ModelRequest,
        stop: BudgetExhausted,
        *,
        call_id: int | None = None,
        blocked_attempt: int | None = None,
        timestamp: str | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Record that an attempt was NOT made because a budget limit was reached."""
        entry = self._base(request, kind="budget_stop", call_id=call_id, attempt=blocked_attempt,
                           timestamp=timestamp, provider=provider, model=model)
        last = stop.last_error
        entry.update({
            "outcome": "budget_exhausted",
            "limit": stop.limit,
            "detail": stop.detail,
            "last_error": last.to_dict() if last is not None else None,
            "cost_usd": "0",
            "cost_source": None,
        })
        return self._append(entry)

    def _append(self, entry: dict[str, Any]) -> dict[str, Any]:
        entry = self._scrub(entry)
        self.entries.append(entry)
        return entry

    def to_json(self) -> str:
        return json.dumps(self.entries, indent=2, ensure_ascii=False) + "\n"

    def write(self, path: str | Path) -> Path:
        """Write all entries to ``path`` (e.g. ``runs/<id>/model_calls.json``)."""
        path = Path(path)
        path.write_text(self.to_json(), encoding="utf-8")
        return path
