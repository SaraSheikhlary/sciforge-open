"""Structured model call with one repair retry (shared by all model stages).

``call_structured``: ``budgeted_call`` (every attempt budgeted + audited) →
validate → on invalid JSON / schema violation ONE repair request that resends
the FULL local history (original messages + the assistant's raw output +
a correction message; never ``previous_response_id``). The repair is a new
logical call and counts as an attempt. Failures are returned, never raised
(except programming errors), so a stage failure cannot crash a run.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generic, TypeVar

from sciforge.llm.audit import ModelCallAudit
from sciforge.llm.budget import BudgetTracker, RetryPolicy, budgeted_call
from sciforge.llm.client import (
    BudgetExhausted,
    ModelAuthError,
    ModelClient,
    ModelError,
    ModelMessage,
    ModelRequest,
    ModelResponseParseError,
    ModelSchemaError,
    stage_key,
)
from sciforge.llm.parsing import parse_json_text
from sciforge.logging_utils import iso_utc, utc_now
from sciforge.prompts import REPAIR_TEMPLATE

T = TypeVar("T")

MAX_REPAIR_ECHO_CHARS = 4000
REPAIRABLE_PARSE_KINDS = frozenset({"invalid_output_json"})


@dataclass
class CallContext:
    """Everything a stage needs to make budgeted, audited model calls."""

    client: ModelClient
    tracker: BudgetTracker
    audit: ModelCallAudit | None = None
    retry: RetryPolicy | None = None
    keep_attempts: int = 0
    sleep: Callable[[float], None] = time.sleep
    now: Callable[[], datetime] = utc_now
    # Effective reasoning effort per logical stage (question, evidence, gaps, hypotheses, report); a stage
    # missing from the mapping sends no reasoning parameter. Built from ModelSettings.reasoning_efforts().
    reasoning_efforts: Mapping[str, str] = field(default_factory=dict)

    @property
    def max_output_tokens(self) -> int:
        """Global per-call output cap (stage-independent fallback)."""
        return self.tracker.limits.max_output_tokens_per_call

    def max_output_tokens_for(self, stage: str | None) -> int:
        """Effective output cap for ``stage`` (stage override from the budget limits, else the global cap)."""
        return self.tracker.limits.max_output_tokens_for(stage)

    def reasoning_effort_for(self, stage: str | None) -> str | None:
        return self.reasoning_efforts.get(stage_key(stage) or "")


@dataclass
class StructuredResult(Generic[T]):
    """Outcome of one structured stage call (incl. an optional repair)."""

    value: T | None = None
    logical_calls: int = 0
    attempts: int = 0
    repaired: bool = False
    error: dict[str, Any] | None = None
    stop: str | None = None          # "budget_exhausted" | "auth_error" → orchestrator stops calling the model
    errors: list[dict[str, Any]] = field(default_factory=list)
    audit_indices: list[int] = field(default_factory=list)  # positions of this call's entries in ctx.audit.entries
    response_sha256: str | None = None  # SHA-256 of the text of the last response received (matches audit entries)

    @property
    def ok(self) -> bool:
        return self.value is not None and self.error is None


def _error_dict(stage: str, exc: BaseException, *, record_id: str | None = None, repair: bool = False) -> dict[str, Any]:
    if isinstance(exc, ModelError):
        info: dict[str, Any] = {"error_type": exc.kind, "message": exc.message, "http_status": exc.http_status}
        if isinstance(exc, BudgetExhausted):
            info["limit"] = exc.limit
    else:
        info = {"error_type": "unexpected_error", "message": type(exc).__name__, "http_status": None}
    return {"stage": stage, "record_id": record_id, "repair_attempt": repair, "timestamp": iso_utc(), **info}


def _raw_text(exc: ModelError) -> str:
    return (exc.raw_text or "")[:MAX_REPAIR_ECHO_CHARS]


def call_structured(
    ctx: CallContext,
    *,
    stage: str,
    instructions: str,
    messages: list[ModelMessage],
    schema_name: str,
    json_schema: dict[str, Any],
    validate: Callable[[Any, str], T],
    record_id: str | None = None,
) -> StructuredResult[T]:
    """Run one structured call with at most one repair retry.

    ``validate(parsed_json, raw_text)`` must return the validated value or
    raise :class:`ModelSchemaError` / :class:`ModelResponseParseError`.
    """
    result: StructuredResult[T] = StructuredResult()
    attempts_before = ctx.tracker.attempts
    audit_before = len(ctx.audit.entries) if ctx.audit is not None else 0
    try:
        return _run(ctx, result, stage=stage, instructions=instructions, messages=messages, schema_name=schema_name,
                    json_schema=json_schema, validate=validate, record_id=record_id)
    finally:
        result.attempts = ctx.tracker.attempts - attempts_before
        if ctx.audit is not None:
            result.audit_indices = list(range(audit_before, len(ctx.audit.entries)))


def _run(ctx: CallContext, result: StructuredResult[T], *, stage: str, instructions: str,
         messages: list[ModelMessage], schema_name: str, json_schema: dict[str, Any],
         validate: Callable[[Any, str], T], record_id: str | None) -> StructuredResult[T]:
    history = list(messages)
    for repair in (False, True):
        request = ModelRequest(messages=tuple(history), max_output_tokens=ctx.max_output_tokens_for(stage),
                               schema_name=schema_name, json_schema=json_schema, instructions=instructions,
                               stage=f"{stage}:repair" if repair else stage,
                               reasoning_effort=ctx.reasoning_effort_for(stage))
        result.logical_calls += 1
        result.repaired = repair
        try:
            response = budgeted_call(ctx.client, request, ctx.tracker, retry=ctx.retry, audit=ctx.audit,
                                     keep_attempts=ctx.keep_attempts, sleep=ctx.sleep, now=ctx.now)
            result.response_sha256 = hashlib.sha256(response.text.encode("utf-8")).hexdigest()
            parsed = response.parsed if response.parsed is not None else parse_json_text(response.text,
                                                                                          usage=response.usage)
            try:
                result.value = validate(parsed, response.text)
            except ModelError as exc:
                if exc.raw_text is None:
                    exc.raw_text = response.text
                raise
            result.error = None
            return result
        except BudgetExhausted as exc:
            result.error = _error_dict(stage, exc, record_id=record_id, repair=repair)
            result.errors.append(result.error)
            result.stop = "budget_exhausted"
            return result
        except ModelAuthError as exc:
            result.error = _error_dict(stage, exc, record_id=record_id, repair=repair)
            result.errors.append(result.error)
            result.stop = "auth_error"
            return result
        except (ModelSchemaError, ModelResponseParseError) as exc:
            result.error = _error_dict(stage, exc, record_id=record_id, repair=repair)
            result.errors.append(result.error)
            repairable = isinstance(exc, ModelSchemaError) or exc.kind in REPAIRABLE_PARSE_KINDS
            if repair or not repairable:
                return result
            summary = exc.message if isinstance(exc, ModelSchemaError) else "output was not valid JSON"
            history = history + [ModelMessage("assistant", _raw_text(exc)),
                                 ModelMessage("user", REPAIR_TEMPLATE.format(summary=summary))]
        except ModelError as exc:
            result.error = _error_dict(stage, exc, record_id=record_id, repair=repair)
            result.errors.append(result.error)
            return result
        except Exception as exc:  # noqa: BLE001 - a stage failure must not crash the run
            result.error = _error_dict(stage, exc, record_id=record_id, repair=repair)
            result.errors.append(result.error)
            return result
    return result
