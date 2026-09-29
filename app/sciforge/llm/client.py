"""Provider-neutral model-client interface for the v0.3 model layer.

Governing principle: "V0.2 remains the authority for source identity. V0.3 can
interpret verified sources, but it cannot create citations." Nothing in this
module turns model text into a citation or bibliographic field; it only moves
text, parsed JSON and usage numbers between the provider and SciForge code.

This module has no configuration or network dependencies so that it can be
imported by every other model-layer module (and by tests) without side effects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal, Protocol, runtime_checkable

Role = Literal["system", "user", "assistant"]
_ROLES = ("system", "user", "assistant")

# Pipeline stage names -> logical model stages used by stage-aware settings
# (SCIFORGE_MODEL_MAX_OUTPUT_TOKENS_<STAGE>, SCIFORGE_MODEL_REASONING_EFFORT_<STAGE>).
_STAGE_ALIASES = {"extraction": "evidence"}
# Must match sciforge.config.REASONING_EFFORTS (this module has no config dependency).
REASONING_EFFORT_VALUES = ("low", "medium", "high", "xhigh")


def stage_key(stage: str | None) -> str | None:
    """Logical stage of a request stage label: ``"gaps:repair"`` -> ``"gaps"``, ``"extraction"`` -> ``"evidence"``."""
    if not stage:
        return None
    base = stage.split(":", 1)[0].strip().lower()
    return _STAGE_ALIASES.get(base, base) or None


@dataclass(frozen=True)
class ModelMessage:
    """One message of the locally kept conversation history."""

    role: Role
    content: str

    def __post_init__(self) -> None:
        if self.role not in _ROLES:
            raise ValueError(f"invalid message role: {self.role!r}")
        if not isinstance(self.content, str):
            raise TypeError("message content must be a string")

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class ModelRequest:
    """A single, self-contained (stateless) model request.

    ``messages`` is the FULL local history: providers never hold conversation
    state for SciForge (no ``previous_response_id``; D1). A repair retry is a
    new request whose messages are the original ones plus the assistant's
    previous output and a correction message.
    """

    messages: tuple[ModelMessage, ...]
    max_output_tokens: int
    schema_name: str | None = None
    json_schema: dict[str, Any] | None = None
    instructions: str | None = None
    temperature: float = 0.0
    stage: str | None = None
    reasoning_effort: str | None = None   # e.g. "high"; sent as {"reasoning": {"effort": ...}} when set

    def __post_init__(self) -> None:
        object.__setattr__(self, "messages", tuple(self.messages))
        if not self.messages:
            raise ValueError("a model request needs at least one message")
        for message in self.messages:
            if not isinstance(message, ModelMessage):
                raise TypeError("messages must be ModelMessage instances")
        if isinstance(self.max_output_tokens, bool) or not isinstance(self.max_output_tokens, int) \
                or self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be a positive integer")
        if (self.json_schema is None) != (self.schema_name is None):
            raise ValueError("schema_name and json_schema must be given together")
        if self.json_schema is not None and not isinstance(self.json_schema, dict):
            raise TypeError("json_schema must be a dict")
        if not 0.0 <= float(self.temperature) <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        if self.reasoning_effort is not None and self.reasoning_effort not in REASONING_EFFORT_VALUES:
            raise ValueError(f"reasoning_effort must be one of {', '.join(REASONING_EFFORT_VALUES)}")

    @property
    def structured(self) -> bool:
        """True when JSON output matching ``json_schema`` is requested."""
        return self.json_schema is not None

    def with_max_output_tokens(self, value: int) -> ModelRequest:
        """Copy with a different output-token cap (used by the budget)."""
        return ModelRequest(
            messages=self.messages, max_output_tokens=value, schema_name=self.schema_name,
            json_schema=self.json_schema, instructions=self.instructions,
            temperature=self.temperature, stage=self.stage, reasoning_effort=self.reasoning_effort,
        )

    def total_chars(self) -> int:
        """Characters of all text sent (messages, instructions, schema)."""
        chars = sum(len(m.content) for m in self.messages)
        if self.instructions:
            chars += len(self.instructions)
        if self.json_schema is not None:
            chars += len(json.dumps(self.json_schema, separators=(",", ":")))
        return chars


@dataclass(frozen=True)
class ModelUsage:
    """Token usage (and provider-reported cost) for one model call.

    Every field is optional because providers may omit them; ``reported`` is
    False when the response had no usage object at all.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_input_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd_reported: Decimal | None = None      # exact provider-reported cost (never float)
    cost_source: str | None = None                # "reported_ticks" | "reported_nano" when cost is reported
    reported: bool = True

    def __post_init__(self) -> None:
        cost = self.cost_usd_reported
        if cost is not None and not isinstance(cost, Decimal):
            if isinstance(cost, bool) or not isinstance(cost, (int, float, str)):
                raise TypeError("cost_usd_reported must be a Decimal, int, float or str")
            object.__setattr__(self, "cost_usd_reported", Decimal(str(cost)))
        if self.cost_usd_reported is not None and self.cost_source is None:
            object.__setattr__(self, "cost_source", "reported")

    @property
    def billable_output_tokens(self) -> int | None:
        """Output-side tokens to charge: ``output_tokens + reasoning_tokens`` (conservative).

        xAI reports reasoning tokens SEPARATELY from output tokens (``usage.output_tokens_details.
        reasoning_tokens``; confirmed externally by the project owner for grok-4.7 — not verifiable offline), so
        the charge is ``output + reasoning`` whenever reasoning is reported. Reasoning is added exactly once. If a
        ``total_tokens`` is reported and ``total - input`` is larger still, that larger value is charged
        (``max`` — only ever raises the charge, never double counts beyond the reported total's implication).
        Without reported reasoning tokens the charge is ``max(output, total - input)``.
        """
        if self.output_tokens is None:
            return None
        candidates = [self.output_tokens]
        if self.reasoning_tokens is not None:
            candidates.append(self.output_tokens + self.reasoning_tokens)
        if self.total_tokens is not None and self.input_tokens is not None:
            candidates.append(self.total_tokens - self.input_tokens)
        return max(candidates)

    @property
    def reasoning_included_in_output(self) -> bool | None:
        """Whether the reported ``output_tokens`` already include the reasoning tokens (None = cannot tell)."""
        if self.output_tokens is None or self.reasoning_tokens is None:
            return None
        if self.reasoning_tokens > self.output_tokens:
            return False
        if self.total_tokens is not None and self.input_tokens is not None:
            if self.total_tokens == self.input_tokens + self.output_tokens + self.reasoning_tokens \
                    and self.reasoning_tokens > 0:
                return False
            if self.total_tokens == self.input_tokens + self.output_tokens:
                return True
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd_reported": None if self.cost_usd_reported is None else str(self.cost_usd_reported),
            "cost_source": self.cost_source,
            "cost_reported_status": "reported" if self.cost_usd_reported is not None else "unavailable",
            "output_side_tokens": self.billable_output_tokens,
            "reasoning_included_in_output_tokens": self.reasoning_included_in_output,
            "reported": self.reported,
        }


NO_USAGE = ModelUsage(reported=False)


@dataclass(frozen=True)
class ModelResponse:
    """A successful model response (text extracted, JSON parsed if requested)."""

    text: str
    parsed: Any | None
    usage: ModelUsage
    model: str | None
    response_id: str | None = None
    status: str | None = None
    http_status: int | None = None
    latency_s: float = 0.0
    attempts: int = 1
    provider: str = "unknown"

    def to_dict(self, *, include_text: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "provider": self.provider,
            "model": self.model,
            "response_id": self.response_id,
            "status": self.status,
            "http_status": self.http_status,
            "latency_s": round(self.latency_s, 3),
            "attempts": self.attempts,
            "usage": self.usage.to_dict(),
        }
        if include_text:
            data["text"] = self.text
        return data


# --------------------------------------------------------------------- errors


class ModelError(Exception):
    """Base class for every model-layer failure.

    ``usage`` is set when the provider processed (and may bill) the request,
    so budget accounting can charge it even though the attempt failed.
    ``retryable`` / ``retry_after_s`` / ``retry_reason`` tell the budgeted
    retry loop (``llm.budget.budgeted_call``) whether another attempt may help;
    clients themselves never retry. ``message`` never contains credentials or
    request headers.
    """

    kind = "model_error"
    outcome = "error"             # audit outcome category
    default_retryable = False

    def __init__(
        self,
        message: str,
        *,
        http_status: int | None = None,
        usage: ModelUsage | None = None,
        raw_text: str | None = None,
        attempts: int = 1,
        kind: str | None = None,
        retryable: bool | None = None,
        retry_after_s: float | None = None,
        retry_reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.http_status = http_status
        self.usage = usage
        self.raw_text = raw_text
        self.attempts = attempts
        if kind is not None:
            self.kind = kind
        self.retryable = self._default_retryable() if retryable is None else retryable
        self.retry_after_s = retry_after_s
        self.retry_reason = retry_reason or (self.kind if self.retryable else None)

    def _default_retryable(self) -> bool:
        return self.default_retryable

    @property
    def reached_provider(self) -> bool:
        """True if the request may have been processed by the provider."""
        return True

    def to_dict(self) -> dict[str, Any]:
        return {"error_type": self.kind, "message": self.message, "http_status": self.http_status,
                "attempts": self.attempts, "retryable": self.retryable}


class ModelConfigError(ModelError):
    """Model layer requested but misconfigured (missing key/model, bad values,
    spend cap without prices). Raised before any network I/O (CLI exit 4)."""

    kind = "config_error"

    @property
    def reached_provider(self) -> bool:
        return False


class ModelHTTPError(ModelError):
    """HTTP error response. 5xx (and 429, see subclass) are retryable."""

    kind = "http_error"
    outcome = "http_error"

    def _default_retryable(self) -> bool:
        return self.http_status is not None and (self.http_status == 429 or self.http_status >= 500)


class ModelAuthError(ModelHTTPError):
    """401/403: missing, invalid or unauthorised API key. Never retried."""

    kind = "auth_error"

    def _default_retryable(self) -> bool:
        return False


class ModelRateLimited(ModelHTTPError):
    """HTTP 429 Too Many Requests (retryable; honours Retry-After)."""

    kind = "rate_limited"

    def _default_retryable(self) -> bool:
        return True


class ModelTimeout(ModelError):
    """The attempt timed out (retryable)."""

    kind = "timeout"
    outcome = "timeout"
    default_retryable = True


class ModelConnectionError(ModelError):
    """Connection / transport failure (retryable)."""

    kind = "connection_error"
    outcome = "connection_error"
    default_retryable = True


class ModelResponseParseError(ModelError):
    """The provider answered 2xx but the payload could not be used."""

    kind = "parse_error"
    outcome = "parse_error"


class ModelIncompleteError(ModelResponseParseError):
    """``status == "incomplete"`` (e.g. the output-token cap was reached)."""

    kind = "incomplete"

    def __init__(self, message: str, *, reason: str | None = None, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.reason = reason


class ModelRefusalError(ModelResponseParseError):
    """The model refused to answer."""

    kind = "refused"


class ModelSchemaError(ModelError):
    """Parsed JSON did not validate against the expected pydantic model."""

    kind = "schema_violation"
    outcome = "parse_error"


class BudgetExhausted(ModelError):
    """A budget limit is reached; raised BEFORE the next attempt is made.

    ``last_error`` is the underlying error of the previous failed attempt when
    a retry was blocked by the budget (also chained as ``__cause__``).
    """

    kind = "budget_exhausted"
    outcome = "budget_exhausted"

    def __init__(self, limit: str, detail: str, *, last_error: ModelError | None = None) -> None:
        super().__init__(f"budget limit {limit} reached: {detail}")
        self.limit = limit
        self.detail = detail
        self.last_error = last_error

    @property
    def reached_provider(self) -> bool:
        return False


# --------------------------------------------------------------------- protocol


@runtime_checkable
class ModelClient(Protocol):
    """What the model layer needs from a provider.

    ``complete`` performs exactly ONE stateless attempt (one HTTP request, no
    internal retries) and returns a :class:`ModelResponse` or raises a
    :class:`ModelError` (with ``retryable`` / ``retry_after_s`` set); it never
    returns partial output silently. Retries are driven by
    :func:`sciforge.llm.budget.budgeted_call` so that every attempt is budgeted.
    """

    name: str
    model: str

    def complete(self, request: ModelRequest) -> ModelResponse: ...


@dataclass
class _Timer:
    """Tiny helper to measure latency with an injectable clock."""

    clock: Any
    start: float = field(default=0.0)

    def __post_init__(self) -> None:
        self.start = self.clock()

    def elapsed(self) -> float:
        return max(0.0, self.clock() - self.start)
