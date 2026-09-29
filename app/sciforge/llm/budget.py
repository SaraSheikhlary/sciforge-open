"""Per-investigation budget accounting for model API attempts (decision D8).

Limits (defaults): 15 API **attempts** (every HTTP attempt counts, including
retries after 429 / 5xx / timeouts / connection errors), 10 sources, 200,000
cumulative input tokens, 2,000 output tokens per attempt, and a $15 spend cap.
The per-attempt output cap can be overridden per model stage
(``stage_max_output_tokens``; e.g. ``SCIFORGE_MODEL_MAX_OUTPUT_TOKENS_HYPOTHESES``);
the effective stage value is what is granted, reserved and charged below.

Where the retry loop lives
--------------------------
Clients (``XAIClient``, ``FakeModelClient``) make exactly one attempt per
``complete()`` call. The retry loop lives in :func:`budgeted_call`: before
EACH attempt it calls :meth:`BudgetTracker.reserve` (pre-check against the
worst case), after each attempt it calls :meth:`BudgetTracker.record`, and it
retries only retryable errors while both the retry policy and the budget allow.
If the budget blocks a retry, :class:`BudgetExhausted` is raised immediately
with the previous attempt's error attached (``last_error`` / ``__cause__``).

Pre-call check (estimates)
--------------------------
worst case = pessimistic input estimate (chars/3 + overhead) and the granted
``max_output_tokens`` (= min(request, effective cap of the request's stage)); worst-case cost = price table applied to both. An
attempt is refused if ``attempts + 1 + keep_attempts > max_attempts``,
``input_used + est_input > max_input_tokens`` or ``spend + worst_cost >
max_spend_usd``. Once ANY of these limits is reached (by a refusal, or because
recorded usage reached it), the tracker is exhausted and refuses every further
attempt ("sticky"). Exception: a refusal caused only by ``keep_attempts``
(attempts held back for later stages) is not sticky, so later stages can still
use the kept attempts.

Recorded spend (actuals) — ``cost_source`` values
-------------------------------------------------
* ``reported_ticks`` — exact ``usage.cost_in_usd_ticks`` / 10**10 (Decimal)
* ``reported_nano``  — exact ``usage.cost_in_nano_usd`` / 10**9 (Decimal)
* ``price_estimate`` — no reported cost but usage reported: price table applied
  to the charged tokens (missing token fields filled with the worst case)
* ``estimated_worst_case`` — no usage at all (timeouts, connection errors, any
  HTTP error or 2xx without usage): the pre-call worst-case cost is charged,
  together with the worst-case tokens (estimated input + granted max output)
* ``unpriced`` — cost unknowable (no reported cost, no prices; only possible
  when the spend cap is disabled). Tokens are still charged.

Output-side tokens: ``output_tokens_charged`` = :attr:`ModelUsage.billable_output_tokens` = reported
``output_tokens + reasoning_tokens`` (xAI reports reasoning tokens separately from output tokens; reasoning is
added exactly once; a larger ``total_tokens - input_tokens`` is charged if reported). Each attempt's accounting also records the raw reported ``output_tokens``,
``reasoning_tokens`` (``usage.output_tokens_details.reasoning_tokens``), ``total_tokens`` and the
provider-reported cost (``reported_cost_usd``; ``reported_cost_status`` = ``reported`` / ``unavailable`` —
never fabricated). Reasoning *content* is never stored.

Token settings vs. money: the per-call output setting (``SCIFORGE_MODEL_MAX_OUTPUT_TOKENS`` and per-stage
overrides) is sent as the request's ``max_output_tokens`` and used for the pre-call worst case; it is NOT a
guaranteed ceiling on billed tokens: xAI reports reasoning tokens separately from output tokens, so
``max_output_tokens`` is not a total token ceiling. ``SCIFORGE_MAX_SPEND_USD`` is the hard financial guard: attempts whose worst case would
exceed it are refused, and once recorded spend reaches it no further attempt is made.

Failed attempts are charged exactly like successes under these rules, i.e. a
failure without reported usage/cost is charged the full worst case. This is
deliberately conservative: xAI's docs do not state which rejected requests
(429/4xx) are free, and over-charging can only stop a run early, while
under-charging could breach the cap. ``assumed_not_billed`` is therefore never
used. The price table is only ever an estimate; a reported cost always wins.
Spend is kept as :class:`decimal.Decimal` and serialised as decimal strings.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from sciforge.llm.client import (
    BudgetExhausted,
    ModelClient,
    ModelConfigError,
    ModelError,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    stage_key,
)
from sciforge.logging_utils import iso_utc, utc_now

if TYPE_CHECKING:
    from sciforge.llm.audit import ModelCallAudit

CHARS_PER_TOKEN_ESTIMATE = 3  # deliberately pessimistic (English averages ~4)
MESSAGE_OVERHEAD_TOKENS = 8   # per message / instructions block, for role markers etc.
_MILLION = Decimal(1_000_000)

COST_SOURCES = ("reported_ticks", "reported_nano", "price_estimate", "estimated_worst_case", "unpriced")


def _dec(value: float | int | Decimal | None) -> Decimal | None:
    if value is None:
        return None
    return value if isinstance(value, Decimal) else Decimal(str(value))


def _dstr(value: Decimal | None) -> str | None:
    return None if value is None else format(value.normalize() if value != 0 else Decimal(0), "f")


@dataclass(frozen=True)
class PriceTable:
    """User-supplied prices in USD per 1M tokens. No prices are shipped.
    Used only for pre-call estimates and for attempts without a reported cost."""

    input_per_mtok: float | Decimal | None = None
    output_per_mtok: float | Decimal | None = None

    def __post_init__(self) -> None:
        for name in ("input_per_mtok", "output_per_mtok"):
            value = _dec(getattr(self, name))
            if value is not None and (not value.is_finite() or value < 0):
                raise ValueError("prices must be non-negative numbers")
            object.__setattr__(self, name, value)

    @property
    def configured(self) -> bool:
        return self.input_per_mtok is not None and self.output_per_mtok is not None

    def cost(self, input_tokens: int, output_tokens: int) -> Decimal | None:
        """Estimated USD cost (exact Decimal), or None when prices are not configured."""
        if not self.configured:
            return None
        assert isinstance(self.input_per_mtok, Decimal) and isinstance(self.output_per_mtok, Decimal)
        return (input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok) / _MILLION


@dataclass(frozen=True)
class BudgetLimits:
    max_attempts: int = 15
    max_sources: int = 10
    max_input_tokens: int = 200_000
    max_output_tokens_per_call: int = 2_000
    max_spend_usd: float | Decimal | None = 15.0  # None = explicitly disabled
    # Per-stage output caps as ((logical stage, cap), ...); stages not listed use max_output_tokens_per_call.
    stage_max_output_tokens: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        for name in ("max_attempts", "max_sources", "max_input_tokens", "max_output_tokens_per_call"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        pairs = tuple(sorted(dict(self.stage_max_output_tokens).items()))
        for stage, value in pairs:
            if not isinstance(stage, str) or not stage:
                raise ValueError("stage_max_output_tokens keys must be stage names")
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("stage_max_output_tokens values must be positive integers")
        object.__setattr__(self, "stage_max_output_tokens", pairs)
        cap = _dec(self.max_spend_usd)
        if cap is not None and not cap > 0:
            raise ValueError("max_spend_usd must be > 0, or None to disable the cap")
        object.__setattr__(self, "max_spend_usd", cap)

    def max_output_tokens_for(self, stage: str | None) -> int:
        """Effective per-attempt output cap for a request stage (``"gaps:repair"`` counts as ``gaps``)."""
        return dict(self.stage_max_output_tokens).get(stage_key(stage) or "", self.max_output_tokens_per_call)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "max_attempts": self.max_attempts,
            "max_sources": self.max_sources,
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens_per_call": self.max_output_tokens_per_call,
            "max_spend_usd": _dstr(self.max_spend_usd),  # type: ignore[arg-type]
        }
        if self.stage_max_output_tokens:
            data["max_output_tokens_by_stage"] = dict(self.stage_max_output_tokens)
        return data


@dataclass(frozen=True)
class RetryPolicy:
    """Retries after retryable failures (same semantics as v0.2 HttpFetcher)."""

    max_retries: int = 2
    backoff_seconds: float = 1.0

    def delay(self, retry_number: int, retry_after_s: float | None) -> float:
        """Delay before retry ``retry_number`` (0-based): Retry-After wins
        (already clamped to [0, 30] s), else ``backoff * 2**retry_number``."""
        if retry_after_s is not None:
            return retry_after_s
        return self.backoff_seconds * (2**retry_number)


@dataclass(frozen=True)
class Reservation:
    """A granted pre-attempt reservation. ``request`` is the capped request to send."""

    id: int
    request: ModelRequest
    estimated_input_tokens: int
    max_output_tokens: int
    worst_case_cost_usd: Decimal | None


def estimate_input_tokens(request: ModelRequest) -> int:
    """Pessimistic local estimate of the request's input tokens (no tokenizer)."""
    blocks = len(request.messages) + (1 if request.instructions else 0)
    return math.ceil(request.total_chars() / CHARS_PER_TOKEN_ESTIMATE) + MESSAGE_OVERHEAD_TOKENS * blocks


class BudgetTracker:
    """Tracks and enforces :class:`BudgetLimits` for one investigation."""

    def __init__(self, limits: BudgetLimits | None = None, prices: PriceTable | None = None) -> None:
        self.limits = limits or BudgetLimits()
        self.prices = prices or PriceTable()
        if self.limits.max_spend_usd is not None and not self.prices.configured:
            raise ModelConfigError(
                "spend cap is enabled but model prices are not configured "
                "(set SCIFORGE_PRICE_INPUT_PER_MTOK and SCIFORGE_PRICE_OUTPUT_PER_MTOK, "
                "or disable the cap with SCIFORGE_MAX_SPEND_USD=none)"
            )
        self.attempts = 0
        self.failed_attempts = 0
        self.logical_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.reasoning_tokens = 0
        self.output_tokens_reported = 0
        self.reported_cost_usd = Decimal(0)
        self.reported_cost_attempts = 0
        self.sources = 0
        self.sources_limited = False
        self.spend_usd = Decimal(0)
        self.spend_by_source: dict[str, Decimal] = {s: Decimal(0) for s in COST_SOURCES}
        self.unpriced_attempts = 0
        self.exhausted_by: str | None = None
        self._pending: dict[int, Reservation] = {}
        self._next_id = 1

    # ------------------------------------------------------------- estimates

    @staticmethod
    def estimate_input_tokens(request: ModelRequest) -> int:
        return estimate_input_tokens(request)

    def _pending_totals(self) -> tuple[int, int, Decimal]:
        attempts = len(self._pending)
        tokens = sum(r.estimated_input_tokens for r in self._pending.values())
        cost = sum((r.worst_case_cost_usd or Decimal(0) for r in self._pending.values()), Decimal(0))
        return attempts, tokens, cost

    def _stop(self, limit: str, detail: str, last_error: ModelError | None) -> BudgetExhausted:
        if self.exhausted_by is None:
            self.exhausted_by = limit
        return BudgetExhausted(limit, detail, last_error=last_error)

    @property
    def exhausted(self) -> bool:
        return self.exhausted_by is not None

    # ------------------------------------------------------------- pre-attempt

    def reserve(self, request: ModelRequest, *, keep_attempts: int = 0,
                last_error: ModelError | None = None) -> Reservation:
        """Check the worst case of ONE attempt of ``request`` against every limit.

        Raises :class:`BudgetExhausted` (without changing any counter) if the
        attempt could exceed a limit, or if a limit was already reached.
        ``keep_attempts`` attempts are kept free for later stages. The returned
        reservation's ``request`` has ``max_output_tokens`` capped at the
        effective per-call limit of the request's stage — send that one.
        """
        if keep_attempts < 0:
            raise ValueError("keep_attempts must be >= 0")
        lim = self.limits
        if self.exhausted_by is not None:
            raise BudgetExhausted(self.exhausted_by, "budget already exhausted; no further attempts",
                                  last_error=last_error)
        pending_attempts, pending_tokens, pending_cost = self._pending_totals()

        if self.attempts + pending_attempts + 1 > lim.max_attempts:
            raise self._stop("max_attempts", f"{self.attempts} of {lim.max_attempts} attempts used", last_error)
        if self.attempts + pending_attempts + 1 + keep_attempts > lim.max_attempts:
            # Not sticky: the kept attempts remain available to later stages.
            raise BudgetExhausted("max_attempts",
                                  f"{self.attempts} of {lim.max_attempts} attempts used, "
                                  f"{keep_attempts} kept for later stages", last_error=last_error)

        granted_output = min(request.max_output_tokens, lim.max_output_tokens_for(request.stage))
        est_input = estimate_input_tokens(request)
        if self.input_tokens + pending_tokens + est_input > lim.max_input_tokens:
            raise self._stop("max_input_tokens",
                             f"{self.input_tokens} used + ~{est_input} estimated > {lim.max_input_tokens}",
                             last_error)

        worst_cost = self.prices.cost(est_input, granted_output)
        cap = lim.max_spend_usd
        if cap is not None:
            assert worst_cost is not None  # guaranteed by the constructor check
            if self.spend_usd + pending_cost + worst_cost > cap:
                raise self._stop("max_spend_usd",
                                 f"${_dstr(self.spend_usd)} recorded + worst case ${_dstr(worst_cost)} "
                                 f"> cap ${_dstr(cap)}", last_error)  # type: ignore[arg-type]

        capped = request if granted_output == request.max_output_tokens \
            else request.with_max_output_tokens(granted_output)
        reservation = Reservation(self._next_id, capped, est_input, granted_output, worst_cost)
        self._next_id += 1
        self._pending[reservation.id] = reservation
        return reservation

    # ------------------------------------------------------------- post-attempt

    def _take(self, reservation: Reservation) -> None:
        if self._pending.pop(reservation.id, None) is None:
            raise ValueError("unknown or already settled reservation")

    def record(self, reservation: Reservation, usage: ModelUsage | None, *, failed: bool = False) -> dict[str, Any]:
        """Charge one attempt that was sent to the provider (success or failure).

        See the module docstring for the ``cost_source`` rules. Returns the
        per-attempt accounting dict (decimal amounts as strings).
        """
        self._take(reservation)
        usage = usage if (usage is not None and usage.reported) else None
        if usage is None:
            in_tok = reservation.estimated_input_tokens
            out_tok = reservation.max_output_tokens
            reasoning = 0
        else:
            in_tok = usage.input_tokens if usage.input_tokens is not None else reservation.estimated_input_tokens
            billable = usage.billable_output_tokens
            out_tok = billable if billable is not None else reservation.max_output_tokens
            reasoning = usage.reasoning_tokens or 0

        estimate = self.prices.cost(in_tok, out_tok)
        if usage is not None and usage.cost_usd_reported is not None:
            cost: Decimal | None = usage.cost_usd_reported
            source = usage.cost_source if usage.cost_source in ("reported_ticks", "reported_nano") \
                else "reported_ticks"
        elif usage is None:
            cost, source = reservation.worst_case_cost_usd, "estimated_worst_case"
        else:
            cost, source = estimate, "price_estimate"
        if cost is None:
            source = "unpriced"

        self.attempts += 1
        if failed:
            self.failed_attempts += 1
        self.input_tokens += in_tok
        self.output_tokens += out_tok
        self.reasoning_tokens += reasoning
        if usage is not None and usage.output_tokens is not None:
            self.output_tokens_reported += usage.output_tokens
        reported_cost = usage.cost_usd_reported if usage is not None else None
        if reported_cost is not None:
            self.reported_cost_usd += reported_cost
            self.reported_cost_attempts += 1
        if cost is None:
            self.unpriced_attempts += 1
        else:
            self.spend_usd += cost
            self.spend_by_source[source] += cost
        self._check_reached()
        return {
            "input_tokens_charged": in_tok,
            "output_tokens_charged": out_tok,
            "output_side_tokens_charged": out_tok,
            "output_tokens_reported": usage.output_tokens if usage is not None else None,
            "reasoning_tokens_reported": usage.reasoning_tokens if usage is not None else None,
            "total_tokens_reported": usage.total_tokens if usage is not None else None,
            "reported_cost_usd": _dstr(reported_cost),
            "reported_cost_status": "reported" if reported_cost is not None else "unavailable",
            "usage_reported": usage is not None,
            "estimated_input_tokens": reservation.estimated_input_tokens,
            "max_output_tokens": reservation.max_output_tokens,
            "worst_case_cost_usd": _dstr(reservation.worst_case_cost_usd),
            "price_estimate_usd": _dstr(estimate),
            "cost_usd": _dstr(cost),
            "cost_source": source,
            "spend_total_usd": _dstr(self.spend_usd),
        }

    def _check_reached(self) -> None:
        """Mark the budget exhausted as soon as recorded usage reaches a limit."""
        if self.exhausted_by is not None:
            return
        lim = self.limits
        if self.attempts >= lim.max_attempts:
            self.exhausted_by = "max_attempts"
        elif self.input_tokens >= lim.max_input_tokens:
            self.exhausted_by = "max_input_tokens"
        elif lim.max_spend_usd is not None and self.spend_usd >= lim.max_spend_usd:  # type: ignore[operator]
            self.exhausted_by = "max_spend_usd"

    def release(self, reservation: Reservation) -> None:
        """Drop a reservation whose request was never sent."""
        self._take(reservation)

    # ------------------------------------------------------------- sources

    def admit_sources(self, requested: int) -> int:
        """Admit up to ``requested`` more sources; returns how many were admitted.
        The source limit restricts what is sent to the model; it does not stop
        attempts for sources already admitted."""
        if requested < 0:
            raise ValueError("requested must be >= 0")
        admitted = max(0, min(requested, self.limits.max_sources - self.sources))
        self.sources += admitted
        if admitted < requested:
            self.sources_limited = True
        return admitted

    # ------------------------------------------------------------- reporting

    @property
    def remaining_attempts(self) -> int:
        if self.exhausted_by is not None:
            return 0
        return max(0, self.limits.max_attempts - self.attempts - len(self._pending))

    def summary(self) -> dict[str, Any]:
        lim = self.limits
        cap = lim.max_spend_usd
        return {
            "limits": lim.to_dict(),
            "used": {
                "attempts": self.attempts,
                "failed_attempts": self.failed_attempts,
                "logical_calls": self.logical_calls,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "output_side_tokens_charged": self.output_tokens,
                "output_tokens_reported": self.output_tokens_reported,
                "reasoning_tokens": self.reasoning_tokens,
                "reported_cost_usd": _dstr(self.reported_cost_usd) if self.reported_cost_attempts else None,
                "reported_cost_attempts": self.reported_cost_attempts,
                "sources": self.sources,
                "spend_usd": _dstr(self.spend_usd),
                "spend_by_cost_source_usd": {k: _dstr(v) for k, v in self.spend_by_source.items()},
                "unpriced_attempts": self.unpriced_attempts,
            },
            "remaining": {
                "attempts": self.remaining_attempts,
                "input_tokens": max(0, lim.max_input_tokens - self.input_tokens),
                "sources": max(0, lim.max_sources - self.sources),
                "spend_usd": None if cap is None
                else _dstr(max(Decimal(0), cap - self.spend_usd)),  # type: ignore[operator]
            },
            "prices_configured": self.prices.configured,
            "spend_cap_enabled": cap is not None,
            "financial_guard": ("SCIFORGE_MAX_SPEND_USD is the hard financial guard; output-token settings are sent "
                                "as max_output_tokens and are not a guaranteed ceiling on billed tokens"),
            "sources_limited": self.sources_limited,
            "exhausted_by": self.exhausted_by,
        }


def budgeted_call(
    client: ModelClient,
    request: ModelRequest,
    tracker: BudgetTracker,
    *,
    retry: RetryPolicy | None = None,
    audit: ModelCallAudit | None = None,
    keep_attempts: int = 0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    now: Callable[[], datetime] = utc_now,
) -> ModelResponse:
    """The only sanctioned way to call a model: one logical call, budgeted per attempt.

    For each attempt: ``tracker.reserve`` (raises :class:`BudgetExhausted`
    BEFORE sending) → ``client.complete`` (one HTTP attempt) → ``tracker.record``
    → audit entry. Retryable errors are retried up to ``retry.max_retries``
    times, but only if the next attempt also fits the budget; otherwise
    :class:`BudgetExhausted` is raised with ``last_error`` set.
    """
    retry = retry or RetryPolicy()
    tracker.logical_calls += 1
    call_id = tracker.logical_calls
    attempt = 1
    reservation = _reserve(tracker, request, keep_attempts, None, audit, call_id, attempt, now, client)
    while True:
        started = clock()
        ts = iso_utc(now())
        try:
            response = client.complete(reservation.request)
        except ModelError as exc:
            error: ModelError = exc
        except Exception as exc:  # unknown failure: may have been billed → worst case
            accounting = tracker.record(reservation, None, failed=True)
            if audit is not None:
                audit.add_attempt(reservation.request, call_id=call_id, attempt=attempt, timestamp=ts,
                                  latency_s=clock() - started, error=exc, accounting=accounting,
                                  provider=client.name, model=client.model)
            raise
        else:
            accounting = tracker.record(reservation, response.usage)
            if audit is not None:
                audit.add_attempt(reservation.request, call_id=call_id, attempt=attempt, timestamp=ts,
                                  latency_s=clock() - started, response=response, accounting=accounting,
                                  provider=client.name, model=client.model)
            return replace(response, attempts=attempt)

        # ---- failed attempt: always recorded (see module docstring for charging rules)
        accounting = tracker.record(reservation, error.usage, failed=True)
        error.attempts = attempt
        latency = clock() - started
        can_retry = error.retryable and attempt <= retry.max_retries
        next_reservation: Reservation | None = None
        blocked: BudgetExhausted | None = None
        backoff: float | None = None
        if can_retry:
            try:
                next_reservation = tracker.reserve(request, keep_attempts=keep_attempts, last_error=error)
            except BudgetExhausted as stop:
                blocked = stop
            else:
                backoff = retry.delay(attempt - 1, error.retry_after_s)
        if audit is not None:
            audit.add_attempt(reservation.request, call_id=call_id, attempt=attempt, timestamp=ts,
                              latency_s=latency, error=error, accounting=accounting, provider=client.name,
                              model=client.model, will_retry=next_reservation is not None, backoff_s=backoff,
                              retry_blocked_by=blocked.limit if blocked else None)
        if blocked is not None:
            if audit is not None:
                audit.add_budget_stop(request, blocked, call_id=call_id, blocked_attempt=attempt + 1,
                                      timestamp=iso_utc(now()), provider=client.name, model=client.model)
            raise blocked from error
        if next_reservation is None:
            raise error
        sleep(backoff or 0.0)
        reservation = next_reservation
        attempt += 1


def _reserve(tracker: BudgetTracker, request: ModelRequest, keep_attempts: int, last_error: ModelError | None,
             audit: ModelCallAudit | None, call_id: int, attempt: int, now: Callable[[], datetime],
             client: ModelClient) -> Reservation:
    try:
        return tracker.reserve(request, keep_attempts=keep_attempts, last_error=last_error)
    except BudgetExhausted as stop:
        if audit is not None:
            audit.add_budget_stop(request, stop, call_id=call_id, blocked_attempt=attempt,
                                  timestamp=iso_utc(now()), provider=client.name, model=client.model)
        raise
