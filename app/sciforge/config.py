"""Runtime configuration, read only from environment variables.

Secrets (``NCBI_API_KEY``) are never read from files in the repository and are
excluded from ``repr`` so they cannot leak through logging of the settings
object.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field

from sciforge import __version__

PROJECT_URL = "https://github.com/SaraSheikhlary/sciforge-open"

DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_BACKOFF_SECONDS = 1.0

# NCBI allows 3 requests/s without an API key and 10 requests/s with one.
# A small safety margin is added to each interval.
PUBMED_INTERVAL_NO_KEY = 0.34
PUBMED_INTERVAL_WITH_KEY = 0.11
# Crossref's public pool is more restrictive than the "polite" pool used when a
# mailto address is supplied; stay well below both.
CROSSREF_INTERVAL_NO_EMAIL = 0.2
CROSSREF_INTERVAL_WITH_EMAIL = 0.1

# Candidate pool: records requested per expanded query and database before deduplication and selection
# (SCIFORGE_CANDIDATE_POOL_PER_QUERY). Each query requests max(pool, max_results) records.
DEFAULT_CANDIDATE_POOL_PER_QUERY = 10
MIN_CANDIDATE_POOL_PER_QUERY = 1
MAX_CANDIDATE_POOL_PER_QUERY = 100


class ConfigError(ValueError):
    """Raised when an environment variable has an invalid value."""


def _clean(value: str | None) -> str | None:
    """Return a stripped string, or None for missing / blank values."""
    if value is None:
        return None
    value = value.strip()
    return value or None


def _parse_float(env: Mapping[str, str], name: str, default: float, lo: float, hi: float) -> float:
    raw = _clean(env.get(name))
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number") from exc
    if not lo <= value <= hi:
        raise ConfigError(f"{name} must be between {lo} and {hi}")
    return value


def _parse_int(env: Mapping[str, str], name: str, default: int, lo: int, hi: int) -> int:
    raw = _clean(env.get(name))
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc
    if not lo <= value <= hi:
        raise ConfigError(f"{name} must be between {lo} and {hi}")
    return value


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings.

    Attributes:
        ncbi_api_key: Optional NCBI E-utilities key (``NCBI_API_KEY``).
        contact_email: Optional contact address sent to NCBI and Crossref
            (``SCIFORGE_CONTACT_EMAIL``).
        timeout_seconds: Per-request timeout (``SCIFORGE_TIMEOUT_SECONDS``).
        max_retries: Retries after the first attempt for 429 / 5xx / timeouts /
            connection errors (``SCIFORGE_MAX_RETRIES``).
        backoff_seconds: Base for exponential backoff between retries
            (``SCIFORGE_BACKOFF_SECONDS``).
        query_expansion: Deterministic rule-based query expansion before
            retrieval (``SCIFORGE_QUERY_EXPANSION``, default true; see
            :mod:`sciforge.query_expansion`). False = question used verbatim only.
        candidate_pool_per_query: Records requested per expanded query and
            database before deduplication and deterministic selection
            (``SCIFORGE_CANDIDATE_POOL_PER_QUERY``, default 10, integer 1-100;
            invalid values raise :class:`ConfigError`). Each query requests
            ``max(candidate_pool_per_query, max_results)`` records.
    """

    ncbi_api_key: str | None = field(default=None, repr=False)
    contact_email: str | None = field(default=None, repr=False)
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS
    query_expansion: bool = True
    candidate_pool_per_query: int = DEFAULT_CANDIDATE_POOL_PER_QUERY

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        """Build settings from ``environ`` (defaults to ``os.environ``)."""
        env = os.environ if environ is None else environ
        email = _clean(env.get("SCIFORGE_CONTACT_EMAIL"))
        if email is not None and ("@" not in email or any(c.isspace() for c in email)):
            raise ConfigError("SCIFORGE_CONTACT_EMAIL does not look like an email address")
        return cls(
            ncbi_api_key=_clean(env.get("NCBI_API_KEY")),
            contact_email=email,
            timeout_seconds=_parse_float(env, "SCIFORGE_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS, 0.1, 300.0),
            max_retries=_parse_int(env, "SCIFORGE_MAX_RETRIES", DEFAULT_MAX_RETRIES, 0, 5),
            backoff_seconds=_parse_float(env, "SCIFORGE_BACKOFF_SECONDS", DEFAULT_BACKOFF_SECONDS, 0.0, 60.0),
            query_expansion=_parse_bool(env, "SCIFORGE_QUERY_EXPANSION", True),
            candidate_pool_per_query=_parse_int(env, "SCIFORGE_CANDIDATE_POOL_PER_QUERY",
                                                DEFAULT_CANDIDATE_POOL_PER_QUERY, MIN_CANDIDATE_POOL_PER_QUERY,
                                                MAX_CANDIDATE_POOL_PER_QUERY),
        )

    @property
    def user_agent(self) -> str:
        """User-Agent header; includes ``mailto:`` only when an email is set."""
        details = PROJECT_URL
        if self.contact_email:
            details += f"; mailto:{self.contact_email}"
        return f"SciForge/{__version__} ({details})"

    @property
    def pubmed_min_interval(self) -> float:
        """Minimum seconds between NCBI requests."""
        return PUBMED_INTERVAL_WITH_KEY if self.ncbi_api_key else PUBMED_INTERVAL_NO_KEY

    @property
    def crossref_min_interval(self) -> float:
        """Minimum seconds between Crossref requests."""
        return CROSSREF_INTERVAL_WITH_EMAIL if self.contact_email else CROSSREF_INTERVAL_NO_EMAIL

    def secret_values(self) -> list[str]:
        """Values that must never appear in logs or output files."""
        return [v for v in (self.ncbi_api_key, self.contact_email) if v]


# ============================================================ v0.3 model layer
#
# Everything below is used only when the model layer is requested. v0.2's
# ``Settings.from_env`` above never reads any of these variables, and
# ``ModelSettings.from_env`` is never called on the v0.2 path.

DEFAULT_XAI_BASE_URL = "https://api.x.ai"
XAI_RESPONSES_PATH = "/v1/responses"

# D8 budget defaults (per investigation).
DEFAULT_MODEL_MAX_ATTEMPTS = 15  # every API attempt counts, retries included
DEFAULT_MODEL_MAX_SOURCES = 10
DEFAULT_MODEL_MAX_INPUT_TOKENS = 200_000   # cumulative input tokens per investigation
DEFAULT_MODEL_MAX_OUTPUT_TOKENS = 2_000   # global per-call output cap (fallback for every stage)
MIN_MODEL_MAX_OUTPUT_TOKENS = 16
MAX_MODEL_MAX_OUTPUT_TOKENS = 128_000
DEFAULT_MAX_SPEND_USD = 15.0
# Per-request xAI timeout (SCIFORGE_MODEL_TIMEOUT_SECONDS). Values that are blank, not a number, not finite,
# <= 0 or above MAX_MODEL_TIMEOUT_SECONDS fall back to the default (never an error, never an unbounded wait).
DEFAULT_MODEL_TIMEOUT_SECONDS = 120.0
MAX_MODEL_TIMEOUT_SECONDS = 600.0
MODEL_TIMEOUT_ENV = "SCIFORGE_MODEL_TIMEOUT_SECONDS"
MAX_SPEND_DISABLED_WORDS = frozenset({"none"})

# Model stages -> environment-variable suffix. Logical stage names are those of
# sciforge.llm.client.stage_key ("evidence" is the extraction stage).
MODEL_STAGES = ("question", "evidence", "gaps", "hypotheses", "report")
OUTPUT_TOKENS_ENV_PREFIX = "SCIFORGE_MODEL_MAX_OUTPUT_TOKENS_"
# Per-stage output caps: blank/unset -> global SCIFORGE_MODEL_MAX_OUTPUT_TOKENS; otherwise an integer
# 16-128000 (anything else is a configuration error, like the global setting).
OUTPUT_TOKENS_STAGE_ENV = {stage: OUTPUT_TOKENS_ENV_PREFIX + stage.upper() for stage in MODEL_STAGES}
# Reasoning effort sent to the xAI Responses API as {"reasoning": {"effort": ...}}.
REASONING_EFFORT_ENV = "SCIFORGE_MODEL_REASONING_EFFORT"
REASONING_EFFORTS = ("low", "medium", "high", "xhigh")
DEFAULT_REASONING_EFFORT = "high"
# The question stage always uses the global value; these stages can override it.
REASONING_EFFORT_STAGES = ("evidence", "gaps", "hypotheses", "report")
REASONING_EFFORT_STAGE_ENV = {stage: f"{REASONING_EFFORT_ENV}_{stage.upper()}" for stage in REASONING_EFFORT_STAGES}

ELIGIBILITY_VERIFIED = "verified"
ELIGIBILITY_VERIFIED_OR_PARTIAL = "verified_or_partial"

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})

# Environment variables read by ModelSettings.from_env (documented in .env.example).
MODEL_ENV_VARS = (
    "XAI_API_KEY", "XAI_MODEL", "XAI_BASE_URL",
    "SCIFORGE_MODEL_MAX_ATTEMPTS", "SCIFORGE_MODEL_MAX_SOURCES", "SCIFORGE_MODEL_MAX_INPUT_TOKENS",
    "SCIFORGE_MODEL_MAX_OUTPUT_TOKENS", "SCIFORGE_MAX_SPEND_USD",
    "SCIFORGE_PRICE_INPUT_PER_MTOK", "SCIFORGE_PRICE_OUTPUT_PER_MTOK",
    "SCIFORGE_MODEL_TIMEOUT_SECONDS", "SCIFORGE_STORE_PROMPTS",
    "SCIFORGE_MODEL_ELIGIBILITY", "SCIFORGE_MODEL_ENTAILMENT",
    *OUTPUT_TOKENS_STAGE_ENV.values(), REASONING_EFFORT_ENV, *REASONING_EFFORT_STAGE_ENV.values(),
)


def _parse_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = _clean(env.get(name))
    if raw is None:
        return default
    lowered = raw.lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    raise ConfigError(f"{name} must be one of 1/0/true/false/yes/no/on/off")


def parse_model_timeout(env: Mapping[str, str]) -> float:
    """``SCIFORGE_MODEL_TIMEOUT_SECONDS``: positive finite seconds, at most 600; default 120.

    Unlike other settings this one never raises: blank, non-numeric, non-finite (nan/inf), non-positive
    or > 600 values fall back to the 120 s default, so a typo can neither break a run nor disable the
    timeout. The raw value is never logged or echoed.
    """
    raw = _clean(env.get(MODEL_TIMEOUT_ENV))
    if raw is None:
        return DEFAULT_MODEL_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_MODEL_TIMEOUT_SECONDS
    if not math.isfinite(value) or value <= 0 or value > MAX_MODEL_TIMEOUT_SECONDS:
        return DEFAULT_MODEL_TIMEOUT_SECONDS
    return value


def _parse_optional_int(env: Mapping[str, str], name: str, lo: int, hi: int) -> int | None:
    """Blank/unset -> None (caller falls back to a global value); otherwise like :func:`_parse_int`."""
    if _clean(env.get(name)) is None:
        return None
    return _parse_int(env, name, 0, lo, hi)


def _parse_reasoning_effort(env: Mapping[str, str], name: str, default: str | None) -> str | None:
    """``low`` / ``medium`` / ``high`` / ``xhigh`` (case-insensitive); blank -> ``default``; else ConfigError."""
    raw = _clean(env.get(name))
    if raw is None:
        return default
    value = raw.lower()
    if value not in REASONING_EFFORTS:
        raise ConfigError(f"{name} must be one of {', '.join(REASONING_EFFORTS)}")
    return value


def _parse_optional_float(env: Mapping[str, str], name: str, lo: float, hi: float) -> float | None:
    if _clean(env.get(name)) is None:
        return None
    return _parse_float(env, name, 0.0, lo, hi)


def _parse_spend_cap(env: Mapping[str, str]) -> float | None:
    """``SCIFORGE_MAX_SPEND_USD``: default 15; ``none`` disables; must be > 0."""
    name = "SCIFORGE_MAX_SPEND_USD"
    raw = _clean(env.get(name))
    if raw is None:
        return DEFAULT_MAX_SPEND_USD
    if raw.lower() in MAX_SPEND_DISABLED_WORDS:
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number of US dollars or 'none'") from exc
    if not 0.0 < value <= 10_000.0:
        raise ConfigError(f"{name} must be greater than 0 and at most 10000 (use 'none' to disable the cap)")
    return value


def _normalize_base_url(raw: str | None) -> str:
    url = (raw or DEFAULT_XAI_BASE_URL).strip().rstrip("/")
    lowered = url.lower()
    if not lowered.startswith("https://"):
        raise ConfigError("XAI_BASE_URL must start with https://")
    host = url[len("https://"):]
    if not host or "@" in host or "?" in host or "#" in host or any(c.isspace() for c in host):
        raise ConfigError("XAI_BASE_URL must be a plain https URL without credentials, query, or fragment")
    if lowered.endswith("/v1"):
        url = url[:-3]
    return url


@dataclass(frozen=True)
class ModelSettings:
    """Settings for the optional v0.3 model layer (xAI Responses API).

    Build with :meth:`from_env` only when the model layer is requested. The API
    key is excluded from ``repr``/``str`` and must never be logged or stored.
    """

    api_key: str = field(repr=False)
    model: str
    base_url: str = DEFAULT_XAI_BASE_URL
    max_attempts: int = DEFAULT_MODEL_MAX_ATTEMPTS
    max_sources: int = DEFAULT_MODEL_MAX_SOURCES
    max_input_tokens: int = DEFAULT_MODEL_MAX_INPUT_TOKENS
    max_output_tokens_per_call: int = DEFAULT_MODEL_MAX_OUTPUT_TOKENS
    max_spend_usd: float | None = DEFAULT_MAX_SPEND_USD
    price_input_per_mtok: float | None = None
    price_output_per_mtok: float | None = None
    timeout_seconds: float = DEFAULT_MODEL_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS
    store_prompts: bool = True
    eligibility: str = ELIGIBILITY_VERIFIED
    entailment: bool = True
    # Stage-aware output caps (None -> max_output_tokens_per_call) and reasoning effort.
    max_output_tokens_question: int | None = None
    max_output_tokens_evidence: int | None = None
    max_output_tokens_gaps: int | None = None
    max_output_tokens_hypotheses: int | None = None
    max_output_tokens_report: int | None = None
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    reasoning_effort_evidence: str | None = None
    reasoning_effort_gaps: str | None = None
    reasoning_effort_hypotheses: str | None = None
    reasoning_effort_report: str | None = None

    def __post_init__(self) -> None:
        from sciforge.llm.client import ModelConfigError

        for stage in MODEL_STAGES:
            value = getattr(self, f"max_output_tokens_{stage}")
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)
                                      or not MIN_MODEL_MAX_OUTPUT_TOKENS <= value <= MAX_MODEL_MAX_OUTPUT_TOKENS):
                raise ModelConfigError(f"{OUTPUT_TOKENS_STAGE_ENV[stage]} must be between "
                                       f"{MIN_MODEL_MAX_OUTPUT_TOKENS} and {MAX_MODEL_MAX_OUTPUT_TOKENS}")
        if self.reasoning_effort not in REASONING_EFFORTS:
            raise ModelConfigError(f"{REASONING_EFFORT_ENV} must be one of {', '.join(REASONING_EFFORTS)}")
        for stage in REASONING_EFFORT_STAGES:
            value = getattr(self, f"reasoning_effort_{stage}")
            if value is not None and value not in REASONING_EFFORTS:
                raise ModelConfigError(f"{REASONING_EFFORT_STAGE_ENV[stage]} must be one of "
                                       f"{', '.join(REASONING_EFFORTS)}")

        if not self.api_key or not self.api_key.strip():
            raise ModelConfigError("XAI_API_KEY is required for the model layer")
        if not self.model or not self.model.strip():
            raise ModelConfigError("XAI_MODEL is required for the model layer (no default model)")
        if self.eligibility not in (ELIGIBILITY_VERIFIED, ELIGIBILITY_VERIFIED_OR_PARTIAL):
            raise ModelConfigError("SCIFORGE_MODEL_ELIGIBILITY must be 'verified' or 'verified_or_partial'")
        if self.max_spend_usd is not None and not self.prices_configured:
            raise ModelConfigError(
                "SCIFORGE_MAX_SPEND_USD is enabled but SCIFORGE_PRICE_INPUT_PER_MTOK and "
                "SCIFORGE_PRICE_OUTPUT_PER_MTOK are not both set; set both prices (USD per 1M tokens, "
                "from your xAI console) or disable the cap with SCIFORGE_MAX_SPEND_USD=none"
            )

    def __str__(self) -> str:
        return repr(self)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> ModelSettings:
        """Read model settings from ``environ``; raise ``ModelConfigError`` if invalid.

        Call only when the model layer is requested; it requires ``XAI_API_KEY``
        and ``XAI_MODEL`` and fails closed when the spend cap has no prices.
        """
        from sciforge.llm.client import ModelConfigError

        env = os.environ if environ is None else environ
        try:
            api_key = _clean(env.get("XAI_API_KEY"))
            model = _clean(env.get("XAI_MODEL"))
            if api_key is None:
                raise ModelConfigError("XAI_API_KEY is required for the model layer (read from the environment only)")
            if model is None:
                raise ModelConfigError("XAI_MODEL is required for the model layer (no default model)")
            eligibility = (_clean(env.get("SCIFORGE_MODEL_ELIGIBILITY")) or ELIGIBILITY_VERIFIED).lower()
            return cls(
                api_key=api_key,
                model=model,
                base_url=_normalize_base_url(_clean(env.get("XAI_BASE_URL"))),
                max_attempts=_parse_int(env, "SCIFORGE_MODEL_MAX_ATTEMPTS", DEFAULT_MODEL_MAX_ATTEMPTS, 1, 500),
                max_sources=_parse_int(env, "SCIFORGE_MODEL_MAX_SOURCES", DEFAULT_MODEL_MAX_SOURCES, 1, 100),
                max_input_tokens=_parse_int(env, "SCIFORGE_MODEL_MAX_INPUT_TOKENS", DEFAULT_MODEL_MAX_INPUT_TOKENS,
                                            1_000, 10_000_000),
                max_output_tokens_per_call=_parse_int(env, "SCIFORGE_MODEL_MAX_OUTPUT_TOKENS",
                                                      DEFAULT_MODEL_MAX_OUTPUT_TOKENS, MIN_MODEL_MAX_OUTPUT_TOKENS,
                                                      MAX_MODEL_MAX_OUTPUT_TOKENS),
                **{f"max_output_tokens_{stage}": _parse_optional_int(env, name, MIN_MODEL_MAX_OUTPUT_TOKENS,
                                                                     MAX_MODEL_MAX_OUTPUT_TOKENS)
                   for stage, name in OUTPUT_TOKENS_STAGE_ENV.items()},
                reasoning_effort=_parse_reasoning_effort(env, REASONING_EFFORT_ENV, DEFAULT_REASONING_EFFORT),
                **{f"reasoning_effort_{stage}": _parse_reasoning_effort(env, name, None)
                   for stage, name in REASONING_EFFORT_STAGE_ENV.items()},
                max_spend_usd=_parse_spend_cap(env),
                price_input_per_mtok=_parse_optional_float(env, "SCIFORGE_PRICE_INPUT_PER_MTOK", 0.0, 10_000.0),
                price_output_per_mtok=_parse_optional_float(env, "SCIFORGE_PRICE_OUTPUT_PER_MTOK", 0.0, 10_000.0),
                timeout_seconds=parse_model_timeout(env),
                max_retries=_parse_int(env, "SCIFORGE_MAX_RETRIES", DEFAULT_MAX_RETRIES, 0, 5),
                backoff_seconds=_parse_float(env, "SCIFORGE_BACKOFF_SECONDS", DEFAULT_BACKOFF_SECONDS, 0.0, 60.0),
                store_prompts=_parse_bool(env, "SCIFORGE_STORE_PROMPTS", True),
                eligibility=eligibility,
                entailment=_parse_bool(env, "SCIFORGE_MODEL_ENTAILMENT", True),
            )
        except ModelConfigError:
            raise
        except ConfigError as exc:
            raise ModelConfigError(str(exc)) from None

    @property
    def prices_configured(self) -> bool:
        return self.price_input_per_mtok is not None and self.price_output_per_mtok is not None

    @property
    def responses_url(self) -> str:
        """Full URL of the Responses endpoint."""
        return self.base_url + XAI_RESPONSES_PATH

    @property
    def include_partially_verified(self) -> bool:
        """D2: partially verified sources only by explicit opt-in."""
        return self.eligibility == ELIGIBILITY_VERIFIED_OR_PARTIAL

    @property
    def user_agent(self) -> str:
        """User-Agent for model requests (no contact email)."""
        return f"SciForge/{__version__} ({PROJECT_URL})"

    def stage_max_output_tokens(self) -> dict[str, int]:
        """Explicit per-stage output caps (only stages whose override is set)."""
        return {stage: v for stage in MODEL_STAGES if (v := getattr(self, f"max_output_tokens_{stage}")) is not None}

    def max_output_tokens_for(self, stage: str | None) -> int:
        """Effective output cap for ``stage``: the stage override if set, else the global value."""
        from sciforge.llm.client import stage_key

        return self.stage_max_output_tokens().get(stage_key(stage) or "", self.max_output_tokens_per_call)

    def reasoning_effort_for(self, stage: str | None) -> str:
        """Effective reasoning effort: stage override (not for ``question``), else the global value."""
        from sciforge.llm.client import stage_key

        key = stage_key(stage)
        override = getattr(self, f"reasoning_effort_{key}", None) if key in REASONING_EFFORT_STAGES else None
        return override or self.reasoning_effort

    def reasoning_efforts(self) -> dict[str, str]:
        """Effective reasoning effort for every model stage (recorded in logs; a setting, not model output)."""
        return {stage: self.reasoning_effort_for(stage) for stage in MODEL_STAGES}

    def budget_limits(self):  # -> sciforge.llm.budget.BudgetLimits
        from sciforge.llm.budget import BudgetLimits

        return BudgetLimits(
            max_attempts=self.max_attempts,
            max_sources=self.max_sources,
            max_input_tokens=self.max_input_tokens,
            max_output_tokens_per_call=self.max_output_tokens_per_call,
            max_spend_usd=self.max_spend_usd,
            stage_max_output_tokens=tuple(self.stage_max_output_tokens().items()),
        )

    def retry_policy(self):  # -> sciforge.llm.budget.RetryPolicy
        from sciforge.llm.budget import RetryPolicy

        return RetryPolicy(max_retries=self.max_retries, backoff_seconds=self.backoff_seconds)

    def price_table(self):  # -> sciforge.llm.budget.PriceTable
        from sciforge.llm.budget import PriceTable

        return PriceTable(input_per_mtok=self.price_input_per_mtok, output_per_mtok=self.price_output_per_mtok)

    def secret_values(self) -> list[str]:
        """Values that must never appear in logs, audit records, or output files."""
        return [self.api_key] if self.api_key else []
