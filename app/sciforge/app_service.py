"""Application interface for the SciForge web app (Streamlit UI calls ONLY this module).

The UI (``streamlit_app.py`` at the repository root) contains no scientific
logic. It collects inputs, calls :func:`run_web_investigation` and renders the
returned :class:`WebInvestigationResult`. This module adds no business logic of
its own either: it validates inputs, picks the model client, runs the existing
pipelines and reshapes their results for display.

* **Demo Mode** (default): bundled SYNTHETIC v0.2-shaped records
  (:mod:`sciforge.demo_data`) + a scripted :class:`~sciforge.llm.fake.FakeModelClient`,
  run through the real :func:`~sciforge.investigation_pipeline.run_model_investigation`
  (source texts, deterministic validation, gaps, hypotheses, report). Fully
  offline: abstracts are served by an ``httpx.MockTransport``; no API key.
* **Live Mode**: v0.2 :func:`~sciforge.pipeline.run_investigation` (PubMed +
  Crossref, verification) followed by ``run_model_investigation`` with the
  existing :class:`~sciforge.llm.xai.XAIClient` and the existing attempt / token /
  spend budgets from :class:`~sciforge.config.ModelSettings`. Enabled only when
  the deployment gate ``SCIFORGE_LIVE_ENABLED`` is exactly ``true`` (default
  false) AND both ``XAI_API_KEY`` and ``XAI_MODEL`` are present (environment
  variables first, then the optional ``secrets`` mapping, e.g. ``st.secrets``).
  The gate is enforced here as well as in the UI (live runs are refused). Presence is
  checked; values are never returned, rendered or logged. NOT validated against
  the real API yet.
* **Live Mode sign-in** (on top of the gate, never instead of it):
  ``SCIFORGE_LIVE_REQUIRE_AUTH`` (default true; ONLY the exact value ``false``,
  case-insensitive, trimmed, disables it — anything else, including a TOML
  boolean, keeps it on) requires a Streamlit OIDC login (``st.login()``; ``[auth]``
  section in secrets + the ``Authlib`` package) AND an email on the
  ``SCIFORGE_LIVE_ALLOWED_EMAILS`` allowlist (comma-separated; environment first,
  then secrets; trimmed, case-insensitive exact match; empty allowlist = nobody).
  The UI builds a :class:`LiveIdentity` from ``st.user`` (``is_logged_in``,
  ``email``, ``email_verified`` only — identity-provider tokens are never read)
  and passes it in :class:`InvestigationRequest`; this module recomputes the
  :class:`LiveAccessDecision` itself (allowlist and auth configuration from
  env/secrets, never from the request) and refuses unauthorised live runs
  before any literature lookup or model-client creation. Only the decision
  (``allowed`` / ``denied:<reason>``) is logged — never an email or token.
* **Kill switch** ``SCIFORGE_LIVE_KILL_SWITCH`` (default off): checked FIRST, in
  :func:`live_availability` (UI) and in :func:`run_web_investigation`. OFF only when
  unset, empty, the string ``false`` (case-insensitive, trimmed) or a TOML ``false``;
  ANY other value (``true``, ``1``, ``yes``, ``on``, typos, TOML ``true``) turns it ON
  and makes Live Mode unavailable regardless of every other setting (fail safe).
* **Usage limit** (sign-in required only): at most ``SCIFORGE_LIVE_MAX_RUNS_PER_USER``
  (default 3) Live runs per verified email per rolling 24 h, recorded at the start of
  an authorised run through a pluggable backend (``SCIFORGE_LIVE_QUOTA_BACKEND``):
  ``file`` (default; hashed, file-locked store outside the repository,
  :mod:`sciforge.live_quota`) or ``postgres`` (Streamlit SQL connection
  ``[connections.<SCIFORGE_LIVE_QUOTA_CONNECTION>]``, :mod:`sciforge.live_quota_sql`).
  A missing/invalid backend configuration makes Live unavailable; any store or
  database error refuses the run (fail closed). With ``SCIFORGE_LIVE_REQUIRE_AUTH=false`` there is no
  quota (behaviour unchanged).
* **Spend cap default** for web Live runs: when ``SCIFORGE_MAX_SPEND_USD`` is unset or
  blank, the web app uses ``2`` USD (:data:`WEB_LIVE_DEFAULT_MAX_SPEND_USD`); the
  CLI/library default stays 15. An explicit value (including ``none``) is honoured.

Privacy
-------
* Run artifacts (the pipelines always write JSON/Markdown files) go to a fresh
  directory under the system temp dir (never the repository) that is deleted
  before :func:`run_web_investigation` returns. Questions are not persisted.
* Every string in the result passes :func:`guard_display` — secret redaction
  (:func:`sciforge.logging_utils.redact_text`), filesystem-path masking and
  withholding of any raw rejected model output (needles from
  :func:`sciforge.output_guard.rejected_text_values`). The run directory is also
  checked with :func:`sciforge.output_guard.find_leaks` before deletion.
* Citations come only from :func:`sciforge.stages.report.render_citation` /
  :func:`~sciforge.stages.report.unresolved_marker` over the v0.2 records.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import httpx

from sciforge.config import ConfigError, ModelSettings, Settings
from sciforge.investigation_pipeline import InvestigationModelResult, run_model_investigation
from sciforge.live_quota import (
    BACKEND_POSTGRES,
    DEFAULT_QUOTA_CONNECTION,
    MAX_RUNS_ENV,
    QUOTA_BACKEND_ENV,
    QUOTA_CONNECTION_ENV,
    QUOTA_PATH_ENV,
    QUOTA_SALT_ENV,
    QuotaConfigError,
    QuotaBackend,
    QuotaDecision,
    QuotaStoreError,
    check_and_record,
    parse_quota_backend,
    parse_quota_settings,
)
from sciforge.live_quota_sql import SqlQuotaBackend
from sciforge.llm.client import ModelClient, ModelConfigError, ModelRequest, ModelResponse
from sciforge.logging_utils import get_logger, redact_text, utc_now
from sciforge.models import Record, SearchOutcome, VerificationResult
from sciforge.evidence_graph import graph_summary
from sciforge.output_guard import find_leaks, rejected_text_values
from sciforge.stages.claim_checks import SEMANTIC_STATUS
from sciforge.stages.report import (
    HYPOTHESIS_DISCLAIMER,
    PREPRINT_BADGE,
    SECTION_TITLES,
    render_citation,
    source_type_badge,
    unresolved_marker,
)

__all__ = [
    "MAX_QUESTION_CHARS", "MAX_SOURCES_LIMIT", "MIN_QUESTION_CHARS", "MIN_SOURCES", "MODE_DEMO", "MODE_LIVE",
    "PROGRESS_STAGES", "InvestigationRequest", "LiveAccessDecision", "LiveAvailability", "LiveIdentity",
    "ProgressEvent", "WebInvestigationResult", "auth_section_configured", "authlib_available", "decide_live_access",
    "guard_display", "identity_from_user", "live_allowed_emails", "live_availability", "live_gate_enabled",
    "live_require_auth", "render_sources", "run_web_investigation", "streamlit_auth_configured", "validate_request",
]

MODE_DEMO = "demo"
MODE_LIVE = "live"
MIN_QUESTION_CHARS = 10
MAX_QUESTION_CHARS = 2000
MIN_SOURCES = 1
MAX_SOURCES_LIMIT = 20
DEFAULT_MAX_SOURCES = 5
MIN_YEAR = 1800
MAX_YEAR = 2100
LIVE_CREDENTIAL_NAMES = ("XAI_API_KEY", "XAI_MODEL")
# Deployment gate: Live Mode is off unless this is exactly "true" (case-insensitive, trimmed). Default false.
LIVE_ENABLED_NAME = "SCIFORGE_LIVE_ENABLED"
# Live Mode sign-in (on top of the gate). Fail closed: only the exact value "false" disables the requirement.
LIVE_REQUIRE_AUTH_NAME = "SCIFORGE_LIVE_REQUIRE_AUTH"
LIVE_ALLOWED_EMAILS_NAME = "SCIFORGE_LIVE_ALLOWED_EMAILS"
# Emergency stop, checked before everything else. Fail safe: only unset/empty/"false" means off.
LIVE_KILL_SWITCH_NAME = "SCIFORGE_LIVE_KILL_SWITCH"
# Web Live Mode spend cap when SCIFORGE_MAX_SPEND_USD is unset/blank (public, experimental default).
WEB_LIVE_DEFAULT_MAX_SPEND_USD = "2"
# Keys of the Streamlit [auth] secrets section (default provider) that must all be non-empty.
AUTH_REQUIRED_KEYS = ("redirect_uri", "cookie_secret", "client_id", "client_secret", "server_metadata_url")
# Every name the app may read from st.secrets (env vars take precedence for each).
LIVE_SETTING_NAMES = (LIVE_KILL_SWITCH_NAME, LIVE_ENABLED_NAME, *LIVE_CREDENTIAL_NAMES, LIVE_REQUIRE_AUTH_NAME,
                      LIVE_ALLOWED_EMAILS_NAME, MAX_RUNS_ENV, QUOTA_PATH_ENV, QUOTA_SALT_ENV, QUOTA_BACKEND_ENV,
                      QUOTA_CONNECTION_ENV)
# Minimum keys of a Streamlit SQL connection section (st.connection(name, type="sql")): a url, or these.
SQL_CONNECTION_REQUIRED_KEYS = ("dialect", "username", "host")
QUOTA_DB_UNAVAILABLE_MESSAGE = ("Live Mode is unavailable: the usage-limit database is not configured for this "
                                "deployment. Demo Mode remains available.")
WITHHELD = "[withheld by privacy guard]"
PATH_MASK = "[path]"
TEMP_PREFIX = "sciforge-web-"

# ------------------------------------------------------------------ user-facing wording (UI + tests)
DEMO_MODE_DESCRIPTION = ("Demo Mode — synthetic, offline demonstration: no real literature search and no xAI "
                         "calls. It always analyses the same bundled synthetic records with a scripted fake model; "
                         "nothing it shows is a real finding.")
PREPRINT_NOTE = "Preprint: not peer-reviewed."
DETERMINISTIC_VALIDATION_NOTE = "Deterministic validation: exact quote, numeric and citation checks."
SEMANTIC_ENTAILMENT_NOTE = "Semantic claim entailment: not implemented."


def live_mode_description(require_auth: bool = True) -> str:
    """Code-built description of Live Mode for the UI (no settings values)."""
    access = ("Sign-in with an allowlisted account is required and each user has a limited number of Live runs "
              "per 24 hours." if require_auth else
              "Sign-in is disabled by this deployment's operator (SCIFORGE_LIVE_REQUIRE_AUTH=false).")
    return ("Live Mode — real literature retrieval (PubMed and Crossref) and real xAI model calls. It costs money "
            "(bounded per investigation by SCIFORGE_MAX_SPEND_USD). " + access
            + " Experimental: outputs need expert review.")


# (key, UI label) in display order.
PROGRESS_STAGES: tuple[tuple[str, str], ...] = (
    ("define", "Defining question"),
    ("search", "Searching literature"),
    ("verify", "Verifying sources"),
    ("extract", "Extracting evidence"),
    ("check", "Checking evidence (deterministic checks)"),
    ("gaps", "Identifying research gaps"),
    ("hypotheses", "Generating hypotheses"),
    ("report", "Building report"),
)
_STAGE_FOR_MODEL = {"question": "define", "extraction": "extract", "gaps": "gaps", "hypotheses": "hypotheses",
                    "hypothesis_critic": "hypotheses", "hypothesis_revision": "hypotheses", "report": "report"}


# ------------------------------------------------------------------ inputs


@dataclass(frozen=True)
class LiveIdentity:
    """Signed-in identity as reported by Streamlit's ``st.user`` (built by :func:`identity_from_user`).

    Only ``is_logged_in``, ``email`` and ``email_verified`` are kept; tokens are never read. The email is
    excluded from ``repr`` and never displayed or logged.
    """

    is_logged_in: bool = False
    email: str | None = field(default=None, repr=False)
    email_verified: bool | None = None


ANONYMOUS = LiveIdentity()


@dataclass(frozen=True)
class InvestigationRequest:
    question: str
    from_year: int | None = None
    to_year: int | None = None
    max_sources: int = DEFAULT_MAX_SOURCES
    mode: str = MODE_DEMO
    # Identity from st.user (Live Mode only). The service computes the authorization decision itself.
    identity: LiveIdentity | None = None


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_request(req: InvestigationRequest) -> list[str]:
    """Human-readable input errors (empty list = valid). Never echoes the question back."""
    errors: list[str] = []
    q = req.question.strip() if isinstance(req.question, str) else ""
    if not q:
        errors.append("Please enter a research question.")
    elif len(q) < MIN_QUESTION_CHARS:
        errors.append(f"The research question is too short (minimum {MIN_QUESTION_CHARS} characters).")
    elif len(q) > MAX_QUESTION_CHARS:
        errors.append(f"The research question is too long (maximum {MAX_QUESTION_CHARS} characters).")
    for name, value in (("Start year", req.from_year), ("End year", req.to_year)):
        if value is not None and (not _is_int(value) or not MIN_YEAR <= value <= MAX_YEAR):
            errors.append(f"{name} must be a whole year between {MIN_YEAR} and {MAX_YEAR}.")
    if _is_int(req.from_year) and _is_int(req.to_year) and req.from_year > req.to_year:
        errors.append("Invalid date range: the start year is later than the end year.")
    if not _is_int(req.max_sources) or not MIN_SOURCES <= req.max_sources <= MAX_SOURCES_LIMIT:
        errors.append(f"Maximum sources must be a whole number between {MIN_SOURCES} and {MAX_SOURCES_LIMIT}.")
    if req.mode not in (MODE_DEMO, MODE_LIVE):
        errors.append("Mode must be Demo or Live.")
    return errors


# ------------------------------------------------------------------ live credentials (presence only)


def _clean(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _credential(name: str, environ: Mapping[str, str], secrets: Mapping[str, Any] | None) -> str | None:
    """Environment variable first, then the secrets mapping (e.g. ``st.secrets``)."""
    value = _clean(environ.get(name))
    if value is None and secrets is not None:
        try:
            value = _clean(secrets.get(name))
        except Exception:  # noqa: BLE001 - a broken secrets source means "not configured"
            value = None
    return value


def live_gate_enabled(environ: Mapping[str, str] | None = None,
                      secrets: Mapping[str, Any] | None = None) -> bool:
    """``SCIFORGE_LIVE_ENABLED``: True only for exactly "true" (case-insensitive, trimmed); default False.

    Environment variable first, then the secrets mapping (a non-empty env value always wins).
    """
    env = os.environ if environ is None else environ
    value = _credential(LIVE_ENABLED_NAME, env, secrets)
    return value is not None and value.lower() == "true"


def live_kill_switch(environ: Mapping[str, str] | None = None, secrets: Mapping[str, Any] | None = None) -> bool:
    """``SCIFORGE_LIVE_KILL_SWITCH``: ON unless unset, empty, "false" (case-insensitive, trimmed) or TOML false.

    Environment first (a non-empty value wins), then the secrets mapping. Unparseable -> ON (fail safe).
    """
    env = os.environ if environ is None else environ
    raw: Any = env.get(LIVE_KILL_SWITCH_NAME)
    if not (isinstance(raw, str) and raw.strip()):
        raw = None
        if secrets is not None:
            try:
                raw = secrets.get(LIVE_KILL_SWITCH_NAME)
            except Exception:  # noqa: BLE001 - a broken secrets source cannot confirm "off"
                return True
    if raw is None or raw is False:
        return False
    if isinstance(raw, str):
        value = raw.strip().lower()
        return not (value == "" or value == "false")
    return True


def live_require_auth(environ: Mapping[str, str] | None = None, secrets: Mapping[str, Any] | None = None) -> bool:
    """``SCIFORGE_LIVE_REQUIRE_AUTH``: True unless the value is exactly "false" (case-insensitive, trimmed).

    Fail closed: unset, blank, "0", "no", typos and non-string (e.g. TOML boolean) values all mean True.
    Environment variable first, then the secrets mapping.
    """
    env = os.environ if environ is None else environ
    value = _credential(LIVE_REQUIRE_AUTH_NAME, env, secrets)
    return not (value is not None and value.lower() == "false")


def live_allowed_emails(environ: Mapping[str, str] | None = None,
                        secrets: Mapping[str, Any] | None = None) -> frozenset[str]:
    """``SCIFORGE_LIVE_ALLOWED_EMAILS``: comma-separated (or a TOML list in secrets); trimmed, lower-cased.

    Environment variable first (a non-empty value wins), then the secrets mapping. Never hard-coded.
    """
    env = os.environ if environ is None else environ
    raw: Any = _clean(env.get(LIVE_ALLOWED_EMAILS_NAME))
    if raw is None and secrets is not None:
        try:
            raw = secrets.get(LIVE_ALLOWED_EMAILS_NAME)
        except Exception:  # noqa: BLE001 - a broken secrets source means "nobody allowed"
            raw = None
    items: list[Any]
    if isinstance(raw, str):
        items = raw.split(",")
    elif isinstance(raw, (list, tuple)):
        items = [p for v in raw if isinstance(v, str) for p in v.split(",")]
    else:
        items = []
    return frozenset(e.strip().lower() for e in items if isinstance(e, str) and e.strip())


def authlib_available() -> bool:
    """Whether the ``Authlib`` package (required by Streamlit's ``st.login``) is importable."""
    try:
        return importlib.util.find_spec("authlib") is not None
    except (ImportError, ValueError):
        return False


def quota_sql_driver_available() -> bool:
    """SQLAlchemy and a PostgreSQL driver (psycopg 3 or psycopg2) are importable (presence only)."""
    try:
        return importlib.util.find_spec("sqlalchemy") is not None and (
            importlib.util.find_spec("psycopg") is not None or importlib.util.find_spec("psycopg2") is not None)
    except (ImportError, ValueError):
        return False


def sql_connection_configured(section: Any) -> bool:
    """A Streamlit SQL connection section has a non-empty ``url`` or ``dialect``+``username``+``host``.

    Presence only: values are never returned, logged or displayed.
    """
    if section is None or not hasattr(section, "get"):
        return False
    try:
        def present(key: str) -> bool:
            value = section.get(key)
            return isinstance(value, str) and bool(value.strip())
        return present("url") or all(present(k) for k in SQL_CONNECTION_REQUIRED_KEYS)
    except Exception:  # noqa: BLE001 - broken secrets source = not configured
        return False


def quota_connection_name(environ: Mapping[str, str] | None = None,
                          secrets: Mapping[str, Any] | None = None) -> str:
    """Configured quota connection name (default ``sciforge_quota``); validated by parse_quota_settings."""
    env = os.environ if environ is None else environ
    value = _clean(env.get(QUOTA_CONNECTION_ENV))
    if value is None and secrets is not None:
        try:
            raw = secrets.get(QUOTA_CONNECTION_ENV)
        except Exception:  # noqa: BLE001
            raw = None
        value = raw.strip() if isinstance(raw, str) and raw.strip() else None
    return value or DEFAULT_QUOTA_CONNECTION


def _quota_connection_from_secrets(secrets: Mapping[str, Any] | None, name: str) -> bool:
    if secrets is None:
        return False
    try:
        connections = secrets.get("connections")
        section = connections.get(name) if connections is not None and hasattr(connections, "get") else None
    except Exception:  # noqa: BLE001
        return False
    return sql_connection_configured(section)


def quota_backend_problem(environ: Mapping[str, str], secrets: Mapping[str, Any] | None,
                          connection_configured: bool | None = None) -> str | None:
    """Deployment-level usage-limit backend check (no network): None when usable, else a reason code."""
    try:
        backend = parse_quota_backend(environ, secrets)
    except QuotaConfigError:
        return "invalid_backend_setting"
    if backend != BACKEND_POSTGRES:
        return None
    configured = connection_configured if connection_configured is not None else \
        _quota_connection_from_secrets(secrets, quota_connection_name(environ, secrets))
    if not configured:
        return "connection_not_configured"
    if not quota_sql_driver_available():
        return "driver_missing"
    return None


def auth_section_configured(section: Any) -> bool:
    """True when a Streamlit ``[auth]`` section has every required key (default provider) as a non-empty string.

    Presence only; values are never returned, displayed or logged.
    """
    if not isinstance(section, Mapping):
        try:
            section = dict(section) if section is not None else None
        except Exception:  # noqa: BLE001
            return False
        if not isinstance(section, Mapping):
            return False
    try:
        return all(isinstance(section.get(k), str) and section.get(k).strip() for k in AUTH_REQUIRED_KEYS)
    except Exception:  # noqa: BLE001 - a broken secrets source means "not configured"
        return False


def streamlit_auth_configured(section: Any) -> bool:
    """Streamlit OIDC auth usable: complete ``[auth]`` section AND Authlib installed (fail closed)."""
    return auth_section_configured(section) and authlib_available()


def identity_from_user(user: Any) -> LiveIdentity:
    """Build a :class:`LiveIdentity` from ``st.user`` (or any mapping). Reads ONLY is_logged_in/email/email_verified.

    Anything unexpected -> anonymous. Identity-provider tokens (``st.user.tokens``) are never accessed.
    """
    try:
        logged_in = user.get("is_logged_in") is True
        if not logged_in:
            return ANONYMOUS
        email = user.get("email")
        verified = user.get("email_verified")
    except Exception:  # noqa: BLE001 - no auth / no script context -> anonymous
        return ANONYMOUS
    return LiveIdentity(is_logged_in=True, email=email.strip() if isinstance(email, str) and email.strip() else None,
                        email_verified=verified if isinstance(verified, bool) else None)


_ACCESS_MESSAGES = {
    "auth_not_required": "Live Mode sign-in is not required on this deployment (SCIFORGE_LIVE_REQUIRE_AUTH=false).",
    "allowed": "Signed in and authorized for Live Mode.",
    "auth_not_configured": ("Live Mode requires sign-in on this deployment, but sign-in is not configured "
                            "(Streamlit [auth] secrets section and the Authlib package are required)."),
    "anonymous": "Live Mode requires sign-in. Please log in to use Live Mode.",
    "email_unverified": "Your sign-in did not provide a verified email address; Live Mode is not available.",
    "no_email": "Your sign-in did not provide an email address; Live Mode is not available.",
    "not_allowlisted": "Your account is not authorized to use Live Mode on this deployment.",
}


@dataclass(frozen=True)
class LiveAccessDecision:
    """Authorization decision for Live Mode (booleans and a reason code only; no identity data)."""

    allowed: bool
    reason: str
    require_auth: bool = True
    auth_configured: bool = False

    @property
    def message(self) -> str:
        return _ACCESS_MESSAGES.get(self.reason, "Live Mode is not available.")

    @property
    def log_value(self) -> str:
        return "allowed" if self.allowed else f"denied:{self.reason}"


def decide_live_access(identity: LiveIdentity | None, *, environ: Mapping[str, str] | None = None,
                       secrets: Mapping[str, Any] | None = None, auth_configured: bool = False) -> LiveAccessDecision:
    """Compute the Live Mode access decision (fail closed). Allowlist/config come from env/secrets only."""
    env = os.environ if environ is None else environ
    require = live_require_auth(env, secrets)
    configured = bool(auth_configured) and authlib_available()
    if not require:
        return LiveAccessDecision(True, "auth_not_required", require_auth=False, auth_configured=configured)
    if not configured:
        return LiveAccessDecision(False, "auth_not_configured", auth_configured=False)
    ident = identity if isinstance(identity, LiveIdentity) else ANONYMOUS
    if not ident.is_logged_in:
        return LiveAccessDecision(False, "anonymous", auth_configured=True)
    if ident.email_verified is False:
        return LiveAccessDecision(False, "email_unverified", auth_configured=True)
    if not ident.email:
        return LiveAccessDecision(False, "no_email", auth_configured=True)
    allowed = live_allowed_emails(env, secrets)
    if ident.email.strip().lower() in allowed:
        return LiveAccessDecision(True, "allowed", auth_configured=True)
    return LiveAccessDecision(False, "not_allowlisted", auth_configured=True)


@dataclass(frozen=True)
class LiveAvailability:
    """Whether Live Mode can be used. Holds names, booleans and reason codes only, never values."""

    available: bool
    missing: tuple[str, ...]
    gate_enabled: bool = False
    access: LiveAccessDecision | None = None
    kill_switch: bool = False
    quota_problem: str | None = None      # usage-limit backend reason code (sign-in required only)

    @property
    def deployment_ready(self) -> bool:
        """Kill switch off, gate on, both credentials present and the usage-limit backend configured."""
        return not self.kill_switch and self.gate_enabled and not self.missing and self.quota_problem is None

    @property
    def can_login(self) -> bool:
        """Offer a login button: deployment ready, sign-in required and configured, user anonymous."""
        return self.deployment_ready and self.access is not None and self.access.reason == "anonymous"

    @property
    def message(self) -> str:
        if self.kill_switch:
            return (f"Live Mode is temporarily switched off for this deployment ({LIVE_KILL_SWITCH_NAME} is on). "
                    "Demo Mode remains available.")
        if not self.gate_enabled:
            return (f"Live Mode is disabled for this deployment ({LIVE_ENABLED_NAME} is not set to \"true\").")
        if self.missing:
            return ("Live Mode is unavailable: " + " and ".join(self.missing)
                    + (" is" if len(self.missing) == 1 else " are") + " not configured.")
        if self.quota_problem == "invalid_backend_setting":
            return (f"Live Mode configuration error: {QUOTA_BACKEND_ENV} must be \"file\" or \"postgres\". "
                    "Demo Mode remains available.")
        if self.quota_problem is not None:
            return QUOTA_DB_UNAVAILABLE_MESSAGE
        if self.access is not None and not self.access.allowed:
            return self.access.message
        return "Live Mode is enabled for this deployment and credentials are configured (values are never displayed)."


def live_availability(environ: Mapping[str, str] | None = None,
                      secrets: Mapping[str, Any] | None = None, *, identity: LiveIdentity | None = None,
                      auth_configured: bool = False,
                      quota_connection_configured: bool | None = None) -> LiveAvailability:
    """Live Mode requires the deployment gate AND both credentials (presence only) AND an allowed access decision.

    With sign-in required, the usage-limit backend must also be configured (postgres: SQL connection section
    present and SQLAlchemy + driver installed; checked without connecting). ``quota_connection_configured``
    lets the UI pass the presence check of ``[connections.<name>]``; by default ``secrets`` is inspected.
    """
    env = os.environ if environ is None else environ
    kill = live_kill_switch(env, secrets)                    # checked first: overrides everything else
    gate = live_gate_enabled(env, secrets)
    missing = tuple(n for n in LIVE_CREDENTIAL_NAMES if _credential(n, env, secrets) is None)
    access = decide_live_access(identity, environ=env, secrets=secrets, auth_configured=auth_configured)
    quota_problem = quota_backend_problem(env, secrets, quota_connection_configured) if access.require_auth else None
    return LiveAvailability(available=not kill and gate and not missing and quota_problem is None and access.allowed,
                            missing=missing, gate_enabled=gate, access=access, kill_switch=kill,
                            quota_problem=quota_problem)


def _model_env(environ: Mapping[str, str], secrets: Mapping[str, Any] | None) -> dict[str, str]:
    env = dict(environ)
    for name in LIVE_CREDENTIAL_NAMES:
        value = _credential(name, environ, secrets)
        if value is not None:
            env[name] = value
    return env


def _web_model_env(environ: Mapping[str, str], secrets: Mapping[str, Any] | None) -> dict[str, str]:
    """Model-layer environment for web Live runs: the $2 web default when SCIFORGE_MAX_SPEND_USD is unset/blank."""
    env = _model_env(environ, secrets)
    if not (env.get("SCIFORGE_MAX_SPEND_USD") or "").strip():
        env["SCIFORGE_MAX_SPEND_USD"] = WEB_LIVE_DEFAULT_MAX_SPEND_USD
    return env


def _secret_candidates(environ: Mapping[str, str], secrets: Mapping[str, Any] | None) -> list[str]:
    """Values to scrub from every displayed string (never returned)."""
    out = [_credential("XAI_API_KEY", environ, secrets), _clean(environ.get("NCBI_API_KEY")),
           _clean(environ.get("SCIFORGE_CONTACT_EMAIL")), _credential(QUOTA_SALT_ENV, environ, secrets)]
    return [v for v in out if v]


# ------------------------------------------------------------------ progress


@dataclass(frozen=True)
class ProgressEvent:
    stage: str          # one of PROGRESS_STAGES keys
    state: str          # running | done | skipped | failed
    detail: str = ""


ProgressCallback = Callable[[ProgressEvent], None]


class _Progress:
    def __init__(self, callback: ProgressCallback | None) -> None:
        self.callback = callback
        self.states: dict[str, str] = {k: "pending" for k, _ in PROGRESS_STAGES}

    def emit(self, stage: str, state: str, detail: str = "") -> None:
        self.states[stage] = state
        if self.callback is not None:
            try:
                self.callback(ProgressEvent(stage, state, detail))
            except Exception:  # noqa: BLE001 - a UI problem must never break the run
                pass


class _ProgressModelClient:
    """Delegating ModelClient that reports stage transitions (request.stage) to the progress tracker."""

    def __init__(self, inner: ModelClient, progress: _Progress) -> None:
        self._inner = inner
        self._progress = progress
        self._current: str | None = None
        self.name = inner.name
        self.model = inner.model

    def complete(self, request: ModelRequest) -> ModelResponse:
        stage = _STAGE_FOR_MODEL.get(request.stage or "")
        if stage is not None and stage != self._current:
            if self._current is not None:
                self._progress.emit(self._current, "done")
            if self._current == "extract":
                self._progress.emit("check", "done")
            self._current = stage
            self._progress.emit(stage, "running")
        return self._inner.complete(request)


# ------------------------------------------------------------------ privacy guard

_PATH_RE = re.compile(r"(?<![\w.:/\-])(?:/(?:tmp|home|Users|var|private|mnt|workspace|root|opt|srv|etc|usr)"
                      r"(?:/[^\s)\]'\"`>|,;]*)?|[A-Za-z]:\\[^\s)\]'\"`>|,;]*)")


def guard_display(value: Any, *, secrets: Iterable[str] = (), withheld: Iterable[str] = (),
                  paths: Iterable[str] = ()) -> Any:
    """Recursively scrub a display value: secrets, key-like tokens, filesystem paths, raw rejected output."""
    secret_list = [s for s in secrets if s]
    needles = sorted({n for n in withheld if n}, key=len, reverse=True)
    path_list = sorted({p for p in paths if p}, key=len, reverse=True)

    def scrub(text: str) -> str:
        for needle in needles:
            if needle in text:
                text = text.replace(needle, WITHHELD)
        for p in path_list:
            text = text.replace(p, PATH_MASK)
        text = redact_text(text, secret_list)
        return _PATH_RE.sub(PATH_MASK, text)

    def walk(node: Any) -> Any:
        if isinstance(node, str):
            return scrub(node)
        if isinstance(node, dict):
            return {walk(k) if isinstance(k, str) else k: walk(v) for k, v in node.items()}
        if isinstance(node, (list, tuple)):
            return [walk(v) for v in node]
        return node

    return walk(value)


# ------------------------------------------------------------------ result


@dataclass
class WebInvestigationResult:
    ok: bool
    mode: str
    demo: bool
    status: str                                  # ok | degraded | budget_exhausted | model_auth_error | error | invalid_input | live_unavailable
    question: str = ""
    notices: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    question_definition: dict[str, Any] | None = None
    report_markdown: str = ""
    sections: dict[str, str] = field(default_factory=dict)       # section title -> Markdown body (from report.md)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    gaps: list[dict[str, Any]] = field(default_factory=list)
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    progress: dict[str, str] = field(default_factory=dict)
    evidence_graph: dict[str, Any] = field(default_factory=dict)   # v0.4 deterministic graph (evidence_graph.json)

    def displayed_text(self) -> str:
        """Every string the UI can show, concatenated (used by privacy tests)."""
        parts: list[str] = []

        def walk(node: Any) -> None:
            if isinstance(node, str):
                parts.append(node)
            elif isinstance(node, dict):
                for k, v in node.items():
                    walk(k)
                    walk(v)
            elif isinstance(node, (list, tuple)):
                for v in node:
                    walk(v)

        for name in ("question", "notices", "errors", "question_definition", "report_markdown", "sections",
                     "evidence", "conflicts", "gaps", "hypotheses", "sources", "validation", "limitations",
                     "evidence_graph"):
            walk(getattr(self, name))
        return "\n".join(parts)


def split_report_sections(report: str) -> dict[str, str]:
    """``{"A. Research Question": body, ...}`` from the code-built report.md (bodies unchanged)."""
    wanted = set(SECTION_TITLES.values())
    sections: dict[str, str] = {}
    current: str | None = None
    lines: list[str] = []
    for line in report.splitlines():
        if line.startswith("## ") and line[3:].strip() in wanted:
            if current is not None:
                sections[current] = "\n".join(lines).strip()
            current, lines = line[3:].strip(), []
        elif line.strip() == "---" and current == SECTION_TITLES["J"]:
            sections[current] = "\n".join(lines).strip()
            current, lines = None, []
        elif current is not None:
            lines.append(line)
    if current is not None:
        sections[current] = "\n".join(lines).strip()
    return sections


def render_sources(records: Sequence[Record], verification: Sequence[VerificationResult],
                   citations: Sequence[Mapping[str, Any]], access_levels: Mapping[str, str | None],
                   source_statuses: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    """Source list for the UI, rendered ONLY by the report module's deterministic citation renderer.

    ``citations`` is ``report_validation["citations"]`` (the [S#] numbering assigned by
    the report). Cited records keep their [S#] ref; unresolved ids get the
    ``[UNRESOLVED CITATION: id]`` placeholder; retrieved-but-uncited v0.2 records are
    listed afterwards with refs ``R1``, ``R2``... (same renderer).
    """
    by_id = {r.record_id: r for r in records}
    ver = {v.record_id: v for v in verification}
    statuses = dict(source_statuses or {})
    out: list[dict[str, Any]] = []
    cited: set[str] = set()
    for c in citations:
        rid = c.get("record_id")
        record = by_id.get(rid) if isinstance(rid, str) else None
        if c.get("status") != "resolved" or record is None:
            out.append({"ref": None, "record_id": rid, "cited": True, "resolved": False,
                        "citation": f"{unresolved_marker(rid)} — no v0.2 record with this id; no citation rendered.",
                        "verification_status": None, "used_as_evidence_source": False, "source_status": None,
                        "source_type": None})
            continue
        cited.add(record.record_id)
        out.append({"ref": c.get("ref"), "record_id": record.record_id, "cited": True, "resolved": True,
                    "citation": render_citation(str(c.get("ref")), record, ver.get(record.record_id),
                                                access_levels.get(record.record_id), statuses.get(record.record_id)),
                    "verification_status": ver[record.record_id].status if record.record_id in ver else None,
                    "used_as_evidence_source": True,
                    "source_status": statuses.get(record.record_id) or "unknown",
                    "source_type": source_type_badge(statuses.get(record.record_id))})
    n = 0
    for record in records:
        if record.record_id in cited:
            continue
        n += 1
        out.append({"ref": f"R{n}", "record_id": record.record_id, "cited": False, "resolved": True,
                    "citation": render_citation(f"R{n}", record, ver.get(record.record_id),
                                                access_levels.get(record.record_id), statuses.get(record.record_id)),
                    "verification_status": ver[record.record_id].status if record.record_id in ver else None,
                    "used_as_evidence_source": False,
                    "source_status": statuses.get(record.record_id) or "unknown",
                    "source_type": source_type_badge(statuses.get(record.record_id))})
    return out


def _demo_search_summary(question: str, records: list[Record], verification: list[VerificationResult],
                         from_year: int | None, to_year: int | None, max_sources: int) -> dict[str, Any]:
    """v0.2 ``summary.json``-shaped payload for the synthetic dataset (built by the v0.2 summary builder)."""
    from sciforge.logging_utils import RunLog
    from sciforge.pipeline import build_summary

    now = utc_now()
    outcomes = [SearchOutcome(database="crossref", status="ok", records=list(records), total_hits=len(records))]
    params = {"max_results_per_source": max_sources, "from_year": from_year, "to_year": to_year}
    summary = build_summary(question, params, now, now, outcomes, list(records), list(records), 0, verification,
                            RunLog())
    summary["databases_queried"] = ["crossref"]
    summary["query_generation"] = ("none: Demo Mode — no database was searched; the bundled SYNTHETIC demo "
                                   "dataset was used")
    # Synthetic records carry no bibliographic type metadata -> "unknown" (nothing is invented).
    summary["source_classification"] = {r.record_id: "unknown" for r in records}
    return summary


# ------------------------------------------------------------------ orchestration


def _status_of(res: InvestigationModelResult) -> str:
    stop = res.base.question.call.stop or res.base.evidence.stop or res.gaps.stop or res.hypotheses.stop \
        or res.narrative.stop
    if stop == "budget_exhausted":
        return "budget_exhausted"
    if stop == "auth_error":
        return "model_auth_error"
    if not res.base.evidence.accepted:
        return "degraded"
    return "ok"


def _reason_counts(items: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        codes = item.get("reason_codes") or [r.get("code") for r in item.get("reasons") or [] if isinstance(r, dict)]
        for code in codes:
            if isinstance(code, str):
                counts[code] = counts.get(code, 0) + 1
    return dict(sorted(counts.items()))


def _build_result(req: InvestigationRequest, res: InvestigationModelResult, records: list[Record],
                  verification: list[VerificationResult], search_summary: Mapping[str, Any] | None,
                  demo: bool) -> WebInvestigationResult:
    report = res.files["report"].read_text(encoding="utf-8")
    validation = res.report_validation
    access = {s.record_id: (s.access_level if s.access_level != "not_accessed" else None)
              for s in res.base.source_texts.sources}
    refs = {c["record_id"]: c["ref"] for c in validation.get("citations", []) if c.get("status") == "resolved"}
    statuses = dict((search_summary or {}).get("source_classification") or {})
    evidence = []
    for ev in res.base.evidence.accepted:
        rid = ev["source_record_id"]
        evidence.append({"evidence_id": ev["evidence_id"], "claim": ev["claim"], "quote": ev["quote"],
                         "finding": ev.get("finding"), "methods": ev.get("methods"),
                         "limitations": ev.get("limitations"), "category": ev["evidence_category"],
                         "confidence": ev["confidence"], "access": ev.get("access_level"),
                         "source_ref": refs.get(rid) or unresolved_marker(rid),
                         "source_type": source_type_badge(statuses.get(rid))})
    conflicts = [e for e in evidence if e["category"] == "conflicting"]
    ev_ref = {e["evidence_id"]: e["source_ref"] for e in evidence}
    gaps = [{"gap_id": g["gap_id"], "label": g["label"], "statement": g["gap_statement"],
             "why_unresolved": g["why_unresolved"], "supporting_evidence_ids": g["supporting_evidence_ids"],
             "conflicting_evidence_ids": g["conflicting_evidence_ids"], "confidence": g["confidence"],
             "source_refs": sorted({ev_ref[e] for e in [*g["supporting_evidence_ids"], *g["conflicting_evidence_ids"]]
                                    if e in ev_ref})}
            for g in res.gaps.accepted]
    hypotheses = [{"hypothesis_id": h["hypothesis_id"], "label": h["label"], "status": HYPOTHESIS_DISCLAIMER,
                   "notice": h["notice"], "hypothesis": h["hypothesis"],
                   "mechanistic_claim_level": h["mechanistic_claim_level"],
                   "supported_claim_level": h["supported_claim_level"],
                   "causality_statement": h["causality_statement"], "rationale": h["rationale"],
                   "prediction": h["prediction"], "alternative_explanation": dict(h["alternative_explanation"]),
                   "falsification_test": dict(h["falsification_test"]), "assumptions": list(h["assumptions"]),
                   "evidence_limitations": list(h["evidence_limitations"]),
                   "source_quality_summary": h["source_quality_summary"], "research_gap_id": h["research_gap_id"],
                   "evidence_ids": list(h["evidence_ids"]), "confidence": h["confidence"],
                   "confidence_detail": dict(h["confidence_detail"]),
                   "critic_status": h["stress_test"]["critic_status"],
                   "revision_status": h["stress_test"]["revision_status"],
                   "stress_test": dict(h["stress_test"]),
                   "source_refs": sorted({ev_ref[e] for e in h["evidence_ids"] if e in ev_ref})}
                  for h in res.hypotheses.accepted]
    sources = render_sources(records, verification, validation.get("citations", []), access, statuses)
    ev_rej = res.base.evidence.rejected
    stages = {name: {"status": r.status, "skip_reason": r.skip_reason, "accepted": len(r.accepted),
                     "rejected": len(r.rejected), "rejection_reasons": _reason_counts(r.rejected)}
              for name, r in (("research_gaps", res.gaps), ("hypotheses", res.hypotheses),
                              ("report_narrative", res.narrative))}
    stages["hypotheses"].update({"critic_status": getattr(res.hypotheses, "critic_status", "not_run"),
                                 "revision_status": getattr(res.hypotheses, "revision_status", "not_run"),
                                 "revised": getattr(res.hypotheses, "revised", 0)})
    budget = res.budget
    used = budget.get("used", {})
    validation_view = {
        "report_status": validation.get("status"),
        "sections_present": validation.get("sections_present", []),
        "citations": {"resolved": validation.get("counts", {}).get("citations_resolved", 0),
                      "unresolved": validation.get("counts", {}).get("citations_unresolved", 0)},
        "issues": [{k: v for k, v in i.items() if k in ("code", "record_id", "detail")}
                   for i in validation.get("issues", [])],
        "question_definition": "ok" if res.base.question.definition is not None else "failed (raw question used)",
        "source_texts": res.base.source_texts.counts(),
        "evidence": {"accepted": len(res.base.evidence.accepted), "rejected": len(ev_rej),
                     "rejection_reasons": _reason_counts(ev_rej)},
        "stages": stages,
        "budget": {"attempts_used": used.get("attempts"), "attempts_limit": budget.get("limits", {}).get("max_attempts"),
                   "sources_used": used.get("sources"), "sources_limit": budget.get("limits", {}).get("max_sources"),
                   "input_tokens_used": used.get("input_tokens"), "output_tokens_used": used.get("output_tokens"),
                   "output_tokens_reported": used.get("output_tokens_reported"),
                   "reasoning_tokens_reported": used.get("reasoning_tokens"),
                   "output_side_tokens_charged": used.get("output_side_tokens_charged"),
                   "reported_cost_usd": used.get("reported_cost_usd"),
                   "reported_cost": "reported" if used.get("reported_cost_attempts") else "unavailable",
                   "estimated_spend_usd": used.get("spend_usd"),
                   "spend_cap_usd": budget.get("limits", {}).get("max_spend_usd"),
                   "financial_guard": "SCIFORGE_MAX_SPEND_USD (output-token settings are not a guaranteed ceiling)",
                   "exhausted_by": budget.get("exhausted_by")},
        "claim_checks": validation.get("claim_checks") or {"semantic_claim_check": {"status": SEMANTIC_STATUS,
                                                                                    "ran": False}},
        "source_types": {"statuses": statuses, "note": (search_summary or {}).get("source_policy", {}).get(
            "status_note", "source_status is derived from bibliographic metadata only")},
        "model": "scripted demo model (FakeModelClient; no API call)" if demo else "xAI Responses API (XAIClient)",
        "raw_rejected_model_output": "never displayed; rejected items are shown as reason codes only",
        "evidence_graph": graph_summary(res.evidence_graph) if res.evidence_graph else {"status": "not_built"},
    }
    limitations = [
        "Evidence comes from abstracts only (no full text).",
        "Claim checks are deterministic only (exact quotes, numbers/units, ids and citations); semantic "
        "(model-based) claim checking is not implemented. SciForge does not judge study quality.",
        "Source types (journal article, preprint, ...) come from bibliographic metadata only; 'peer-reviewed "
        "journal article' is a metadata label, not a guarantee of peer review.",
        "Citation verification confirms that identifiers resolve and match; it cannot confirm what a paper claims.",
        "Research gaps are the pipeline's inference and hypotheses are untested proposals, not findings.",
    ]
    if demo:
        limitations.insert(0, "DEMO MODE: every source, abstract, finding, gap and hypothesis is SYNTHETIC and "
                              "illustrates the pipeline only. Nothing here is a real finding.")
    else:
        limitations.insert(0, "Live Mode has not yet been validated against the real xAI API; treat output with care.")
    definition = res.base.question.definition.model_dump() if res.base.question.definition else None
    notices = []
    status = _status_of(res)
    if status == "degraded":
        notices.append("No evidence passed validation; the report contains no findings (degraded result).")
    elif status == "budget_exhausted":
        notices.append("The model budget was exhausted; later sections were built by code only.")
    elif status == "model_auth_error":
        notices.append("The model provider rejected the credentials; later sections were built by code only.")
    policy = (search_summary or {}).get("source_policy") or {}
    if policy.get("fewer_than_requested") and policy.get("shortfall_note"):
        notices.append(str(policy["shortfall_note"]))
    preprints = sum(1 for s_ in sources if s_.get("cited") and s_.get("source_status") == "preprint")
    if preprints:
        notices.append(f"{preprints} cited source(s) are preprints; they are labelled \"{PREPRINT_BADGE}\".")
    return WebInvestigationResult(
        ok=True, mode=req.mode, demo=demo, status=status, question=req.question.strip(), notices=notices,
        question_definition=definition, report_markdown=report, sections=split_report_sections(report),
        evidence=evidence, conflicts=conflicts, gaps=gaps, hypotheses=hypotheses, sources=sources,
        validation=validation_view, limitations=limitations, evidence_graph=dict(res.evidence_graph or {}))


def _rejected_needles(res: InvestigationModelResult) -> list[str]:
    raw = [*res.base.evidence.rejected_raw, *res.gaps.rejected_raw, *res.hypotheses.rejected_raw,
           *res.narrative.rejected_raw]
    return rejected_text_values(raw)


def run_web_investigation(
    request: InvestigationRequest,
    *,
    progress: ProgressCallback | None = None,
    environ: Mapping[str, str] | None = None,
    secrets: Mapping[str, Any] | None = None,
    live_http_client: httpx.Client | None = None,
    live_model_client_factory: Callable[[ModelSettings], ModelClient] | None = None,
    sleep: Callable[[float], None] | None = None,
    auth_configured: bool | None = None,
    quota_clock: Callable[[], float] | None = None,
    quota_backend: QuotaBackend | None = None,
    quota_connection_factory: Callable[[str], Any] | None = None,
    quota_connection_configured: bool | None = None,
) -> WebInvestigationResult:
    """Run one investigation for the web UI. Never raises; problems come back in ``errors``.

    ``live_http_client`` / ``live_model_client_factory`` exist for offline tests
    (MockTransport / FakeModelClient); the app never passes them.
    ``auth_configured`` (Streamlit OIDC ready) defaults to checking ``secrets["auth"]``; Authlib
    availability is always re-checked here. Demo Mode never needs sign-in and never touches the quota backend.
    ``quota_connection_factory(name)`` returns a SQLAlchemy engine for the postgres usage-limit backend (the
    app passes ``st.connection(name, type="sql").engine``); ``quota_backend`` overrides the backend (tests).
    """
    env = os.environ if environ is None else environ
    tracker = _Progress(progress)
    demo = request.mode == MODE_DEMO
    secret_values = _secret_candidates(env, secrets)
    errors = validate_request(request)
    if errors:
        return WebInvestigationResult(ok=False, mode=request.mode, demo=demo, status="invalid_input", errors=errors,
                                      progress=dict(tracker.states))
    access: LiveAccessDecision | None = None
    quota: QuotaDecision | None = None
    if not demo:
        # Defence in depth: the service refuses live runs itself (kill switch, deployment gate, credentials,
        # sign-in, usage limit), not only the UI — before any literature lookup, temp dir or model-client creation.
        if live_kill_switch(env, secrets):
            return WebInvestigationResult(ok=False, mode=request.mode, demo=False, status="live_unavailable",
                                          errors=[LiveAvailability(False, (), kill_switch=True).message],
                                          progress=dict(tracker.states))
        if auth_configured is None:
            auth_configured = _auth_configured_from_secrets(secrets)
        availability = live_availability(env, secrets, identity=request.identity, auth_configured=auth_configured,
                                         quota_connection_configured=quota_connection_configured)
        access = availability.access
        if not availability.deployment_ready:
            return WebInvestigationResult(ok=False, mode=request.mode, demo=False, status="live_unavailable",
                                          errors=[availability.message], progress=dict(tracker.states))
        get_logger().info("live access decision: %s", access.log_value if access else "denied:missing")
        if access is None or not access.allowed:
            status = "live_unavailable" if access is None or access.reason == "auth_not_configured" \
                else "live_unauthorized"
            return WebInvestigationResult(ok=False, mode=request.mode, demo=False, status=status,
                                          errors=[availability.message], progress=dict(tracker.states))
        refusal = _preflight_live(request, env, secrets, secret_values, tracker, access, quota_clock,
                                  quota_backend, quota_connection_factory)
        if isinstance(refusal, WebInvestigationResult):
            return refusal
        quota = refusal
    tmp = Path(tempfile.mkdtemp(prefix=TEMP_PREFIX))
    # Only specific (multi-component) paths are masked literally; a bare "/" must never be replaced.
    paths = [p for p in {str(tmp), str(tmp.resolve()), str(Path.home()), str(Path.cwd())} if len(Path(p).parts) >= 3]
    try:
        try:
            if demo:
                result, needles, leaks = _run_demo(request, tracker, tmp, secret_values)
            else:
                result, needles, leaks = _run_live(request, tracker, tmp, env, secrets, secret_values,
                                                   live_http_client, live_model_client_factory, sleep,
                                                   access=access, quota=quota,
                                                   quota_connection_configured=quota_connection_configured)
        except (ModelConfigError, ConfigError) as exc:
            # Config messages name variables only (never values); still scrubbed below.
            return guard_display_result(WebInvestigationResult(
                ok=False, mode=request.mode, demo=demo, status="error",
                errors=[f"Live Mode configuration error: {exc}"], progress=dict(tracker.states)),
                secret_values, [], paths)
        except Exception as exc:  # noqa: BLE001 - never show tracebacks / paths in the UI
            for key, state in tracker.states.items():
                if state == "running":
                    tracker.emit(key, "failed")
            return WebInvestigationResult(ok=False, mode=request.mode, demo=demo, status="error",
                                          errors=[f"The investigation failed unexpectedly ({type(exc).__name__})."],
                                          progress=dict(tracker.states))
        _finish_progress(tracker, result)
        result.progress = dict(tracker.states)
        result.validation["privacy_guard"] = {
            "run_artifacts": "written to a temporary directory outside the repository and deleted after the run",
            "rejected_output_leaks_in_run_files": len(leaks),
            "display_scrubbed": "secrets, key-like tokens, filesystem paths and raw rejected output",
        }
        return guard_display_result(result, secret_values, needles, paths)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _preflight_live(request: InvestigationRequest, env: Mapping[str, str], secrets: Mapping[str, Any] | None,
                    secret_values: list[str], tracker: _Progress, access: LiveAccessDecision,
                    clock: Callable[[], float] | None = None, backend: QuotaBackend | None = None,
                    connection_factory: Callable[[str], Any] | None = None,
                    ) -> WebInvestigationResult | QuotaDecision | None:
    """Local checks for an authorised live request (no network): configuration, then the usage limit.

    Returns a refusal result, the recorded :class:`QuotaDecision` (sign-in required) or None (sign-in disabled:
    no quota, previous behaviour). Configuration is validated first so a misconfiguration never uses up a run.
    """
    def refuse(status: str, message: str) -> WebInvestigationResult:
        return WebInvestigationResult(ok=False, mode=request.mode, demo=False, status=status,
                                      errors=[guard_display(message, secrets=secret_values)],
                                      progress=dict(tracker.states))

    try:
        model_env = _web_model_env(env, secrets)
        Settings.from_env(model_env)
        ModelSettings.from_env(model_env)
        quota_settings = parse_quota_settings(env, secrets) if access.require_auth else None
    except (ModelConfigError, ConfigError, QuotaConfigError) as exc:
        return refuse("error", f"Live Mode configuration error: {exc}")
    if quota_settings is None:
        return None
    ident = request.identity if isinstance(request.identity, LiveIdentity) else ANONYMOUS
    if not (ident.is_logged_in and ident.email and ident.email_verified is True):
        get_logger().info("live quota decision: denied:no_verified_email")
        return refuse("live_unauthorized", "Live Mode requires a verified email address for the per-user usage "
                                           "limit; your sign-in did not provide one.")
    if backend is None and quota_settings.backend == BACKEND_POSTGRES and connection_factory is not None:
        name = quota_settings.connection
        backend = SqlQuotaBackend(lambda: connection_factory(name))
    try:
        decision = check_and_record(ident.email, quota_settings, now=clock, backend=backend)
    except (QuotaStoreError, ValueError) as exc:
        get_logger().warning("live quota decision: denied:store_error (%s)", type(exc).__name__)
        return refuse("live_unavailable", "Live Mode is unavailable: the usage-limit store could not be read or "
                                          "written (fail closed). Demo Mode remains available.")
    except Exception as exc:  # noqa: BLE001 - any other backend failure also fails closed
        get_logger().warning("live quota decision: denied:store_error (%s)", type(exc).__name__)
        return refuse("live_unavailable", "Live Mode is unavailable: the usage-limit store could not be read or "
                                          "written (fail closed). Demo Mode remains available.")
    get_logger().info("live quota decision: %s", decision.log_value)
    if not decision.allowed:
        return refuse("live_quota_exceeded", decision.message)
    return decision


def _auth_configured_from_secrets(secrets: Mapping[str, Any] | None) -> bool:
    if secrets is None:
        return False
    try:
        return streamlit_auth_configured(secrets.get("auth"))
    except Exception:  # noqa: BLE001
        return False


def _finish_progress(tracker: _Progress, result: WebInvestigationResult) -> None:
    """Close every stage with its final state and a short, code-built detail line."""
    v = result.validation
    ev = v.get("evidence", {})
    stages = v.get("stages", {})
    details = {
        "define": f"question definition {v.get('question_definition', 'n/a')}",
        "extract": f"evidence items proposed: {ev.get('accepted', 0) + ev.get('rejected', 0)}",
        "check": (f"{ev.get('accepted', 0)} accepted, {ev.get('rejected', 0)} rejected by deterministic checks "
                  f"(exact quote, numbers, ids); semantic claim check: {SEMANTIC_STATUS}"),
        "gaps": f"research gaps accepted: {stages.get('research_gaps', {}).get('accepted', 0)}",
        "hypotheses": f"hypotheses accepted: {stages.get('hypotheses', {}).get('accepted', 0)}",
        "report": f"report validation: {v.get('report_status', 'n/a')}",
    }
    for key, _label in PROGRESS_STAGES:
        state = tracker.states[key]
        if key in details or state in ("pending", "running"):
            final = "skipped" if state == "pending" else ("done" if state == "running" else state)
            tracker.emit(key, final, details.get(key, ""))


def guard_display_result(result: WebInvestigationResult, secrets: Sequence[str], needles: Sequence[str],
                         paths: Sequence[str]) -> WebInvestigationResult:
    for name in ("question", "notices", "errors", "question_definition", "report_markdown", "sections", "evidence",
                 "conflicts", "gaps", "hypotheses", "sources", "validation", "limitations"):
        setattr(result, name, guard_display(getattr(result, name), secrets=secrets, withheld=needles, paths=paths))
    return result


def _pipeline_kwargs(tmp: Path, sleep: Callable[[float], None]) -> dict[str, Any]:
    from sciforge.http_utils import RateLimiter

    return {"run_dir": tmp / "model", "sleep": sleep, "debug_keep_rejected_raw": False,
            "pubmed_limiter": RateLimiter(0, sleep=sleep), "crossref_limiter": RateLimiter(0, sleep=sleep)}


def _run_demo(req: InvestigationRequest, tracker: _Progress, tmp: Path,
              secret_values: list[str]) -> tuple[WebInvestigationResult, list[str], list]:
    from sciforge.demo_data import DEMO_LABEL, demo_http_client, demo_model_client, demo_records
    from sciforge.llm.budget import BudgetLimits, BudgetTracker

    question = req.question.strip()
    tracker.emit("search", "running", "Loading the bundled synthetic demo dataset (no database is searched)")
    records, verification = demo_records(req.from_year, req.to_year)
    tracker.emit("search", "done", f"{len(records)} synthetic records")
    tracker.emit("verify", "running", "Demo verification statuses are synthetic")
    summary = _demo_search_summary(question, records, verification, req.from_year, req.to_year, req.max_sources)
    tracker.emit("verify", "done")
    budget = BudgetTracker(BudgetLimits(max_sources=req.max_sources, max_spend_usd=None))
    client = _ProgressModelClient(demo_model_client(), tracker)
    no_sleep: Callable[[float], None] = lambda _s: None  # noqa: E731
    http = demo_http_client()
    try:
        res = run_model_investigation(question, records, verification, model_client=client,
                                      settings=Settings(), tracker=budget, search_summary=summary,
                                      http_client=http, max_source_chars=4000, include_partially_verified=False,
                                      **_pipeline_kwargs(tmp, no_sleep))
    finally:
        http.close()
    needles = _rejected_needles(res)
    leaks = find_leaks(res.run_dir, needles)
    result = _build_result(req, res, records, verification, summary, demo=True)
    result.notices.insert(0, f"{DEMO_LABEL}. Demo Mode always analyses the same bundled synthetic example "
                             "dataset; the entered question is echoed but not searched.")
    return result, needles, leaks


def _run_live(req: InvestigationRequest, tracker: _Progress, tmp: Path, env: Mapping[str, str],
              secrets: Mapping[str, Any] | None, secret_values: list[str], http_client: httpx.Client | None,
              factory: Callable[[ModelSettings], ModelClient] | None,
              sleep: Callable[[float], None] | None, *,
              access: LiveAccessDecision | None = None,
              quota: QuotaDecision | None = None,
              quota_connection_configured: bool | None = None) -> tuple[WebInvestigationResult, list[str], list]:
    from sciforge.pipeline import run_investigation

    availability = live_availability(env, secrets, quota_connection_configured=quota_connection_configured)
    if not availability.deployment_ready:              # includes the kill switch
        raise ModelConfigError(availability.message)
    if access is None or not access.allowed:          # re-checked here too: no lookup / client before authorization
        raise ModelConfigError("Live Mode is not authorized for this request.")
    if access.require_auth and (quota is None or not quota.allowed):
        raise ModelConfigError("Live Mode usage limit was not checked for this request.")
    model_env = _web_model_env(env, secrets)
    settings = Settings.from_env(model_env)
    model_settings = ModelSettings.from_env(model_env)          # budgets, prices, spend cap (fail closed)
    model_settings = replace(model_settings, max_sources=min(model_settings.max_sources, req.max_sources))
    secret_values.extend(v for v in settings.secret_values() + model_settings.secret_values() if v)
    question = req.question.strip()

    tracker.emit("search", "running", "Expanded queries on PubMed and Crossref (candidate pool per query), then "
                                      "deduplication, bounded abstract enrichment and deterministic ranking")
    # Candidate pool per query (SCIFORGE_CANDIDATE_POOL_PER_QUERY) -> dedup -> bounded abstract enrichment ->
    # deterministic scoring -> source policy -> verification with backfill -> at most max_sources verified records.
    v02 = run_investigation(question, max_results=req.max_sources, from_year=req.from_year, to_year=req.to_year,
                            output_dir=tmp / "v02", settings=settings, client=http_client,
                            max_selected=model_settings.max_sources,
                            accept_partially_verified=model_settings.include_partially_verified,
                            **({"sleep": sleep} if sleep is not None else {}))
    enr = v02.summary.get("abstract_enrichment") or {}
    queries = len((v02.summary.get("query_expansion") or {}).get("queries") or []) or 1
    tracker.emit("search", "done", f"{queries} quer{'y' if queries == 1 else 'ies'} per database; "
                                   f"{v02.summary.get('total_retrieved', 0)} candidate records retrieved "
                                   f"({v02.summary.get('unique_records', 0)} after deduplication); abstracts for "
                                   f"{enr.get('with_abstract', 0)} of {enr.get('considered', 0)} candidates "
                                   f"considered for ranking")
    selected = len((v02.summary.get("selection") or {}).get("selected_record_ids") or [])
    policy = v02.summary.get("source_policy") or {}
    detail = (f"DOI/PMID verification: {selected} of {model_settings.max_sources} requested records selected and "
              f"verified (source policy {policy.get('policy', 'n/a')}; preprints selected: "
              f"{policy.get('preprints_selected', 0)})")
    if policy.get("fewer_than_requested"):
        detail += " — fewer than requested"
    tracker.emit("verify", "done", detail)
    if factory is not None:
        inner = factory(model_settings)
    else:
        from sciforge.llm.xai import XAIClient

        inner = XAIClient(model_settings)
    client = _ProgressModelClient(inner, tracker)
    try:
        kwargs = _pipeline_kwargs(tmp, sleep or time.sleep)
        if http_client is None:
            kwargs.pop("pubmed_limiter")
            kwargs.pop("crossref_limiter")
        res = run_model_investigation(question, v02.records, v02.verification, model_client=client,
                                      settings=settings, model_settings=model_settings, search_summary=v02.summary,
                                      http_client=http_client, **kwargs)
    finally:
        close = getattr(inner, "close", None)
        if callable(close) and factory is None:
            close()
    needles = _rejected_needles(res)
    leaks = find_leaks(res.run_dir, needles)
    result = _build_result(req, res, v02.records, v02.verification, v02.summary, demo=False)
    result.notices.insert(0, "Live Mode is experimental: real literature and real xAI calls (costs money); "
                             "outputs need expert review.")
    return result, needles, leaks
