"""Runtime configuration, read only from environment variables.

Secrets (``NCBI_API_KEY``) are never read from files in the repository and are
excluded from ``repr`` so they cannot leak through logging of the settings
object.
"""

from __future__ import annotations

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
    """

    ncbi_api_key: str | None = field(default=None, repr=False)
    contact_email: str | None = field(default=None, repr=False)
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS

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
