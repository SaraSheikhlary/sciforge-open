"""Timestamps, secret redaction, and the per-run request/error log."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from sciforge.models import ErrorEntry, RequestLogEntry

REDACTED = "[REDACTED]"

# Parameter names whose values are never written anywhere. The contact email is
# not a credential, but it is personal data and is not needed to rerun a search.
SENSITIVE_PARAM_NAMES = frozenset(
    {"api_key", "apikey", "key", "token", "access_token", "password", "secret", "email", "mailto"}
)
_SENSITIVE_IN_TEXT_RE = re.compile(
    r"(?i)\b(api_key|apikey|access_token|token|password|secret|email|mailto)=([^&\s\"']+)"
)
_MIN_SECRET_LENGTH = 4


def utc_now() -> datetime:
    """Current time as an aware UTC datetime."""
    return datetime.now(timezone.utc)


def iso_utc(dt: datetime | None = None) -> str:
    """ISO 8601 UTC timestamp with a ``Z`` suffix, e.g. ``2026-09-28T23:41:00Z``."""
    dt = (dt or utc_now()).astimezone(timezone.utc)
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z")


def run_stamp(dt: datetime | None = None) -> str:
    """Compact UTC stamp for run directory names, e.g. ``20260928T234100Z``."""
    dt = (dt or utc_now()).astimezone(timezone.utc)
    return dt.strftime("%Y%m%dT%H%M%SZ")


def is_sensitive_param(name: str) -> bool:
    """True if a query-parameter name holds a secret or personal data."""
    return name.lower() in SENSITIVE_PARAM_NAMES


def redact_params(params: Mapping[str, Any] | None) -> dict[str, Any]:
    """Copy of ``params`` with sensitive values replaced by ``[REDACTED]``."""
    if not params:
        return {}
    return {k: (REDACTED if is_sensitive_param(str(k)) else v) for k, v in params.items()}


def redact_url(url: str) -> str:
    """Redact sensitive query parameters in a URL, keeping everything else."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return redact_text(url)
    if not parts.query:
        return url
    pairs = parse_qsl(parts.query, keep_blank_values=True)
    redacted = [(k, REDACTED if is_sensitive_param(k) else v) for k, v in pairs]
    query = urlencode(redacted, safe="[]:,/")
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))


def redact_text(text: str, secrets: Iterable[str | None] = ()) -> str:
    """Remove secrets from free text (error messages, log lines).

    Replaces ``name=value`` pairs for sensitive names and any literal
    occurrence of the given secret values.
    """
    result = _SENSITIVE_IN_TEXT_RE.sub(lambda m: f"{m.group(1)}={REDACTED}", text)
    for secret in secrets:
        if secret and len(secret) >= _MIN_SECRET_LENGTH:
            result = result.replace(secret, REDACTED)
    return result


class RedactingFilter(logging.Filter):
    """Logging filter that scrubs secrets from every record's message."""

    def __init__(self, secrets: Iterable[str | None] = ()) -> None:
        super().__init__()
        self._secrets = [s for s in secrets if s]

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003 - logging API
        record.msg = redact_text(record.getMessage(), self._secrets)
        record.args = None
        return True


def get_logger() -> logging.Logger:
    """The package logger (``sciforge``)."""
    return logging.getLogger("sciforge")


class RunLog:
    """Collects request log entries and error entries for one run."""

    def __init__(self, secrets: Iterable[str | None] = ()) -> None:
        self.entries: list[RequestLogEntry] = []
        self.errors: list[ErrorEntry] = []
        self._secrets = [s for s in secrets if s]

    def scrub(self, text: str) -> str:
        """Redact secrets from ``text``."""
        return redact_text(text, self._secrets)

    def add_entry(self, entry: RequestLogEntry) -> RequestLogEntry:
        """Append a request entry (already redacted by the caller)."""
        self.entries.append(entry)
        return entry

    def add_error(self, error: ErrorEntry) -> ErrorEntry:
        """Append an error entry, scrubbing its message and URL."""
        error.message = self.scrub(error.message)
        if error.url:
            error.url = self.scrub(redact_url(error.url))
        self.errors.append(error)
        get_logger().warning("%s %s error (%s): %s", error.database, error.stage, error.error_type, error.message)
        return error

    def mark_failed(self, entry: RequestLogEntry, error: ErrorEntry) -> None:
        """Mark a logged request as failed (e.g. its body was malformed) and record the error."""
        entry.status = "error"
        entry.error_type = error.error_type
        entry.message = self.scrub(error.message)
        self.add_error(error)
