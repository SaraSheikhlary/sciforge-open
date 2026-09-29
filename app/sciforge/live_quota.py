"""Server-side Live Mode usage limit: at most N Live runs per signed-in user per rolling 24 h (fail closed).

Used only when Live Mode sign-in is required (``SCIFORGE_LIVE_REQUIRE_AUTH`` not ``false``). With sign-in
disabled the previous behaviour is unchanged (no quota; there is no verified identity to count against).

Settings (environment first, then the optional secrets mapping):

* ``SCIFORGE_LIVE_MAX_RUNS_PER_USER`` — integer 1-1000, default 3. Invalid values are a configuration error.
* ``SCIFORGE_LIVE_QUOTA_PATH`` — absolute path of the JSON store (``~`` is expanded). Default:
  ``$XDG_STATE_HOME/sciforge/live_quota.json`` or ``~/.local/state/sciforge/live_quota.json`` — outside the
  repository and outside run directories. A relative path is a configuration error.
* ``SCIFORGE_LIVE_QUOTA_BACKEND`` — ``file`` (default) or ``postgres``; anything else is a configuration error.
  ``postgres`` uses :class:`sciforge.live_quota_sql.SqlQuotaBackend` over a Streamlit SQL connection.
* ``SCIFORGE_LIVE_QUOTA_CONNECTION`` — name of that connection (default ``sciforge_quota`` =
  ``[connections.sciforge_quota]`` in the Streamlit secrets; credentials live only there).
* ``SCIFORGE_LIVE_QUOTA_SALT`` — optional server-side secret. With a salt the user key is
  ``HMAC-SHA256(salt, email)``; without one it is ``SHA-256("sciforge-live-quota-v1:" + email)``.
  Tradeoff: an unsalted SHA-256 of an email can be reversed by guessing candidate emails (e.g. the
  allowlist), so anyone who can read the store could tell which allowlisted user ran how often; with a
  secret salt that requires the salt as well. Changing or adding the salt starts every user's count afresh.

Identity: ONLY the verified email (``email_verified is True``), trimmed and lower-cased. A missing or
unverified email is refused (fail closed). The plaintext email is never written to the store, logged or
returned; the store holds ``{key_hash: [unix timestamps]}`` only.

When a run counts: at the START of an authorised Live run — after the kill switch, gate, credentials,
sign-in/allowlist and configuration checks, before any literature lookup, temp directory or model client.
A run that later fails (network error, budget stop, model error) therefore still counts; this is deliberate
(each started run may already have spent money). A refused run (quota exhausted) is not recorded.

Backends implement :class:`QuotaBackend` — one atomic ``check_and_record(identity_hash, now, limit,
window_s)`` that counts the identity's runs after ``now - window_s`` and records a new run only when the
count is below ``limit``. Only the identity hash and timestamps are ever passed to a backend.

File backend concurrency: an exclusive ``fcntl.flock`` on a sidecar lock file serialises read-modify-write across
Streamlit sessions/processes on one host; writes go to a temp file in the same directory, are fsynced and
atomically ``os.replace``-d. The file is created with mode 0600 (directory 0700). Any read/parse/lock/write
problem raises :class:`QuotaStoreError` and the caller refuses the run (fail closed). The store is per host:
multiple replicas without a shared filesystem would each keep their own counts.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import tempfile
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

try:  # POSIX only; without fcntl the store cannot be locked -> fail closed
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None  # type: ignore[assignment]

__all__ = [
    "BACKEND_FILE", "BACKEND_POSTGRES", "DEFAULT_MAX_RUNS_PER_USER", "DEFAULT_QUOTA_CONNECTION",
    "FileQuotaBackend", "MAX_RUNS_ENV", "QUOTA_BACKEND_ENV", "QUOTA_CONNECTION_ENV", "QUOTA_PATH_ENV",
    "QUOTA_SALT_ENV", "QuotaBackend", "QuotaConfigError", "QuotaDecision", "QuotaSettings", "QuotaStoreError",
    "WINDOW_SECONDS", "check_and_record", "default_quota_path", "parse_quota_settings", "quota_key",
]

MAX_RUNS_ENV = "SCIFORGE_LIVE_MAX_RUNS_PER_USER"
QUOTA_PATH_ENV = "SCIFORGE_LIVE_QUOTA_PATH"
QUOTA_SALT_ENV = "SCIFORGE_LIVE_QUOTA_SALT"
QUOTA_BACKEND_ENV = "SCIFORGE_LIVE_QUOTA_BACKEND"
QUOTA_CONNECTION_ENV = "SCIFORGE_LIVE_QUOTA_CONNECTION"
BACKEND_FILE = "file"
BACKEND_POSTGRES = "postgres"
QUOTA_BACKENDS = (BACKEND_FILE, BACKEND_POSTGRES)
DEFAULT_QUOTA_CONNECTION = "sciforge_quota"
_CONNECTION_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
DEFAULT_MAX_RUNS_PER_USER = 3
MIN_MAX_RUNS_PER_USER = 1
MAX_MAX_RUNS_PER_USER = 1000
WINDOW_SECONDS = 24 * 60 * 60
STORE_VERSION = 1
_UNSALTED_PREFIX = "sciforge-live-quota-v1:"


class QuotaConfigError(ValueError):
    """Invalid quota setting (message names the variable, never a value)."""


class QuotaStoreError(RuntimeError):
    """The quota store could not be locked, read, parsed or written (callers fail closed)."""


def default_quota_path(environ: Mapping[str, str] | None = None) -> Path:
    env = os.environ if environ is None else environ
    base = (env.get("XDG_STATE_HOME") or "").strip()
    root = Path(base).expanduser() if base and Path(base).expanduser().is_absolute() \
        else Path.home() / ".local" / "state"
    return root / "sciforge" / "live_quota.json"


@dataclass(frozen=True)
class QuotaSettings:
    max_runs: int = DEFAULT_MAX_RUNS_PER_USER
    path: Path = field(default_factory=default_quota_path)
    salt: str | None = field(default=None, repr=False)
    backend: str = BACKEND_FILE
    connection: str = DEFAULT_QUOTA_CONNECTION

    @property
    def key_scheme(self) -> str:
        return "hmac-sha256" if self.salt else "sha256"


def _setting(name: str, environ: Mapping[str, str], secrets: Mapping[str, Any] | None) -> str | None:
    value = environ.get(name)
    if isinstance(value, str) and value.strip():
        return value.strip()
    if secrets is not None:
        try:
            raw = secrets.get(name)
        except Exception:  # noqa: BLE001 - broken secrets source = not set
            raw = None
        if isinstance(raw, bool):
            return str(raw)
        if isinstance(raw, int):
            return str(raw)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return None


def parse_quota_settings(environ: Mapping[str, str] | None = None,
                         secrets: Mapping[str, Any] | None = None) -> QuotaSettings:
    """Parse and validate the quota settings (raises :class:`QuotaConfigError`)."""
    env = os.environ if environ is None else environ
    raw = _setting(MAX_RUNS_ENV, env, secrets)
    if raw is None:
        max_runs = DEFAULT_MAX_RUNS_PER_USER
    else:
        try:
            max_runs = int(raw, 10)
        except ValueError:
            raise QuotaConfigError(f"{MAX_RUNS_ENV} must be an integer between {MIN_MAX_RUNS_PER_USER} and "
                                   f"{MAX_MAX_RUNS_PER_USER}") from None
        if not MIN_MAX_RUNS_PER_USER <= max_runs <= MAX_MAX_RUNS_PER_USER:
            raise QuotaConfigError(f"{MAX_RUNS_ENV} must be an integer between {MIN_MAX_RUNS_PER_USER} and "
                                   f"{MAX_MAX_RUNS_PER_USER}")
    raw_path = _setting(QUOTA_PATH_ENV, env, secrets)
    if raw_path is None:
        path = default_quota_path(env)
    else:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            raise QuotaConfigError(f"{QUOTA_PATH_ENV} must be an absolute path")
    salt = _setting(QUOTA_SALT_ENV, env, secrets)
    backend = parse_quota_backend(env, secrets)
    connection = _setting(QUOTA_CONNECTION_ENV, env, secrets) or DEFAULT_QUOTA_CONNECTION
    if not _CONNECTION_NAME.match(connection):
        raise QuotaConfigError(f"{QUOTA_CONNECTION_ENV} must be 1-64 letters, digits, '_' or '-'")
    return QuotaSettings(max_runs=max_runs, path=path, salt=salt, backend=backend, connection=connection)


def parse_quota_backend(environ: Mapping[str, str] | None = None,
                        secrets: Mapping[str, Any] | None = None) -> str:
    """``file`` (default) or ``postgres`` (case-insensitive); anything else raises :class:`QuotaConfigError`."""
    env = os.environ if environ is None else environ
    raw = _setting(QUOTA_BACKEND_ENV, env, secrets)
    if raw is None:
        return BACKEND_FILE
    value = raw.strip().lower()
    if value not in QUOTA_BACKENDS:
        raise QuotaConfigError(f"{QUOTA_BACKEND_ENV} must be \"file\" or \"postgres\"")
    return value


def normalize_email(email: str | None) -> str | None:
    if not isinstance(email, str):
        return None
    value = email.strip().lower()
    return value or None


def quota_key(email: str, salt: str | None) -> str:
    """Keyed hash of the normalised email (the only user identifier ever stored)."""
    norm = normalize_email(email)
    if norm is None:
        raise ValueError("email required")
    if salt:
        return "h1:" + hmac.new(salt.encode("utf-8"), norm.encode("utf-8"), hashlib.sha256).hexdigest()
    return "s1:" + hashlib.sha256((_UNSALTED_PREFIX + norm).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class QuotaDecision:
    allowed: bool
    used: int                      # runs in the window, including this one when allowed
    limit: int
    retry_after_s: float | None = None

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    @property
    def log_value(self) -> str:
        return f"{'allowed' if self.allowed else 'denied'}:{self.used}/{self.limit}"

    @property
    def message(self) -> str:
        if self.allowed:
            return f"Live run {self.used} of {self.limit} in the last 24 hours."
        hours = max(1, math.ceil((self.retry_after_s or 0) / 3600))
        return (f"Live Mode usage limit reached: {self.limit} Live runs per user per rolling 24 hours. "
                f"Try again in about {hours} hour{'s' if hours != 1 else ''}. Demo Mode remains available.")


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": STORE_VERSION, "window_seconds": WINDOW_SECONDS, "users": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise QuotaStoreError(f"quota store unreadable ({type(exc).__name__})") from None
    users = data.get("users") if isinstance(data, dict) else None
    if not isinstance(users, dict) or data.get("version") != STORE_VERSION:
        raise QuotaStoreError("quota store has an unexpected format")
    for key, stamps in users.items():
        if not isinstance(key, str) or not isinstance(stamps, list) or \
                not all(isinstance(t, (int, float)) and not isinstance(t, bool) for t in stamps):
            raise QuotaStoreError("quota store has an unexpected format")
    return data


def _atomic_write(path: Path, data: dict[str, Any]) -> None:
    fd, tmp = tempfile.mkstemp(prefix=".live_quota.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, sort_keys=True, separators=(",", ":"))
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


@runtime_checkable
class QuotaBackend(Protocol):
    """Atomic per-identity run counter (rolling window). Receives only the identity hash and timestamps."""

    name: str

    def check_and_record(self, identity_hash: str, now: float, limit: int, window_s: int) -> QuotaDecision:
        """Count runs after ``now - window_s``; record one at ``now`` only if the count is below ``limit``.

        Must be atomic per identity and raise :class:`QuotaStoreError` on any storage problem.
        """
        ...


class FileQuotaBackend:
    """Local JSON store with an ``fcntl`` lock and atomic replace (single host; local dev and tests)."""

    name = BACKEND_FILE

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def check_and_record(self, identity_hash: str, now: float, limit: int, window_s: int) -> QuotaDecision:
        if fcntl is None:
            raise QuotaStoreError("file locking is unavailable on this platform")
        t = float(now)
        path = self.path
        try:
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            lock_fd = os.open(str(path) + ".lock", os.O_RDWR | os.O_CREAT, 0o600)
        except OSError as exc:
            raise QuotaStoreError(f"quota store not writable ({type(exc).__name__})") from None
        try:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
            except OSError as exc:
                raise QuotaStoreError(f"quota store lock failed ({type(exc).__name__})") from None
            data = _load(path)
            cutoff = t - window_s
            users: dict[str, list[float]] = {}
            for k, stamps in data["users"].items():
                kept = sorted(float(x) for x in stamps if cutoff < float(x) <= t + 300)
                if kept:
                    users[k] = kept
            mine = users.get(identity_hash, [])
            if len(mine) >= limit:
                return QuotaDecision(False, len(mine), limit,
                                     retry_after_s=max(0.0, mine[len(mine) - limit] + window_s - t))
            users[identity_hash] = [*mine, t]
            try:
                _atomic_write(path, {"version": STORE_VERSION, "window_seconds": WINDOW_SECONDS, "users": users})
            except OSError as exc:
                raise QuotaStoreError(f"quota store not writable ({type(exc).__name__})") from None
            return QuotaDecision(True, len(users[identity_hash]), limit)
        finally:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)


def check_and_record(email: str | None, settings: QuotaSettings, *,
                     now: Callable[[], float] | None = None,
                     backend: QuotaBackend | None = None) -> QuotaDecision:
    """Atomically count one Live run for ``email`` if it is under the limit (rolling 24 h).

    Only the keyed hash of the normalised email reaches the backend. Without ``backend`` the file backend
    at ``settings.path`` is used (a postgres setting then needs an explicit backend: refused otherwise).
    Raises :class:`QuotaStoreError` on any store problem and ``ValueError`` without an email.
    """
    key = quota_key(email or "", settings.salt) if normalize_email(email) else None
    if key is None:
        raise ValueError("a verified email is required for the Live Mode usage limit")
    if backend is None:
        if settings.backend != BACKEND_FILE:
            raise QuotaStoreError("the configured quota backend is not available")
        backend = FileQuotaBackend(settings.path)
    t = float((now or time.time)())
    decision = backend.check_and_record(key, t, settings.max_runs, WINDOW_SECONDS)
    if not isinstance(decision, QuotaDecision):
        raise QuotaStoreError("quota backend returned an invalid decision")
    return decision
