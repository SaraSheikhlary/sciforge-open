"""SQL (PostgreSQL) backend for the Live Mode usage limit.

Schema: ``deploy/sql/001_live_quota.sql`` (table ``sciforge_live_quota_events``). The app never creates or
alters tables: apply the migration first; a missing table is a database error and Live Mode is refused.

Atomic check-and-record, all inside ONE transaction (``engine.begin()``):

1. serialise per identity — PostgreSQL: ``SELECT pg_advisory_xact_lock(hashtext(:identity_hash))`` (released
   at commit/rollback; a hash collision only adds serialisation, never a wrong count). SQLite (offline tests
   only): the first statement is a write, which takes SQLite's database write lock;
2. delete this identity's events at or before ``now - window`` (keeps the table small);
3. count events after ``now - window`` and find the oldest (events dated after ``now`` — e.g. from a replica
   whose clock runs slightly ahead, or a concurrent request stamped a moment later — are counted too, so
   clock skew can only make the limit stricter, never looser);
4. insert one event at ``now`` only if the count is below the limit; 5. commit.

Time source: the application clock (``now`` is passed in, stored as UTC ``timestamptz``); the database
clock is not used, so replicas need synchronised clocks (NTP). The column default ``now()`` is unused.

Only the identity hash and timestamps are written. Any problem (no SQLAlchemy/driver, connection or
authentication failure, missing table, unsupported dialect, SQL error) raises :class:`QuotaStoreError` with
the exception type name only (never messages, URLs or credentials); callers refuse the run (fail closed).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

from sciforge.live_quota import BACKEND_POSTGRES, QuotaDecision, QuotaStoreError

__all__ = ["COUNT_SQL", "DELETE_SQL", "INSERT_SQL", "LOCK_SQL", "SUPPORTED_DIALECTS", "SqlQuotaBackend",
           "TABLE"]

TABLE = "sciforge_live_quota_events"
LOCK_SQL = "SELECT pg_advisory_xact_lock(hashtext(:identity_hash))"
DELETE_SQL = f"DELETE FROM {TABLE} WHERE identity_hash = :identity_hash AND created_at <= :cutoff"
COUNT_SQL = (f"SELECT COUNT(*), MIN(created_at) FROM {TABLE} "
             "WHERE identity_hash = :identity_hash AND created_at > :cutoff")
INSERT_SQL = f"INSERT INTO {TABLE} (identity_hash, created_at) VALUES (:identity_hash, :created_at)"
SUPPORTED_DIALECTS = ("postgresql", "sqlite")


def _sqlalchemy_text() -> Callable[[str], Any]:
    try:
        from sqlalchemy import text
    except ImportError:
        raise QuotaStoreError("SQLAlchemy is not installed") from None
    return text


def _as_utc(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if not isinstance(value, datetime):
        raise QuotaStoreError("quota database returned an unexpected timestamp")
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


class SqlQuotaBackend:
    """:class:`~sciforge.live_quota.QuotaBackend` over a SQLAlchemy-style engine.

    ``engine_provider`` returns an object with ``dialect.name`` and ``begin()`` (a context manager yielding a
    connection with ``execute(statement, params)``) — in the app ``st.connection(name, type="sql").engine``.
    ``text`` wraps SQL strings (default ``sqlalchemy.text``; tests may pass ``str``).
    """

    name = BACKEND_POSTGRES

    def __init__(self, engine_provider: Callable[[], Any], *, text: Callable[[str], Any] | None = None) -> None:
        self._engine_provider = engine_provider
        self._text = text

    def check_and_record(self, identity_hash: str, now: float, limit: int, window_s: int) -> QuotaDecision:
        if not isinstance(identity_hash, str) or not identity_hash.startswith(("h1:", "s1:")):
            raise QuotaStoreError("invalid identity hash")
        try:
            engine = self._engine_provider()
        except QuotaStoreError:
            raise
        except Exception as exc:  # noqa: BLE001 - missing config/driver, invalid URL, ...
            raise QuotaStoreError(f"quota database unavailable ({type(exc).__name__})") from None
        dialect = str(getattr(getattr(engine, "dialect", None), "name", "") or "")
        if dialect not in SUPPORTED_DIALECTS:
            raise QuotaStoreError("unsupported quota database dialect")
        text = self._text or _sqlalchemy_text()
        now_dt = datetime.fromtimestamp(float(now), timezone.utc)
        cutoff = now_dt - timedelta(seconds=window_s)
        try:
            with engine.begin() as conn:
                if dialect == "postgresql":
                    conn.execute(text(LOCK_SQL), {"identity_hash": identity_hash})
                conn.execute(text(DELETE_SQL), {"identity_hash": identity_hash, "cutoff": cutoff})
                row = conn.execute(text(COUNT_SQL), {"identity_hash": identity_hash, "cutoff": cutoff}).fetchone()
                count = int(row[0]) if row is not None and row[0] is not None else 0
                oldest = _as_utc(row[1]) if row is not None else None
                if count >= limit:
                    retry = (oldest + timedelta(seconds=window_s) - now_dt).total_seconds() if oldest else window_s
                    return QuotaDecision(False, count, limit, retry_after_s=max(0.0, retry))
                conn.execute(text(INSERT_SQL), {"identity_hash": identity_hash, "created_at": now_dt})
                return QuotaDecision(True, count + 1, limit)
        except QuotaStoreError:
            raise
        except Exception as exc:  # noqa: BLE001 - connection/auth failure, missing table, SQL error: fail closed
            raise QuotaStoreError(f"quota database error ({type(exc).__name__})") from None
