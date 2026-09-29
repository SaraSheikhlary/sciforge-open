"""Pluggable Live usage-limit backends: file (unchanged) and postgres (SQL), tested offline.

SQLAlchemy / psycopg are not needed: the SQL backend is exercised against stdlib ``sqlite3`` through a tiny
SQLAlchemy-shaped engine adapter (``dialect.name``, ``begin()``, ``execute(sql, params)``) and against a fake
PostgreSQL engine that records statements and emulates ``pg_advisory_xact_lock``. No real database is used.
"""

from __future__ import annotations

import re
import sqlite3
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pytest

from sciforge import app_service as svc
from sciforge import live_quota as lq
from sciforge import live_quota_sql as lqs
from sciforge.demo_data import DEMO_QUESTION
from test_live_quota_killswitch import AUTH, DAY, EMAIL, EMAIL2, USER1, USER2, Calls, Clock, env_for, live

REPO = Path(__file__).resolve().parents[1]
T0 = 1_800_000_000.0
H1 = "h1:" + "a" * 64
H2 = "h1:" + "b" * 64
SQLITE_SCHEMA = ("CREATE TABLE IF NOT EXISTS sciforge_live_quota_events (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                 "identity_hash TEXT NOT NULL, created_at TEXT NOT NULL)")


@pytest.fixture
def qpath(tmp_path) -> Path:
    return tmp_path / "state" / "live_quota.json"


@pytest.fixture(autouse=True)
def _authlib(monkeypatch):
    monkeypatch.setattr(svc, "authlib_available", lambda: True)


# ------------------------------------------------------------------ engines


def _param(value):
    if isinstance(value, datetime):     # fixed-width UTC text keeps lexical order == time order
        return value.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
    return value


class _SqliteConn:
    def __init__(self, raw: sqlite3.Connection) -> None:
        self.raw = raw

    def execute(self, sql, params=None):
        return self.raw.execute(sql, {k: _param(v) for k, v in (params or {}).items()})


class SqliteEngine:
    """SQLAlchemy-shaped engine over stdlib sqlite3 (deferred BEGIN: the backend's first write serialises)."""

    class dialect:
        name = "sqlite"

    def __init__(self, path: Path, *, schema: bool = True) -> None:
        self.path = str(path)
        if schema:
            with sqlite3.connect(self.path) as c:
                c.execute(SQLITE_SCHEMA)

    @contextmanager
    def begin(self):
        raw = sqlite3.connect(self.path, timeout=30, isolation_level=None, check_same_thread=False)
        try:
            raw.execute("BEGIN")
            try:
                yield _SqliteConn(raw)
            except BaseException:
                raw.execute("ROLLBACK")
                raise
            raw.execute("COMMIT")
        finally:
            raw.close()

    def rows(self):
        with sqlite3.connect(self.path) as c:
            return c.execute("SELECT * FROM sciforge_live_quota_events ORDER BY id").fetchall()


class _Result:
    def __init__(self, row=None) -> None:
        self.row = row

    def fetchone(self):
        return self.row


class FakePostgresEngine:
    """Records statements; emulates pg_advisory_xact_lock (per-key lock held until the transaction ends)."""

    class dialect:
        name = "postgresql"

    def __init__(self, *, fail_on: str | None = None, delay: float = 0.0) -> None:
        self.events: list[tuple[str, datetime]] = []
        self.log: list[tuple[int, str]] = []
        self.fail_on = fail_on
        self.delay = delay
        self.begins = 0
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    @contextmanager
    def begin(self):
        with self._guard:
            self.begins += 1
            tx = self.begins
        held: list[threading.Lock] = []
        engine = self

        class Conn:
            def execute(self, sql, params):
                engine.log.append((tx, sql))
                if engine.fail_on and engine.fail_on in sql:
                    raise RuntimeError("relation does not exist; url=postgresql://u:hunter2secret@db/x")
                h = params["identity_hash"]
                if sql == lqs.LOCK_SQL:
                    with engine._guard:
                        lock = engine._locks.setdefault(h, threading.Lock())
                    lock.acquire()
                    held.append(lock)
                    return _Result((None,))
                if sql == lqs.DELETE_SQL:
                    engine.events = [e for e in engine.events if not (e[0] == h and e[1] <= params["cutoff"])]
                    return _Result()
                if sql == lqs.COUNT_SQL:
                    mine = [e[1] for e in engine.events if e[0] == h and params["cutoff"] < e[1]]
                    if engine.delay:
                        threading.Event().wait(engine.delay)      # widen the race window
                    return _Result((len(mine), min(mine) if mine else None))
                if sql == lqs.INSERT_SQL:
                    engine.events.append((h, params["created_at"]))
                    return _Result()
                raise AssertionError(sql)

        try:
            yield Conn()
        finally:
            for lock in held:
                lock.release()


def sql_backend(engine):
    return lqs.SqlQuotaBackend(lambda: engine, text=str)


# ------------------------------------------------------------------ settings


def test_backend_setting_default_valid_and_invalid():
    assert lq.parse_quota_backend({}) == "file" and lq.parse_quota_backend({}, {}) == "file"
    assert lq.parse_quota_backend({"SCIFORGE_LIVE_QUOTA_BACKEND": " Postgres "}) == "postgres"
    assert lq.parse_quota_backend({}, {"SCIFORGE_LIVE_QUOTA_BACKEND": "postgres"}) == "postgres"
    assert lq.parse_quota_backend({"SCIFORGE_LIVE_QUOTA_BACKEND": "file"},
                                  {"SCIFORGE_LIVE_QUOTA_BACKEND": "postgres"}) == "file"      # env wins
    for bad in ("mysql", "sqlite", "redis", "true"):
        with pytest.raises(lq.QuotaConfigError, match="SCIFORGE_LIVE_QUOTA_BACKEND"):
            lq.parse_quota_backend({"SCIFORGE_LIVE_QUOTA_BACKEND": bad})
    s = lq.parse_quota_settings({"SCIFORGE_LIVE_QUOTA_PATH": "/tmp/q.json"})
    assert s.backend == "file" and s.connection == "sciforge_quota"
    s = lq.parse_quota_settings({"SCIFORGE_LIVE_QUOTA_PATH": "/tmp/q.json", "SCIFORGE_LIVE_QUOTA_BACKEND": "postgres",
                                 "SCIFORGE_LIVE_QUOTA_CONNECTION": "quota_db"})
    assert s.backend == "postgres" and s.connection == "quota_db"
    with pytest.raises(lq.QuotaConfigError, match="SCIFORGE_LIVE_QUOTA_CONNECTION"):
        lq.parse_quota_settings({"SCIFORGE_LIVE_QUOTA_CONNECTION": "bad name;drop"})


def test_backends_implement_the_protocol(tmp_path):
    assert isinstance(lq.FileQuotaBackend(tmp_path / "q.json"), lq.QuotaBackend)
    assert isinstance(sql_backend(FakePostgresEngine()), lq.QuotaBackend)


def test_file_backend_unchanged_through_the_wrapper(tmp_path):
    s = lq.QuotaSettings(max_runs=2, path=tmp_path / "q.json")
    assert [lq.check_and_record(EMAIL, s, now=lambda: T0 + i).allowed for i in range(3)] == [True, True, False]
    assert EMAIL.lower() not in (tmp_path / "q.json").read_text()
    # postgres configured but no backend supplied -> refused, never silently the file store
    with pytest.raises(lq.QuotaStoreError):
        lq.check_and_record(EMAIL, lq.QuotaSettings(path=tmp_path / "p.json", backend="postgres"), now=lambda: T0)
    assert not (tmp_path / "p.json").exists()


# ------------------------------------------------------------------ SQL logic against SQLite


def test_sqlite_atomic_increment_limit_and_retry(tmp_path):
    eng = SqliteEngine(tmp_path / "q.db")
    b = sql_backend(eng)
    got = [b.check_and_record(H1, T0 + i, 3, DAY) for i in range(4)]
    assert [d.allowed for d in got] == [True, True, True, False]
    assert [d.used for d in got] == [1, 2, 3, 3]
    assert got[3].retry_after_s == pytest.approx(DAY - 3)
    assert len(eng.rows()) == 3                        # the refused attempt was not recorded


def test_sqlite_rolling_window_and_separate_users(tmp_path):
    eng = SqliteEngine(tmp_path / "q.db")
    b = sql_backend(eng)
    for i in range(3):
        assert b.check_and_record(H1, T0 + i * 3600, 3, DAY).allowed
    assert not b.check_and_record(H1, T0 + DAY - 1, 3, DAY).allowed
    assert b.check_and_record(H2, T0 + DAY - 1, 3, DAY).allowed            # other identity unaffected
    d = b.check_and_record(H1, T0 + DAY + 1, 3, DAY)                         # oldest run left the window
    assert d.allowed and d.used == 3
    assert not b.check_and_record(H1, T0 + DAY + 2, 3, DAY).allowed
    hashes = [r[1] for r in eng.rows()]
    assert hashes.count(H1) == 3 and hashes.count(H2) == 1                   # expired row deleted


def test_sqlite_concurrent_requests_exactly_limit_succeed(tmp_path):
    eng = SqliteEngine(tmp_path / "q.db")
    limit, extra = 3, 9
    barrier = threading.Barrier(limit + extra)
    results: list[bool] = []
    errors: list[BaseException] = []

    def worker(i: int) -> None:
        try:
            barrier.wait()
            results.append(sql_backend(eng).check_and_record(H1, T0 + i / 1000, limit, DAY).allowed)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(limit + extra)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [] and results.count(True) == limit and results.count(False) == extra
    assert len(eng.rows()) == limit


def test_sqlite_missing_table_fails_closed(tmp_path):
    b = sql_backend(SqliteEngine(tmp_path / "empty.db", schema=False))
    with pytest.raises(lq.QuotaStoreError, match="quota database error"):
        b.check_and_record(H1, T0, 3, DAY)


# ------------------------------------------------------------------ PostgreSQL statements (fake engine)


def test_postgres_statements_in_one_transaction_lock_first_insert_only_below_limit():
    eng = FakePostgresEngine()
    b = sql_backend(eng)
    assert b.check_and_record(H1, T0, 1, DAY).allowed
    first = [sql for tx, sql in eng.log if tx == 1]
    assert first == [lqs.LOCK_SQL, lqs.DELETE_SQL, lqs.COUNT_SQL, lqs.INSERT_SQL]
    assert "pg_advisory_xact_lock(hashtext(:identity_hash))" in lqs.LOCK_SQL
    d = b.check_and_record(H1, T0 + 5, 1, DAY)
    assert not d.allowed and d.retry_after_s == pytest.approx(DAY - 5)
    second = [sql for tx, sql in eng.log if tx == 2]
    assert second == [lqs.LOCK_SQL, lqs.DELETE_SQL, lqs.COUNT_SQL]           # no INSERT at the limit
    assert eng.begins == 2 and len(eng.events) == 1
    for sql in (lqs.DELETE_SQL, lqs.COUNT_SQL, lqs.INSERT_SQL):
        assert "sciforge_live_quota_events" in sql and ":identity_hash" in sql   # bound parameters only


def test_postgres_advisory_lock_serialises_concurrent_requests():
    eng = FakePostgresEngine(delay=0.005)
    limit, extra = 4, 8
    barrier = threading.Barrier(limit + extra)
    results: list[bool] = []

    def worker(i):
        barrier.wait()
        results.append(sql_backend(eng).check_and_record(H1, T0 + i / 1000, limit, DAY).allowed)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(limit + extra)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count(True) == limit and len(eng.events) == limit


def test_postgres_errors_fail_closed_without_details():
    for fail in ("pg_advisory_xact_lock", "DELETE", "SELECT COUNT", "INSERT"):
        with pytest.raises(lq.QuotaStoreError) as info:
            sql_backend(FakePostgresEngine(fail_on=fail)).check_and_record(H1, T0, 3, DAY)
        assert "hunter2secret" not in str(info.value) and "postgresql://" not in str(info.value)
        assert info.value.__cause__ is None and info.value.__suppress_context__

    def broken_provider():
        raise RuntimeError("could not connect: password=hunter2secret")
    with pytest.raises(lq.QuotaStoreError, match="unavailable") as info:
        lqs.SqlQuotaBackend(broken_provider, text=str).check_and_record(H1, T0, 3, DAY)
    assert "hunter2secret" not in str(info.value)

    class Mysql:
        class dialect:
            name = "mysql"
    with pytest.raises(lq.QuotaStoreError, match="dialect"):
        sql_backend(Mysql()).check_and_record(H1, T0, 3, DAY)
    with pytest.raises(lq.QuotaStoreError, match="identity hash"):
        sql_backend(FakePostgresEngine()).check_and_record(EMAIL, T0, 3, DAY)   # never a plaintext email


def test_without_sqlalchemy_the_backend_fails_closed():
    pytest.importorskip("sqlite3")
    try:
        import sqlalchemy  # noqa: F401
    except ImportError:
        with pytest.raises(lq.QuotaStoreError, match="SQLAlchemy"):
            lqs.SqlQuotaBackend(lambda: FakePostgresEngine()).check_and_record(H1, T0, 3, DAY)
    else:  # pragma: no cover - SQLAlchemy is not installed in this environment
        pytest.skip("SQLAlchemy installed")


def test_migration_file_is_idempotent_and_secret_free():
    sql = (REPO / "deploy" / "sql" / "001_live_quota.sql").read_text(encoding="utf-8")
    assert "CREATE TABLE IF NOT EXISTS sciforge_live_quota_events" in sql
    assert "CREATE INDEX IF NOT EXISTS" in sql and "(identity_hash, created_at)" in sql
    assert "TIMESTAMPTZ" in sql and "identity_hash TEXT" in sql
    assert not re.search(r"(?i)email\s+text|password\s*=|postgres(ql)?(\+\w+)?://", sql)


# ------------------------------------------------------------------ service integration


def pg_env(qpath, **extra):
    return env_for(qpath, SCIFORGE_LIVE_QUOTA_BACKEND="postgres", **extra)


CONN = {"connections": {"sciforge_quota": {"url": "postgresql+psycopg://<DB_USER>:<DB_PASSWORD>@<DB_HOST>/<DB>"}}}


def live_pg(identity, env, calls, backend=None, factory=None, clock=None, configured=True):
    return svc.run_web_investigation(
        svc.InvestigationRequest(question=DEMO_QUESTION, mode=svc.MODE_LIVE, identity=identity, max_sources=2),
        environ=env, secrets={"auth": AUTH}, live_http_client=calls.http_client(),
        live_model_client_factory=calls.factory_fn, quota_clock=clock, quota_backend=backend,
        quota_connection_factory=factory, quota_connection_configured=configured,
        sleep=lambda s: None)


@pytest.fixture
def driver(monkeypatch):
    monkeypatch.setattr(svc, "quota_sql_driver_available", lambda: True)


def test_availability_postgres_missing_connection_or_driver(qpath, monkeypatch):
    env = pg_env(qpath)
    a = svc.live_availability(env, {"auth": AUTH}, identity=USER1, auth_configured=True)
    assert not a.available and a.quota_problem == "connection_not_configured"
    assert a.message == svc.QUOTA_DB_UNAVAILABLE_MESSAGE and not a.can_login
    monkeypatch.setattr(svc, "quota_sql_driver_available", lambda: False)
    a = svc.live_availability(env, {"auth": AUTH, **CONN}, identity=USER1, auth_configured=True)
    assert not a.available and a.quota_problem == "driver_missing"
    assert "<DB_PASSWORD>" not in a.message and "postgresql" not in a.message
    monkeypatch.setattr(svc, "quota_sql_driver_available", lambda: True)
    assert svc.live_availability(env, {"auth": AUTH, **CONN}, identity=USER1, auth_configured=True).available
    incomplete = {"connections": {"sciforge_quota": {"dialect": "postgresql", "host": "<DB_HOST>"}}}
    assert not svc.live_availability(env, {"auth": AUTH, **incomplete}, identity=USER1,
                                     auth_configured=True).available
    other = {"connections": {"other_db": CONN["connections"]["sciforge_quota"]}}
    assert not svc.live_availability(env, {"auth": AUTH, **other}, identity=USER1, auth_configured=True).available
    assert svc.live_availability({**env, "SCIFORGE_LIVE_QUOTA_CONNECTION": "other_db"}, {"auth": AUTH, **other},
                                 identity=USER1, auth_configured=True).available


def test_availability_invalid_backend_setting(qpath):
    a = svc.live_availability(env_for(qpath, SCIFORGE_LIVE_QUOTA_BACKEND="mongo"), {"auth": AUTH},
                              identity=USER1, auth_configured=True)
    assert not a.available and "SCIFORGE_LIVE_QUOTA_BACKEND" in a.message and "configuration error" in a.message


def test_missing_db_config_refuses_before_any_activity(qpath, monkeypatch):
    monkeypatch.setattr(tempfile, "mkdtemp", lambda *a, **k: pytest.fail("no temp dir"))
    calls = Calls()
    r = live_pg(USER1, pg_env(qpath), calls, factory=lambda n: pytest.fail("no connection"), configured=False)
    assert r.status == "live_unavailable" and r.errors == [svc.QUOTA_DB_UNAVAILABLE_MESSAGE]
    assert calls.http == [] and calls.factory == []


def test_postgres_backend_quota_runs_and_refusal_before_lookup(qpath, tmp_path, driver, monkeypatch):
    eng = SqliteEngine(tmp_path / "q.db")
    names: list[str] = []

    def factory(name):
        names.append(name)
        return eng

    monkeypatch.setattr(lqs, "_sqlalchemy_text", lambda: str)   # SQLAlchemy is not installed locally
    clock = Clock(T0)
    env = pg_env(qpath, SCIFORGE_LIVE_QUOTA_SALT="pepper-salt-value")
    calls = Calls()
    for i in range(3):
        clock.t = T0 + i
        r = live_pg(USER1, env, calls, factory=factory, clock=clock)
        assert r.status not in ("live_quota_exceeded", "live_unavailable", "error"), (r.status, r.errors)
    assert len(calls.factory) == 3 and names == ["sciforge_quota"] * 3
    before_http, before_factory = len(calls.http), len(calls.factory)
    real_mkdtemp = tempfile.mkdtemp
    monkeypatch.setattr(tempfile, "mkdtemp", lambda *a, **k: pytest.fail("no temp dir"))
    r = live_pg(USER1, env, calls, factory=factory, clock=clock)
    assert r.status == "live_quota_exceeded" and "3 Live runs per user" in r.errors[0]
    assert len(calls.http) == before_http and len(calls.factory) == before_factory
    monkeypatch.setattr(tempfile, "mkdtemp", real_mkdtemp)
    assert live_pg(USER2, env, Calls(), factory=factory, clock=clock).status not in (
        "live_quota_exceeded", "live_unavailable")                               # separate user
    # rows hold only keyed hashes and timestamps: no email, salt or key
    rows = eng.rows()
    dump = repr(rows)
    assert len(rows) == 4 and all(re.fullmatch(r"h1:[0-9a-f]{64}", r[1]) for r in rows)
    for needle in (EMAIL, EMAIL.lower(), EMAIL2, "pepper-salt-value", "xai-"):
        assert needle not in dump
    assert not (qpath.exists())                                                    # file store not used


def test_database_failure_fails_closed_before_activity(qpath, driver, monkeypatch):
    monkeypatch.setattr(tempfile, "mkdtemp", lambda *a, **k: pytest.fail("no temp dir"))
    for backend, factory in ((sql_backend(FakePostgresEngine(fail_on="DELETE")), None),
                             (None, lambda name: (_ for _ in ()).throw(RuntimeError("password=hunter2secret"))),
                             (None, None)):                                        # no connection factory at all
        calls = Calls()
        r = live_pg(USER1, pg_env(qpath), calls, backend=backend, factory=factory)
        assert r.status == "live_unavailable" and "usage-limit store" in r.errors[0]
        assert "hunter2secret" not in " ".join(r.errors)
        assert calls.http == [] and calls.factory == []


def test_demo_never_touches_the_quota_backend_or_database(qpath, monkeypatch):
    touched: list[str] = []

    class Spy:
        name = "spy"

        def check_and_record(self, *a):
            touched.append("backend")
            raise AssertionError("demo must not use the quota backend")

    monkeypatch.setattr(svc, "quota_sql_driver_available", lambda: touched.append("driver") or False)
    r = svc.run_web_investigation(svc.InvestigationRequest(question=DEMO_QUESTION), environ=pg_env(qpath),
                                  secrets={}, quota_backend=Spy(),
                                  quota_connection_factory=lambda n: touched.append("connection"))
    assert r.ok and r.demo and touched == []


def test_auth_disabled_ignores_the_postgres_backend(qpath, monkeypatch):
    monkeypatch.setattr(svc, "quota_sql_driver_available", lambda: False)
    env = pg_env(qpath, SCIFORGE_LIVE_REQUIRE_AUTH="false")
    assert svc.live_availability(env, {}).available                 # unchanged: no quota without sign-in
    calls = Calls()
    r = live_pg(svc.ANONYMOUS, env, calls, factory=lambda n: pytest.fail("no DB"), configured=False)
    assert r.status not in ("live_unavailable", "live_quota_exceeded", "live_unauthorized") and calls.factory


def test_sql_connection_configured_presence_only():
    assert svc.sql_connection_configured({"url": "postgresql+psycopg://<U>:<P>@<H>/<D>"})
    assert svc.sql_connection_configured({"dialect": "postgresql", "username": "<U>", "host": "<H>"})
    for bad in (None, {}, {"url": "  "}, {"dialect": "postgresql", "host": "<H>"}, "url", 42):
        assert not svc.sql_connection_configured(bad)
    assert "SCIFORGE_LIVE_QUOTA_BACKEND" in svc.LIVE_SETTING_NAMES
    assert "SCIFORGE_LIVE_QUOTA_CONNECTION" in svc.LIVE_SETTING_NAMES
    assert "connections" not in svc.LIVE_SETTING_NAMES          # DB credentials never enter the service
