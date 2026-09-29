"""Live Mode usage limit (per verified user, rolling 24 h), kill switch and web spend default (offline only)."""

from __future__ import annotations

import json
import logging
import tempfile
import threading
from pathlib import Path

import httpx
import pytest

from conftest import FAKE_XAI_KEY, FAKE_XAI_MODEL, Sleeper, api_handler, mock_client
from m2_support import efetch_xml, question_output, xml_response
from sciforge import app_service as svc
from sciforge import live_quota as lq
from sciforge.demo_data import DEMO_QUESTION
from sciforge.llm.fake import FakeModelClient

EMAIL = "Quota.User@Example.org"
EMAIL2 = "second.user@example.org"
AUTH = {"redirect_uri": "https://example.invalid/oauth2callback", "cookie_secret": "placeholder-cookie",
        "client_id": "placeholder-id", "client_secret": "placeholder-secret",
        "server_metadata_url": "https://example.invalid/.well-known/openid-configuration"}
USER1 = svc.LiveIdentity(True, EMAIL, True)
USER2 = svc.LiveIdentity(True, EMAIL2, True)
DAY = 24 * 3600


class Clock:
    def __init__(self, t: float = 1_800_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture(autouse=True)
def _authlib(monkeypatch):
    monkeypatch.setattr(svc, "authlib_available", lambda: True)


@pytest.fixture
def qpath(tmp_path) -> Path:
    return tmp_path / "state" / "live_quota.json"


def env_for(qpath: Path, **extra: str) -> dict[str, str]:
    env = {"SCIFORGE_LIVE_ENABLED": "true", "XAI_API_KEY": FAKE_XAI_KEY, "XAI_MODEL": FAKE_XAI_MODEL,
           "SCIFORGE_MAX_SPEND_USD": "none", "SCIFORGE_LIVE_ALLOWED_EMAILS": f"{EMAIL}, {EMAIL2}",
           "SCIFORGE_LIVE_QUOTA_PATH": str(qpath)}
    env.update(extra)
    return env


def _live_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("efetch.fcgi"):
        return xml_response(efetch_xml([("111", "<AbstractText>Shear exposure raised platelet activation.</AbstractText>")]))
    return api_handler(request)


class Calls:
    def __init__(self) -> None:
        self.http: list = []
        self.factory: list = []

    def http_client(self):
        def handler(request):
            self.http.append(request.url.path)
            return _live_handler(request)
        return mock_client(handler)

    def factory_fn(self, ms):
        self.factory.append(ms.max_spend_usd)
        return FakeModelClient([lambda r: question_output() if r.stage == "question" else {"items": []}] * 20)


def live(identity, env, calls: Calls, clock=None, secrets=None, monkeypatch=None):
    return svc.run_web_investigation(
        svc.InvestigationRequest(question=DEMO_QUESTION, mode=svc.MODE_LIVE, identity=identity, max_sources=2),
        environ=env, secrets=secrets if secrets is not None else {"auth": AUTH},
        live_http_client=calls.http_client(), live_model_client_factory=calls.factory_fn, sleep=Sleeper(),
        quota_clock=clock)


# ------------------------------------------------------------------ settings


def test_quota_settings_defaults_and_validation(monkeypatch, tmp_path):
    s = lq.parse_quota_settings({"SCIFORGE_LIVE_QUOTA_PATH": str(tmp_path / "q.json")}, {})
    assert s.max_runs == 3 and s.salt is None and s.key_scheme == "sha256"
    for ok in ("1", "1000", " 7 "):
        assert lq.parse_quota_settings({"SCIFORGE_LIVE_MAX_RUNS_PER_USER": ok,
                                        "SCIFORGE_LIVE_QUOTA_PATH": "/tmp/x.json"}).max_runs == int(ok)
    for bad in ("0", "1001", "-1", "three", "2.5"):
        with pytest.raises(lq.QuotaConfigError, match="SCIFORGE_LIVE_MAX_RUNS_PER_USER"):
            lq.parse_quota_settings({"SCIFORGE_LIVE_MAX_RUNS_PER_USER": bad})
    with pytest.raises(lq.QuotaConfigError, match="absolute"):
        lq.parse_quota_settings({"SCIFORGE_LIVE_QUOTA_PATH": "relative/q.json"})
    assert lq.parse_quota_settings({"SCIFORGE_LIVE_QUOTA_SALT": "pepper", "SCIFORGE_LIVE_QUOTA_PATH": "/tmp/q"}
                                   ).key_scheme == "hmac-sha256"
    assert "pepper" not in repr(lq.parse_quota_settings({"SCIFORGE_LIVE_QUOTA_SALT": "pepper"}))


def test_default_quota_path_is_outside_the_repository():
    real = lq.QuotaSettings.__dataclass_fields__["path"].default_factory   # original (conftest redirects the name)
    repo = Path(__file__).resolve().parents[1]
    for env in ({}, {"XDG_STATE_HOME": "/var/lib/state"}, {"XDG_STATE_HOME": "relative"}):
        path = real(env)
        assert path.name == "live_quota.json" and path.parent.name == "sciforge" and path.is_absolute()
        assert repo not in path.parents
    assert real({"XDG_STATE_HOME": "/var/lib/state"}) == Path("/var/lib/state/sciforge/live_quota.json")


def test_invalid_quota_setting_is_a_config_error_before_any_activity(qpath, monkeypatch):
    monkeypatch.setattr(tempfile, "mkdtemp", lambda *a, **k: pytest.fail("no temp dir"))
    calls = Calls()
    r = live(USER1, env_for(qpath, SCIFORGE_LIVE_MAX_RUNS_PER_USER="0"), calls)
    assert r.status == "error" and "SCIFORGE_LIVE_MAX_RUNS_PER_USER" in r.errors[0]
    assert calls.http == [] and calls.factory == [] and not qpath.exists()


# ------------------------------------------------------------------ store behaviour


def test_hash_keys_and_normalisation():
    assert lq.quota_key(" Quota.User@Example.ORG ", None) == lq.quota_key("quota.user@example.org", None)
    assert lq.quota_key(EMAIL, None).startswith("s1:") and lq.quota_key(EMAIL, "salt").startswith("h1:")
    assert lq.quota_key(EMAIL, "salt") != lq.quota_key(EMAIL, "other") != lq.quota_key(EMAIL, None)
    with pytest.raises(ValueError):
        lq.quota_key("  ", None)


def test_rolling_window_with_fake_clock(qpath):
    s = lq.QuotaSettings(max_runs=3, path=qpath)
    clock = Clock()
    results = []
    for step in (0, 3600, 7200, 7300):
        clock.t = 1_800_000_000.0 + step
        results.append(lq.check_and_record(EMAIL, s, now=clock))
    assert [r.allowed for r in results] == [True, True, True, False]
    assert [r.used for r in results[:3]] == [1, 2, 3] and results[3].remaining == 0
    assert results[3].retry_after_s == pytest.approx(DAY - 7300)
    assert "3 Live runs per user per rolling 24 hours" in results[3].message
    clock.t = 1_800_000_000.0 + DAY + 1          # first run expired -> exactly one slot again
    assert lq.check_and_record(EMAIL, s, now=clock).allowed
    assert not lq.check_and_record(EMAIL, s, now=clock).allowed
    clock.t += 2 * DAY                           # everything expired
    assert lq.check_and_record(EMAIL, s, now=clock).used == 1


def test_store_contains_only_hashes_and_timestamps(qpath):
    s = lq.QuotaSettings(max_runs=3, path=qpath, salt="server-side-salt")
    lq.check_and_record(EMAIL, s, now=Clock())
    text = qpath.read_text()
    data = json.loads(text)
    assert set(data) == {"version", "window_seconds", "users"}
    assert list(data["users"]) == [lq.quota_key(EMAIL, "server-side-salt")]
    for needle in (EMAIL, EMAIL.lower(), "quota.user", "example.org", "server-side-salt"):
        assert needle not in text
    assert (qpath.stat().st_mode & 0o777) == 0o600


@pytest.mark.parametrize("content", ["{not json", json.dumps({"version": 99, "users": {}}),
                                     json.dumps({"version": 1, "users": {"k": ["x"]}}), json.dumps([1, 2])])
def test_corrupt_store_fails_closed(qpath, content):
    qpath.parent.mkdir(parents=True)
    qpath.write_text(content)
    with pytest.raises(lq.QuotaStoreError):
        lq.check_and_record(EMAIL, lq.QuotaSettings(path=qpath), now=Clock())


def test_unwritable_store_fails_closed(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(lq.QuotaStoreError):
        lq.check_and_record(EMAIL, lq.QuotaSettings(path=blocker / "sub" / "q.json"), now=Clock())


def test_concurrent_sessions_never_exceed_the_limit(qpath):
    s = lq.QuotaSettings(max_runs=3, path=qpath)
    out: list[bool] = []
    lock = threading.Lock()

    def worker():
        d = lq.check_and_record(EMAIL, s, now=Clock())
        with lock:
            out.append(d.allowed)

    threads = [threading.Thread(target=worker) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert out.count(True) == 3 and out.count(False) == 9
    assert len(json.loads(qpath.read_text())["users"][lq.quota_key(EMAIL, None)]) == 3


# ------------------------------------------------------------------ service enforcement


def test_runs_one_to_three_allowed_fourth_refused_before_any_activity(qpath, monkeypatch, tmp_path, caplog):
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.INFO, logger="sciforge")
    clock = Clock()
    env = env_for(qpath)
    for n in (1, 2, 3):
        calls = Calls()
        clock.t += 60
        r = live(USER1, env, calls, clock)
        assert r.ok and not r.demo, (n, r.status, r.errors)
        assert calls.factory and calls.http                                  # run really happened
    calls = Calls()
    real_mkdtemp = tempfile.mkdtemp
    monkeypatch.setattr(tempfile, "mkdtemp", lambda *a, **k: pytest.fail("no temp dir after refusal"))
    import sciforge.llm.xai as xai

    monkeypatch.setattr(xai.XAIClient, "__init__", lambda *a, **k: pytest.fail("no xAI client after refusal"))
    monkeypatch.setattr("sciforge.pipeline.run_investigation", lambda *a, **k: pytest.fail("no lookup"))
    r = live(USER1, env, calls, clock)
    assert not r.ok and r.status == "live_quota_exceeded" and "usage limit reached" in r.errors[0]
    assert calls.http == [] and calls.factory == []
    monkeypatch.setattr(tempfile, "mkdtemp", real_mkdtemp)
    log = caplog.text
    assert "live quota decision: allowed:1/3" in log and "live quota decision: denied:3/3" in log
    for needle in (EMAIL, EMAIL.lower(), "quota.user"):
        assert needle not in log and needle not in qpath.read_text() and needle not in r.displayed_text()


def test_separate_users_have_separate_quotas(qpath, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    env = env_for(qpath, SCIFORGE_LIVE_MAX_RUNS_PER_USER="1")
    clock = Clock()
    assert live(USER1, env, Calls(), clock).ok
    assert live(USER1, env, Calls(), clock).status == "live_quota_exceeded"
    assert live(USER2, env, Calls(), clock).ok
    clock.t += DAY + 1
    assert live(USER1, env, Calls(), clock).ok                               # window rolled over


def test_failed_run_still_counts(qpath, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    env = env_for(qpath, SCIFORGE_LIVE_MAX_RUNS_PER_USER="1")

    class Broken(Calls):
        def factory_fn(self, ms):
            raise RuntimeError("model client failed")

    r = live(USER1, env, Broken(), Clock())
    assert r.status == "error"
    assert live(USER1, env, Calls(), Clock()).status == "live_quota_exceeded"


@pytest.mark.parametrize("identity", [svc.LiveIdentity(True, EMAIL, None), svc.LiveIdentity(True, EMAIL, False),
                                      svc.LiveIdentity(True, None, True), svc.ANONYMOUS, None])
def test_missing_or_unverified_identity_fails_closed(qpath, identity):
    calls = Calls()
    r = live(identity, env_for(qpath), calls)
    assert not r.ok and r.status == "live_unauthorized" and calls.http == [] and calls.factory == []
    assert not qpath.exists()


def test_store_error_fails_closed_before_activity(qpath):
    qpath.parent.mkdir(parents=True)
    qpath.write_text("{broken")
    calls = Calls()
    r = live(USER1, env_for(qpath), calls)
    assert r.status == "live_unavailable" and "usage-limit store" in r.errors[0]
    assert calls.http == [] and calls.factory == []


def test_config_error_does_not_use_up_a_run(qpath):
    env = env_for(qpath)
    env.pop("SCIFORGE_MAX_SPEND_USD")                                      # web default $2 needs prices
    r = live(USER1, env, Calls())
    assert r.status == "error" and "SCIFORGE_PRICE_INPUT_PER_MTOK" in r.errors[0]
    assert not qpath.exists()


def test_demo_ignores_quota_and_kill_switch(qpath):
    env = env_for(qpath, SCIFORGE_LIVE_MAX_RUNS_PER_USER="1", SCIFORGE_LIVE_KILL_SWITCH="true")
    for _ in range(3):
        r = svc.run_web_investigation(svc.InvestigationRequest(question=DEMO_QUESTION, identity=USER1), environ=env,
                                      secrets={"auth": AUTH})
        assert r.ok and r.demo
    assert not qpath.exists()


def test_auth_disabled_behaviour_unchanged_no_quota(qpath, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    env = env_for(qpath, SCIFORGE_LIVE_REQUIRE_AUTH="false", SCIFORGE_LIVE_MAX_RUNS_PER_USER="1")
    for _ in range(3):
        r = live(None, env, Calls(), Clock(), secrets={})
        assert r.ok
    assert not qpath.exists()


def test_live_runner_rechecks_quota():
    access = svc.LiveAccessDecision(True, "allowed", require_auth=True, auth_configured=True)
    env = env_for(Path("/nonexistent/q.json"))
    with pytest.raises(svc.ModelConfigError, match="usage limit"):
        svc._run_live(svc.InvestigationRequest(question=DEMO_QUESTION, mode=svc.MODE_LIVE), svc._Progress(None),
                      Path(tempfile.gettempdir()), env, None, [], None, None, None, access=access)


# ------------------------------------------------------------------ kill switch


@pytest.mark.parametrize("value, on", [
    (None, False), ("", False), ("  ", False), ("false", False), ("FALSE", False), (" False ", False),
    ("true", True), ("1", True), ("yes", True), ("on", True), ("TRUE", True), ("0", True), ("no", True),
    ("off", True), ("flase", True),
])
def test_kill_switch_parsing_is_fail_safe(value, on):
    env = {} if value is None else {"SCIFORGE_LIVE_KILL_SWITCH": value}
    assert svc.live_kill_switch(env, {}) is on


def test_kill_switch_secrets_and_precedence():
    assert svc.live_kill_switch({}, {"SCIFORGE_LIVE_KILL_SWITCH": True}) is True
    assert svc.live_kill_switch({}, {"SCIFORGE_LIVE_KILL_SWITCH": False}) is False
    assert svc.live_kill_switch({}, {"SCIFORGE_LIVE_KILL_SWITCH": "on"}) is True
    assert svc.live_kill_switch({}, {"SCIFORGE_LIVE_KILL_SWITCH": 0}) is True        # not "false": on
    assert svc.live_kill_switch({"SCIFORGE_LIVE_KILL_SWITCH": "false"}, {"SCIFORGE_LIVE_KILL_SWITCH": "true"}) is False

    class Broken(dict):
        def get(self, *a, **k):
            raise RuntimeError("secrets broken")

    assert svc.live_kill_switch({}, Broken()) is True


def test_kill_switch_overrides_everything_in_ui_availability(qpath):
    env = env_for(qpath, SCIFORGE_LIVE_REQUIRE_AUTH="false", SCIFORGE_LIVE_KILL_SWITCH="true")
    av = svc.live_availability(env, {}, identity=USER1, auth_configured=True)
    assert not av.available and av.kill_switch and not av.deployment_ready and not av.can_login
    assert av.message.startswith("Live Mode is temporarily switched off")
    assert svc.live_availability(env_for(qpath, SCIFORGE_LIVE_REQUIRE_AUTH="false"), {}).available


@pytest.mark.parametrize("extra", [{}, {"SCIFORGE_LIVE_REQUIRE_AUTH": "false"}])
def test_kill_switch_refused_in_service_before_anything(qpath, monkeypatch, extra):
    import sciforge.llm.xai as xai

    monkeypatch.setattr(tempfile, "mkdtemp", lambda *a, **k: pytest.fail("no temp dir"))
    monkeypatch.setattr(xai.XAIClient, "__init__", lambda *a, **k: pytest.fail("no client"))
    monkeypatch.setattr(svc, "decide_live_access", lambda *a, **k: pytest.fail("kill switch is checked first"))
    calls = Calls()
    r = live(USER1, env_for(qpath, SCIFORGE_LIVE_KILL_SWITCH="yes", **extra), calls)
    assert r.status == "live_unavailable" and "switched off" in r.errors[0]
    assert calls.http == [] and calls.factory == [] and not qpath.exists()


def test_live_runner_rechecks_kill_switch():
    access = svc.LiveAccessDecision(True, "auth_not_required", require_auth=False)
    env = {**env_for(Path("/x/q.json")), "SCIFORGE_LIVE_KILL_SWITCH": "true"}
    with pytest.raises(svc.ModelConfigError, match="switched off"):
        svc._run_live(svc.InvestigationRequest(question=DEMO_QUESTION, mode=svc.MODE_LIVE), svc._Progress(None),
                      Path(tempfile.gettempdir()), env, None, [], None, None, None, access=access)


def test_ui_kill_switch_disables_live(monkeypatch, qpath):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    for k, v in env_for(qpath, SCIFORGE_LIVE_REQUIRE_AUTH="false", SCIFORGE_LIVE_KILL_SWITCH="on").items():
        monkeypatch.setenv(k, v)
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py"), default_timeout=60)
    at.run()
    assert not at.exception and at.radio(key="mode").disabled is True
    assert any("temporarily switched off" in c.value for c in at.caption)


# ------------------------------------------------------------------ web spend default


def test_web_live_uses_two_dollar_cap_when_unset(qpath, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    env = env_for(qpath, SCIFORGE_PRICE_INPUT_PER_MTOK="3", SCIFORGE_PRICE_OUTPUT_PER_MTOK="15")
    env.pop("SCIFORGE_MAX_SPEND_USD")
    calls = Calls()
    r = live(USER1, env, calls, Clock())
    assert r.ok and [float(x) for x in calls.factory] == [2.0]
    assert r.validation["budget"]["spend_cap_usd"] == "2"
    calls = Calls()
    live(USER2, {**env, "SCIFORGE_MAX_SPEND_USD": "5"}, calls, Clock())          # explicit value honoured
    assert [float(x) for x in calls.factory] == [5.0]
    from sciforge.config import ModelSettings

    assert ModelSettings.from_env({"XAI_API_KEY": FAKE_XAI_KEY, "XAI_MODEL": "m", "SCIFORGE_PRICE_INPUT_PER_MTOK": "1",
                                   "SCIFORGE_PRICE_OUTPUT_PER_MTOK": "1"}).max_spend_usd == 15.0   # CLI default kept
