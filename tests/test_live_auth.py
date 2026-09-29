"""Public Live Mode protection: sign-in (Streamlit OIDC) + email allowlist, fail closed (offline only)."""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path

import httpx
import pytest

from conftest import FAKE_XAI_KEY, FAKE_XAI_MODEL, Sleeper, api_handler, mock_client
from m2_support import efetch_xml, question_output, xml_response
from sciforge import app_service as svc
from sciforge.demo_data import DEMO_QUESTION
from sciforge.llm.fake import FakeModelClient

EMAIL = "Allowed.Person@Example.org"
AUTH = {"redirect_uri": "https://example.invalid/oauth2callback", "cookie_secret": "placeholder-cookie",
        "client_id": "placeholder-id", "client_secret": "placeholder-secret",
        "server_metadata_url": "https://example.invalid/.well-known/openid-configuration"}
BASE = {"SCIFORGE_LIVE_ENABLED": "true", "XAI_API_KEY": FAKE_XAI_KEY, "XAI_MODEL": FAKE_XAI_MODEL,
        "SCIFORGE_MAX_SPEND_USD": "none"}
ALLOW = {"SCIFORGE_LIVE_ALLOWED_EMAILS": " someone@else.org , allowed.person@example.ORG "}
ALLOWED = svc.LiveIdentity(is_logged_in=True, email=EMAIL, email_verified=True)
OTHER = svc.LiveIdentity(is_logged_in=True, email="intruder@example.org", email_verified=True)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in ("SCIFORGE_LIVE_ENABLED", "SCIFORGE_LIVE_REQUIRE_AUTH", "SCIFORGE_LIVE_ALLOWED_EMAILS",
                 "SCIFORGE_DEBUG_KEEP_REJECTED_RAW", "SCIFORGE_MAX_SOURCE_CHARS"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def authlib(monkeypatch):
    """Pretend Authlib is installed (it is not in this offline environment)."""
    monkeypatch.setattr(svc, "authlib_available", lambda: True)


# ------------------------------------------------------------------ parsing


@pytest.mark.parametrize("value, expected", [
    (None, True), ("", True), ("true", True), ("0", True), ("no", True), ("off", True), ("flase", True),
    ("false", False), ("FALSE", False), (" false \n", False),
])
def test_require_auth_is_fail_closed(value, expected):
    env = {} if value is None else {"SCIFORGE_LIVE_REQUIRE_AUTH": value}
    assert svc.live_require_auth(env, {}) is expected


def test_require_auth_toml_boolean_and_env_precedence():
    assert svc.live_require_auth({}, {"SCIFORGE_LIVE_REQUIRE_AUTH": False}) is True        # not the string
    assert svc.live_require_auth({}, {"SCIFORGE_LIVE_REQUIRE_AUTH": "false"}) is False
    assert svc.live_require_auth({"SCIFORGE_LIVE_REQUIRE_AUTH": "true"},
                                 {"SCIFORGE_LIVE_REQUIRE_AUTH": "false"}) is True


def test_allowlist_parsing():
    assert svc.live_allowed_emails(ALLOW, {}) == {"someone@else.org", "allowed.person@example.org"}
    assert svc.live_allowed_emails({}, {"SCIFORGE_LIVE_ALLOWED_EMAILS": ["A@x.org", " b@y.org,c@z.org "]}) == {
        "a@x.org", "b@y.org", "c@z.org"}
    assert svc.live_allowed_emails({"SCIFORGE_LIVE_ALLOWED_EMAILS": "env@x.org"},
                                   {"SCIFORGE_LIVE_ALLOWED_EMAILS": "sec@x.org"}) == {"env@x.org"}
    assert svc.live_allowed_emails({}, {}) == frozenset()
    assert svc.live_allowed_emails({"SCIFORGE_LIVE_ALLOWED_EMAILS": " , ,"}, {}) == frozenset()


def test_auth_section_presence_only():
    assert svc.auth_section_configured(AUTH)
    assert not svc.auth_section_configured(None)
    assert not svc.auth_section_configured({**AUTH, "client_secret": " "})
    assert not svc.auth_section_configured({k: v for k, v in AUTH.items() if k != "server_metadata_url"})
    assert not svc.auth_section_configured("not a table")


def test_streamlit_auth_needs_authlib(monkeypatch):
    monkeypatch.setattr(svc, "authlib_available", lambda: False)
    assert not svc.streamlit_auth_configured(AUTH)
    monkeypatch.setattr(svc, "authlib_available", lambda: True)
    assert svc.streamlit_auth_configured(AUTH)


class TokenTrap(Mapping):
    """st.user stand-in that fails if anything other than the three identity fields is read."""

    def __init__(self, data):
        self.data = data

    def __getitem__(self, key):
        if key not in ("is_logged_in", "email", "email_verified"):
            raise AssertionError(f"identity field {key!r} must not be read")
        return self.data[key]

    def __iter__(self) -> Iterator:
        raise AssertionError("st.user must not be iterated")

    def __len__(self):
        return len(self.data)


def test_identity_from_user_reads_only_identity_fields():
    user = TokenTrap({"is_logged_in": True, "email": f" {EMAIL} ", "email_verified": True,
                      "tokens": {"id": "never", "access": "never"}})
    ident = svc.identity_from_user(user)
    assert ident == svc.LiveIdentity(True, EMAIL, True)
    assert EMAIL not in repr(ident)
    assert svc.identity_from_user({"email": "test@example.com"}) == svc.ANONYMOUS       # no is_logged_in
    assert svc.identity_from_user({"is_logged_in": "yes", "email": EMAIL}) == svc.ANONYMOUS
    assert svc.identity_from_user(object()) == svc.ANONYMOUS


# ------------------------------------------------------------------ decisions


@pytest.mark.parametrize("identity, env, reason, allowed", [
    (None, ALLOW, "anonymous", False),
    (svc.ANONYMOUS, ALLOW, "anonymous", False),
    (ALLOWED, ALLOW, "allowed", True),
    (OTHER, ALLOW, "not_allowlisted", False),
    (ALLOWED, {}, "not_allowlisted", False),                                          # empty allowlist
    (svc.LiveIdentity(True, EMAIL, False), ALLOW, "email_unverified", False),
    (svc.LiveIdentity(True, None, True), ALLOW, "no_email", False),
    (svc.LiveIdentity(True, EMAIL, None), ALLOW, "allowed", True),                    # provider omits the flag
    ("not an identity", ALLOW, "anonymous", False),
])
def test_access_decisions(authlib, identity, env, reason, allowed):
    d = svc.decide_live_access(identity, environ=env, secrets={}, auth_configured=True)
    assert (d.reason, d.allowed) == (reason, allowed)
    assert d.log_value == ("allowed" if allowed else f"denied:{reason}")
    assert EMAIL.lower() not in d.message.lower() and "@" not in d.log_value


def test_auth_unconfigured_or_authlib_missing_is_fail_closed(monkeypatch):
    monkeypatch.setattr(svc, "authlib_available", lambda: True)
    assert svc.decide_live_access(ALLOWED, environ=ALLOW, auth_configured=False).reason == "auth_not_configured"
    monkeypatch.setattr(svc, "authlib_available", lambda: False)
    assert svc.decide_live_access(ALLOWED, environ=ALLOW, auth_configured=True).reason == "auth_not_configured"


def test_auth_not_required_keeps_previous_behaviour():
    d = svc.decide_live_access(None, environ={"SCIFORGE_LIVE_REQUIRE_AUTH": "false"}, auth_configured=False)
    assert d.allowed and d.reason == "auth_not_required"


def test_availability_combines_gate_credentials_and_access(authlib):
    av = svc.live_availability({**BASE, **ALLOW}, {}, identity=svc.ANONYMOUS, auth_configured=True)
    assert not av.available and av.deployment_ready and av.can_login and "log in" in av.message.lower()
    av = svc.live_availability({**BASE, **ALLOW}, {}, identity=ALLOWED, auth_configured=True)
    assert av.available and not av.can_login
    av = svc.live_availability({**BASE, **ALLOW}, {}, identity=OTHER, auth_configured=True)
    assert not av.available and not av.can_login and "not authorized" in av.message
    av = svc.live_availability({**BASE, **ALLOW}, {}, identity=ALLOWED, auth_configured=False)
    assert not av.available and not av.can_login and "sign-in is not configured" in av.message
    gate_off = {**BASE, **ALLOW, "SCIFORGE_LIVE_ENABLED": "false"}
    av = svc.live_availability(gate_off, {}, identity=ALLOWED, auth_configured=True)
    assert not av.available and not av.can_login and "disabled for this deployment" in av.message


# ------------------------------------------------------------------ service enforcement


def _boom(*a, **k):
    raise AssertionError("must not run before authorization")


@pytest.fixture
def no_activity(monkeypatch):
    """Fail on any literature lookup, temp dir, or model-client creation."""
    import sciforge.llm.xai as xai

    monkeypatch.setattr("sciforge.pipeline.run_investigation", _boom)
    monkeypatch.setattr(svc, "run_model_investigation", _boom)
    monkeypatch.setattr(xai.XAIClient, "__init__", _boom)
    monkeypatch.setattr(tempfile, "mkdtemp", _boom)
    calls: list = []
    return calls


def _live(identity, env, calls, *, secrets=None, auth_configured=None):
    return svc.run_web_investigation(
        svc.InvestigationRequest(question=DEMO_QUESTION, mode=svc.MODE_LIVE, identity=identity),
        environ=env, secrets=secrets if secrets is not None else {}, auth_configured=auth_configured,
        live_http_client=mock_client(lambda r: calls.append(r) or httpx.Response(500)),
        live_model_client_factory=lambda ms: calls.append(ms) or FakeModelClient([]))


@pytest.mark.parametrize("identity, env, status", [
    (None, {**BASE, **ALLOW}, "live_unauthorized"),                          # anonymous
    (svc.ANONYMOUS, {**BASE, **ALLOW}, "live_unauthorized"),
    (OTHER, {**BASE, **ALLOW}, "live_unauthorized"),                         # not allowlisted
    (ALLOWED, dict(BASE), "live_unauthorized"),                              # empty allowlist
    (svc.LiveIdentity(True, EMAIL, False), {**BASE, **ALLOW}, "live_unauthorized"),
    (ALLOWED, {**BASE, **ALLOW, "SCIFORGE_LIVE_ENABLED": "false"}, "live_unavailable"),   # gate off
    (ALLOWED, {**ALLOW, "SCIFORGE_LIVE_ENABLED": "true"}, "live_unavailable"),            # no key/model
])
def test_service_refuses_before_any_lookup_or_client(authlib, no_activity, caplog, identity, env, status):
    caplog.set_level(logging.DEBUG)
    result = _live(identity, env, no_activity, auth_configured=True)
    assert not result.ok and result.status == status and no_activity == []
    text = result.displayed_text() + caplog.text
    assert EMAIL.lower() not in text.lower() and "intruder@" not in text and FAKE_XAI_KEY not in text


def test_service_refuses_when_auth_section_missing(authlib, no_activity):
    result = _live(ALLOWED, {**BASE, **ALLOW}, no_activity)                  # secrets without [auth]
    assert result.status == "live_unavailable" and "sign-in is not configured" in result.errors[0]
    assert no_activity == []


def test_service_refuses_when_authlib_missing(monkeypatch, no_activity):
    monkeypatch.setattr(svc, "authlib_available", lambda: False)
    result = _live(ALLOWED, {**BASE, **ALLOW}, no_activity, secrets={"auth": AUTH})
    assert result.status == "live_unavailable" and no_activity == []
    result = _live(ALLOWED, {**BASE, **ALLOW}, no_activity, auth_configured=True)     # UI claim re-checked
    assert result.status == "live_unavailable" and no_activity == []


def test_denial_is_logged_without_identity(authlib, no_activity, caplog):
    caplog.set_level(logging.INFO, logger="sciforge")
    _live(OTHER, {**BASE, **ALLOW}, no_activity, auth_configured=True)
    assert "live access decision: denied:not_allowlisted" in caplog.text
    assert "intruder" not in caplog.text


def _live_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("efetch.fcgi"):
        return xml_response(efetch_xml([("111", "<AbstractText>Shear exposure raised platelet activation.</AbstractText>")]))
    return api_handler(request)


def test_allowed_user_runs_live_with_auth_from_secrets(authlib, tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.INFO, logger="sciforge")
    built = []

    def factory(ms):
        built.append(ms.max_sources)
        return FakeModelClient([lambda r: question_output() if r.stage == "question" else {"items": []}] * 20)

    result = svc.run_web_investigation(
        svc.InvestigationRequest(question=DEMO_QUESTION, mode=svc.MODE_LIVE, max_sources=3, identity=ALLOWED),
        environ={**BASE, **ALLOW}, secrets={"auth": AUTH}, live_http_client=mock_client(_live_handler),
        live_model_client_factory=factory, sleep=Sleeper())
    assert result.ok and result.status in ("ok", "degraded") and built == [3]
    assert "live access decision: allowed" in caplog.text and EMAIL.lower() not in caplog.text.lower()
    assert EMAIL.lower() not in result.displayed_text().lower()


def test_live_runner_rechecks_authorization():
    with pytest.raises(svc.ModelConfigError, match="not authorized"):
        svc._run_live(svc.InvestigationRequest(question=DEMO_QUESTION, mode=svc.MODE_LIVE), svc._Progress(None),
                      Path(tempfile.gettempdir()), dict(BASE), None, [], None, None, None)
    denied = svc.LiveAccessDecision(False, "anonymous")
    with pytest.raises(svc.ModelConfigError, match="not authorized"):
        svc._run_live(svc.InvestigationRequest(question=DEMO_QUESTION, mode=svc.MODE_LIVE), svc._Progress(None),
                      Path(tempfile.gettempdir()), dict(BASE), None, [], None, None, None, access=denied)


def test_demo_mode_needs_no_sign_in(monkeypatch):
    import sciforge.llm.xai as xai

    monkeypatch.setattr(xai.XAIClient, "__init__", _boom)
    for env in ({}, {**BASE, **ALLOW}, {"SCIFORGE_LIVE_REQUIRE_AUTH": "true"}):
        result = svc.run_web_investigation(svc.InvestigationRequest(question=DEMO_QUESTION), environ=env,
                                           secrets={})
        assert result.ok and result.demo and result.status == "ok"


# ------------------------------------------------------------------ Streamlit UI


APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")


def _app(secrets):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=60)
    at.secrets = dict(secrets)
    at.run()
    assert not at.exception
    return at


def _captions(at):
    return "\n".join(c.value for c in at.caption)


def test_ui_live_unavailable_when_sign_in_not_configured(monkeypatch):
    pytest.importorskip("streamlit")
    for k, v in BASE.items():
        monkeypatch.setenv(k, v)
    at = _app({"SCIFORGE_LIVE_ALLOWED_EMAILS": EMAIL})
    assert at.radio(key="mode").disabled is True
    assert "sign-in is not configured" in _captions(at)
    assert not [b for b in at.button if b.key == "login"]


def test_ui_anonymous_user_sees_login_button(monkeypatch, authlib):
    pytest.importorskip("streamlit")
    for k, v in BASE.items():
        monkeypatch.setenv(k, v)
    at = _app({"SCIFORGE_LIVE_ALLOWED_EMAILS": EMAIL, "auth": AUTH})
    assert at.radio(key="mode").disabled is True
    assert "Please log in" in _captions(at)
    assert [b.label for b in at.button if b.key == "login"] == ["Log in for Live Mode"]
    at.button(key="investigate").click().run()                                # Demo still works
    assert not at.exception and any("SYNTHETIC DEMO DATA" in w.value for w in at.warning)


def test_ui_allowed_and_disallowed_users(monkeypatch, authlib):
    pytest.importorskip("streamlit")
    for k, v in BASE.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(svc, "identity_from_user", lambda user: ALLOWED)
    at = _app({"SCIFORGE_LIVE_ALLOWED_EMAILS": EMAIL, "auth": AUTH})
    assert at.radio(key="mode").disabled is False
    assert [b.label for b in at.button if b.key == "logout"] == ["Log out"]
    rendered = "\n".join(str(getattr(e, "value", "")) for e in [*at.caption, *at.markdown, *at.info])
    assert EMAIL.lower() not in rendered.lower()
    monkeypatch.setattr(svc, "identity_from_user", lambda user: OTHER)
    at = _app({"SCIFORGE_LIVE_ALLOWED_EMAILS": EMAIL, "auth": AUTH})
    assert at.radio(key="mode").disabled is True and "not authorized" in _captions(at)
