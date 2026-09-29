"""SciForge web app: application-interface module (offline; network guard from conftest applies)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx
import pytest

from conftest import FAKE_XAI_KEY, FAKE_XAI_MODEL, Sleeper, api_handler, mock_client
from m2_support import efetch_xml, question_output, xml_response
from sciforge import app_service as svc
from sciforge.demo_data import DEMO_QUESTION, demo_model_client, demo_records, demo_sources
from sciforge.llm.fake import FakeModelClient
from sciforge.stages.report import render_citation

REJECTED_DEMO_STRINGS = ("DEMO-REJECTED-CLAIM", "DEMO-REJECTED-QUOTE",
                         "preregistration eliminates all publication bias in every field")


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("SCIFORGE_DEBUG_KEEP_REJECTED_RAW", "SCIFORGE_MAX_SOURCE_CHARS", "SCIFORGE_LIVE_ENABLED",
                 "SCIFORGE_LIVE_REQUIRE_AUTH", "SCIFORGE_LIVE_ALLOWED_EMAILS"):
        monkeypatch.delenv(name, raising=False)


# Gate/credential tests opt out of the sign-in layer explicitly (it has its own tests in test_live_auth.py).
NOAUTH = {"SCIFORGE_LIVE_REQUIRE_AUTH": "false"}
GATE = {"SCIFORGE_LIVE_ENABLED": "true", **NOAUTH}
CREDS = {"XAI_API_KEY": FAKE_XAI_KEY, "XAI_MODEL": FAKE_XAI_MODEL}


def demo(**kw) -> svc.WebInvestigationResult:
    kw.setdefault("question", DEMO_QUESTION)
    return svc.run_web_investigation(svc.InvestigationRequest(**kw), environ={})


# ------------------------------------------------------------------ import


def test_app_service_import_has_no_streamlit_dependency():
    code = "import sys, sciforge.app_service, sciforge.demo_data; print('streamlit' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and out.stdout.strip() == "False"


# ------------------------------------------------------------------ input validation


@pytest.mark.parametrize("question, fragment", [
    ("", "Please enter a research question"),
    ("   \n  ", "Please enter a research question"),
    ("too short", "too short"),
    ("x" * (svc.MAX_QUESTION_CHARS + 1), "too long"),
])
def test_question_validation(question, fragment):
    errors = svc.validate_request(svc.InvestigationRequest(question=question))
    assert len(errors) == 1 and fragment in errors[0]


def test_question_length_bounds_are_inclusive():
    assert svc.validate_request(svc.InvestigationRequest(question="q" * svc.MIN_QUESTION_CHARS)) == []
    assert svc.validate_request(svc.InvestigationRequest(question="q" * svc.MAX_QUESTION_CHARS)) == []


@pytest.mark.parametrize("from_year, to_year, fragment", [
    (2024, 2020, "Invalid date range"),
    (1700, None, "Start year"),
    (None, 2200, "End year"),
    ("2020", None, "Start year"),
])
def test_date_range_validation(from_year, to_year, fragment):
    errors = svc.validate_request(svc.InvestigationRequest(question=DEMO_QUESTION, from_year=from_year,
                                                           to_year=to_year))
    assert any(fragment in e for e in errors)


def test_date_range_optional_and_same_year_ok():
    assert svc.validate_request(svc.InvestigationRequest(question=DEMO_QUESTION)) == []
    assert svc.validate_request(svc.InvestigationRequest(question=DEMO_QUESTION, from_year=2021, to_year=2021)) == []


@pytest.mark.parametrize("value", [0, svc.MAX_SOURCES_LIMIT + 1, -3, True, "5", 2.5])
def test_max_sources_bounds(value):
    errors = svc.validate_request(svc.InvestigationRequest(question=DEMO_QUESTION, max_sources=value))
    assert any("Maximum sources" in e for e in errors)


@pytest.mark.parametrize("value", [svc.MIN_SOURCES, svc.MAX_SOURCES_LIMIT])
def test_max_sources_bounds_inclusive(value):
    assert svc.validate_request(svc.InvestigationRequest(question=DEMO_QUESTION, max_sources=value)) == []


def test_invalid_mode_and_invalid_input_result_runs_nothing(monkeypatch):
    called = []
    monkeypatch.setattr(svc, "run_model_investigation", lambda *a, **k: called.append(1))
    result = svc.run_web_investigation(svc.InvestigationRequest(question="", mode="turbo"), environ={})
    assert not result.ok and result.status == "invalid_input" and not called
    assert any("Mode must be" in e for e in result.errors)


# ------------------------------------------------------------------ demo mode end to end


def test_demo_mode_end_to_end_offline():
    events: list[svc.ProgressEvent] = []
    result = svc.run_web_investigation(svc.InvestigationRequest(question=DEMO_QUESTION), progress=events.append,
                                       environ={})
    assert result.ok and result.demo and result.status == "ok" and result.errors == []
    assert "SYNTHETIC DEMO DATA" in result.notices[0]
    assert list(result.sections) == ["A. Research Question", "B. Search Strategy", "C. Key Findings",
                                     "D. Evidence Matrix", "E. Conflicting Evidence", "F. Limitations",
                                     "G. Research Gaps", "H. Candidate Hypotheses", "I. Proposed Next Steps",
                                     "J. Sources"]
    assert [e["evidence_id"] for e in result.evidence] == ["ev_0001", "ev_0002", "ev_0003"]
    assert [e["source_ref"] for e in result.evidence] == ["S1", "S2", "S3"]
    assert [c["evidence_id"] for c in result.conflicts] == ["ev_0003"]
    assert [g["gap_id"] for g in result.gaps] == ["gap_01", "gap_02"]
    assert [h["hypothesis_id"] for h in result.hypotheses] == ["hyp_01", "hyp_02"]
    assert all(h["label"] == "Unvalidated, AI-generated hypothesis for further investigation"
               for h in result.hypotheses)
    # v0.4 stress test in Demo Mode: hyp_01 passes the critic; hyp_02 over-claims causality, is revised
    assert [(h["critic_status"], h["revision_status"]) for h in result.hypotheses] == [
        ("completed", "no revision required"), ("completed", "revised")]
    assert [h["mechanistic_claim_level"] for h in result.hypotheses] == ["association", "association"]
    assert [h["confidence"] for h in result.hypotheses] == ["low", "low"]
    v = result.validation
    assert v["report_status"] == "passed" and v["citations"] == {"resolved": 3, "unresolved": 0}
    assert v["evidence"] == {"accepted": 3, "rejected": 1, "rejection_reasons": {"quote_not_in_source": 1}}
    assert v["privacy_guard"]["rejected_output_leaks_in_run_files"] == 0
    assert v["budget"]["attempts_used"] == 9 and v["budget"]["sources_limit"] == svc.DEFAULT_MAX_SOURCES
    assert result.progress == {k: "done" for k, _ in svc.PROGRESS_STAGES}
    assert {e.stage for e in events} == {k for k, _ in svc.PROGRESS_STAGES}
    assert any("DEMO MODE" in item and "SYNTHETIC" in item for item in result.limitations)


def test_demo_mode_is_deterministic():
    a, b = demo(), demo()
    strip = lambda r: [line for line in r.report_markdown.splitlines() if "**Run:**" not in line]  # noqa: E731
    assert strip(a) == strip(b)
    assert a.sources == b.sources and a.evidence == b.evidence


def test_demo_date_range_and_max_sources_are_applied():
    by_year = demo(from_year=2022, to_year=2023)
    years = {r.record_id: r.year for r in demo_records()[0]}
    assert sorted(years[s["record_id"]] for s in by_year.sources) == [2022, 2023]
    assert len(by_year.evidence) == 2 and by_year.gaps == [] and by_year.hypotheses == []
    assert by_year.progress["hypotheses"] == "skipped"
    one = demo(max_sources=1)
    assert len(one.evidence) == 1 and one.validation["budget"]["sources_used"] == 1
    empty = demo(from_year=2099)
    assert empty.ok and empty.status == "degraded" and empty.evidence == [] and empty.sources == []


def test_demo_writes_nothing_outside_a_deleted_temp_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    created: list[str] = []
    real = tempfile.mkdtemp

    def recording(*a, **k):
        path = real(*a, **k)
        created.append(path)
        return path

    monkeypatch.setattr(tempfile, "mkdtemp", recording)
    result = demo()
    assert result.ok and len(created) == 1 and not Path(created[0]).exists()
    assert list(tmp_path.iterdir()) == []                     # nothing written to the working directory
    assert Path(created[0]).name.startswith(svc.TEMP_PREFIX)


def test_demo_data_is_clearly_synthetic():
    records, verification = demo_records()
    assert records and len(records) == len(verification) == len(demo_sources())
    for r in records:
        assert r.doi.startswith("10.0000/demo.") and r.pmid is None and r.source_url is None
        assert r.title.startswith("[SYNTHETIC DEMO]") and all(a.startswith("Demo Author") for a in r.authors)
        assert "not a real journal" in r.journal
    for v in verification:
        assert any("SYNTHETIC DEMO" in reason for reason in v.reasons)
    for s in demo_sources():
        assert s["abstract"].startswith("SYNTHETIC DEMO ABSTRACT")


def test_demo_model_client_is_a_fake_client():
    client = demo_model_client()
    assert isinstance(client, FakeModelClient) and client.name == "fake"


# ------------------------------------------------------------------ citations


def test_citations_come_only_from_the_deterministic_renderer():
    records, verification = demo_records()
    result = demo()
    by_id = {r.record_id: r for r in records}
    ver = {v.record_id: v for v in verification}
    for s in result.sources:
        expected = render_citation(s["ref"], by_id[s["record_id"]], ver[s["record_id"]],
                                   "crossref_abstract" if s["cited"] else None)
        assert s["citation"] == expected
    assert [s["ref"] for s in result.sources] == ["S1", "S2", "S3", "R1", "R2"]
    assert [s["verification_status"] for s in result.sources] == ["verified"] * 3 + ["partially_verified",
                                                                                       "not_verified"]
    # section J of the report lists exactly the cited citation lines
    j = result.sections["J. Sources"].splitlines()
    assert j[0].startswith("Each source starts with its source type") and j[1] == ""
    assert j[2:] == [s["citation"] for s in result.sources if s["cited"]]


def test_render_sources_unresolved_placeholder_and_determinism():
    records, verification = demo_records()
    citations = [{"ref": "S1", "record_id": records[0].record_id, "status": "resolved"},
                 {"ref": None, "record_id": "rec_ffffffffffffffff", "status": "unresolved"}]
    first = svc.render_sources(records[:2], verification[:2], citations, {})
    second = svc.render_sources(records[:2], verification[:2], citations, {})
    assert first == second
    assert first[0]["citation"] == render_citation("S1", records[0], verification[0], None)
    assert first[1]["citation"].startswith("[UNRESOLVED CITATION: rec_ffffffffffffffff]")
    assert first[1]["resolved"] is False and first[1]["ref"] is None
    assert first[2]["ref"] == "R1" and first[2]["cited"] is False
    # an id that claims to be resolved but has no v0.2 record fails closed too
    ghost = svc.render_sources(records[:1], verification[:1],
                               [{"ref": "S1", "record_id": "rec_0000000000000000", "status": "resolved"}], {})
    assert ghost[0]["citation"].startswith("[UNRESOLVED CITATION: rec_0000000000000000]")


def test_report_sections_split_matches_report():
    result = demo()
    for title, body in result.sections.items():
        assert f"## {title}" in result.report_markdown and body in result.report_markdown
    assert "**Candidate hypothesis:**" in result.sections["H. Candidate Hypotheses"]
    assert "report_validation.json" not in result.sections["J. Sources"]


# ------------------------------------------------------------------ live mode availability


def test_live_mode_disabled_by_default():
    av = svc.live_availability(environ={}, secrets={})
    assert not av.available and not av.gate_enabled and av.missing == ("XAI_API_KEY", "XAI_MODEL")
    assert "disabled for this deployment" in av.message
    assert svc.live_gate_enabled(environ={}) is False


@pytest.mark.parametrize("flag", [None, "", "false", "False", "0", "1", "yes", "on", "enabled", "truee", "t",
                                  "true!", "no"])
def test_live_mode_disabled_with_credentials_when_flag_not_true(flag):
    env = dict(CREDS)
    if flag is not None:
        env["SCIFORGE_LIVE_ENABLED"] = flag
    av = svc.live_availability(environ=env, secrets={})
    assert not av.available and not av.gate_enabled and av.missing == ()
    assert "disabled for this deployment" in av.message
    assert FAKE_XAI_KEY not in av.message and FAKE_XAI_KEY not in repr(av)


@pytest.mark.parametrize("flag", ["true", "TRUE", "True", "  true  ", "\ttrue\n"])
def test_live_mode_enabled_only_with_true_flag_and_both_credentials(flag):
    av = svc.live_availability(environ={**CREDS, **NOAUTH, "SCIFORGE_LIVE_ENABLED": flag})
    assert av.available and av.gate_enabled and av.missing == ()
    assert FAKE_XAI_KEY not in av.message


@pytest.mark.parametrize("env, missing", [
    ({"XAI_API_KEY": FAKE_XAI_KEY}, ("XAI_MODEL",)),
    ({"XAI_MODEL": FAKE_XAI_MODEL}, ("XAI_API_KEY",)),
    ({"XAI_API_KEY": "  ", "XAI_MODEL": ""}, ("XAI_API_KEY", "XAI_MODEL")),
    ({}, ("XAI_API_KEY", "XAI_MODEL")),
])
def test_missing_either_credential_disables_even_with_flag_true(env, missing):
    av = svc.live_availability(environ={**GATE, **env}, secrets={})
    assert not av.available and av.gate_enabled and av.missing == missing
    assert " and ".join(missing) in av.message and FAKE_XAI_KEY not in av.message


def test_gate_and_credentials_from_secrets_with_env_precedence():
    assert svc.live_availability(environ={}, secrets={**GATE, **CREDS}).available
    assert svc.live_availability(environ=GATE, secrets=CREDS).available
    assert svc.live_availability(environ={"XAI_MODEL": FAKE_XAI_MODEL},
                                 secrets={**GATE, "XAI_API_KEY": FAKE_XAI_KEY}).available
    # a non-empty environment value always wins over st.secrets, for the gate too
    assert not svc.live_availability(environ={"SCIFORGE_LIVE_ENABLED": "false"}, secrets={**GATE, **CREDS}).available
    assert svc.live_availability(environ={"SCIFORGE_LIVE_ENABLED": "true"},
                                 secrets={"SCIFORGE_LIVE_ENABLED": "false", **NOAUTH, **CREDS}).available
    # TOML booleans are not the exact string "true": fail closed
    assert not svc.live_availability(environ={}, secrets={"SCIFORGE_LIVE_ENABLED": True, **NOAUTH, **CREDS}).available
    env = svc._model_env({"XAI_MODEL": "env-model"}, {"XAI_MODEL": "secret-model", "XAI_API_KEY": "k" * 20})
    assert env["XAI_MODEL"] == "env-model" and env["XAI_API_KEY"] == "k" * 20


def _boom(*a, **k):
    raise AssertionError("must not run")


def test_live_mode_refuses_to_run_without_credentials(monkeypatch):
    monkeypatch.setattr("sciforge.pipeline.run_investigation", _boom)
    monkeypatch.setattr(svc, "run_model_investigation", _boom)
    result = svc.run_web_investigation(svc.InvestigationRequest(question=DEMO_QUESTION, mode=svc.MODE_LIVE),
                                       environ=GATE)
    assert not result.ok and result.status == "live_unavailable"
    assert "XAI_API_KEY and XAI_MODEL are not configured" in result.errors[0]


@pytest.mark.parametrize("flag", [None, "false", "yes", "1"])
def test_service_refuses_live_request_when_gated_off(monkeypatch, flag):
    """Defence in depth: even with credentials (and a factory), the service never starts a gated live run."""
    import sciforge.llm.xai as xai

    monkeypatch.setattr("sciforge.pipeline.run_investigation", _boom)
    monkeypatch.setattr(svc, "run_model_investigation", _boom)
    monkeypatch.setattr(xai.XAIClient, "__init__", _boom)
    monkeypatch.setattr(tempfile, "mkdtemp", _boom)
    env = {**CREDS, "SCIFORGE_MAX_SPEND_USD": "none"}
    if flag is not None:
        env["SCIFORGE_LIVE_ENABLED"] = flag
    called = []
    result = svc.run_web_investigation(
        svc.InvestigationRequest(question=DEMO_QUESTION, mode=svc.MODE_LIVE), environ=env,
        live_http_client=mock_client(lambda r: called.append(r) or httpx.Response(500)),
        live_model_client_factory=lambda ms: called.append(ms) or FakeModelClient([]))
    assert not result.ok and result.status == "live_unavailable" and called == []
    assert "disabled for this deployment" in result.errors[0]
    assert FAKE_XAI_KEY not in result.displayed_text()


def test_live_mode_gate_rechecked_inside_live_runner():
    with pytest.raises(svc.ModelConfigError, match="disabled for this deployment"):
        svc._run_live(svc.InvestigationRequest(question=DEMO_QUESTION, mode=svc.MODE_LIVE), svc._Progress(None),
                      Path(tempfile.gettempdir()), dict(CREDS), None, [], None, None, None)


def test_demo_mode_makes_no_xai_call_even_with_live_fully_enabled(monkeypatch):
    import sciforge.llm.xai as xai

    monkeypatch.setattr(xai.XAIClient, "__init__", _boom)
    monkeypatch.setattr(xai.XAIClient, "complete", _boom)
    monkeypatch.setattr("sciforge.pipeline.run_investigation", _boom)
    env = {**GATE, **CREDS, "SCIFORGE_MAX_SPEND_USD": "none"}
    result = svc.run_web_investigation(svc.InvestigationRequest(question=DEMO_QUESTION), environ=env)
    assert result.ok and result.demo and result.status == "ok"
    assert FAKE_XAI_KEY not in result.displayed_text()


def test_demo_mode_works_without_any_credentials_or_gate():
    result = svc.run_web_investigation(svc.InvestigationRequest(question=DEMO_QUESTION), environ={}, secrets={})
    assert result.ok and result.status == "ok" and len(result.evidence) == 3


def test_broken_secrets_source_counts_as_missing():
    class Broken(dict):
        def get(self, *a, **k):
            raise FileNotFoundError("no secrets.toml")

    assert not svc.live_availability(environ={}, secrets=Broken()).available


def test_live_mode_config_error_never_shows_the_key(monkeypatch):
    env = {**GATE, **CREDS}                               # spend cap on, no prices -> fail closed
    result = svc.run_web_investigation(svc.InvestigationRequest(question=DEMO_QUESTION, mode=svc.MODE_LIVE),
                                       environ=env)
    assert not result.ok and "SCIFORGE_PRICE_INPUT_PER_MTOK" in result.errors[0]
    assert FAKE_XAI_KEY not in result.displayed_text()


def _live_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("efetch.fcgi"):
        return xml_response(efetch_xml([("111", "<AbstractText>Shear exposure raised platelet activation.</AbstractText>")]))
    return api_handler(request)


def test_live_mode_wiring_offline_with_fake_model_and_mock_transport(tmp_path, monkeypatch):
    """Live path with injected MockTransport + FakeModelClient: no real network, key never displayed."""
    monkeypatch.chdir(tmp_path)
    env = {**GATE, **CREDS, "SCIFORGE_MAX_SPEND_USD": "none", "SCIFORGE_MODEL_MAX_SOURCES": "10"}
    seen = {}

    def responder(request):
        if request.stage == "question":
            # a hostile/buggy model echoing a key-like string and a path must be scrubbed from the display
            return question_output(scope=f"Scope {FAKE_XAI_KEY} see /home/someone/private/notes.txt")
        return {"items": []}

    def factory(model_settings):
        seen["max_sources"] = model_settings.max_sources
        seen["max_attempts"] = model_settings.max_attempts
        return FakeModelClient([responder] * 20)

    result = svc.run_web_investigation(
        svc.InvestigationRequest(question=DEMO_QUESTION, mode=svc.MODE_LIVE, max_sources=3), environ=env,
        live_http_client=mock_client(_live_handler), live_model_client_factory=factory, sleep=Sleeper())
    assert result.ok and not result.demo and result.status == "degraded"
    assert seen == {"max_sources": 3, "max_attempts": 15}           # UI cap applied within the existing budget
    assert "real xAI calls (costs money)" in result.notices[0]
    text = result.displayed_text()
    assert FAKE_XAI_KEY not in text and "/home/someone" not in text
    assert "[REDACTED]" in json.dumps(result.question_definition) and "[path]" in json.dumps(result.question_definition)
    assert all(s["citation"] for s in result.sources)
    # v0.4: the Live path also builds the deterministic evidence graph (empty here: no evidence accepted)
    assert result.evidence_graph["graph_version"] == "v0.4-evidence-graph-1"
    assert result.validation["evidence_graph"]["status"] == "passed" and result.evidence_graph["nodes"] == []
    assert list(tmp_path.iterdir()) == []


def test_live_mode_uses_xai_client_by_default(monkeypatch):
    """Without a factory the existing XAIClient is constructed (its HTTP call is intercepted: no network)."""
    import sciforge.llm.xai as xai

    built = []

    class Recorder(xai.XAIClient):
        def __init__(self, settings, **kw):
            built.append(settings.model)
            super().__init__(settings, http=mock_client(lambda r: httpx.Response(401)))

    monkeypatch.setattr(xai, "XAIClient", Recorder)
    env = {**GATE, **CREDS, "SCIFORGE_MAX_SPEND_USD": "none"}
    result = svc.run_web_investigation(svc.InvestigationRequest(question=DEMO_QUESTION, mode=svc.MODE_LIVE),
                                       environ=env, live_http_client=mock_client(_live_handler), sleep=Sleeper())
    assert built == [FAKE_XAI_MODEL]
    assert result.ok and result.status == "model_auth_error"
    assert FAKE_XAI_KEY not in result.displayed_text()


# ------------------------------------------------------------------ privacy


def test_guard_display_scrubs_keys_paths_and_withheld_text():
    text = (f"key {FAKE_XAI_KEY} Bearer abcd1234efgh api_key=zzz secret-value-123 at /tmp/sciforge-web-x/model/r.md "
            r"and /home/u/.env and C:\Users\u\x.txt; keep https://doi.org/10.0000/demo.1 and and/or. RAW-REJECTED-TEXT")
    out = svc.guard_display({"a": [text]}, secrets=["secret-value-123"], withheld=["RAW-REJECTED-TEXT"])["a"][0]
    for bad in (FAKE_XAI_KEY, "abcd1234efgh", "zzz", "secret-value-123", "/tmp/", "/home/u", "C:\\Users",
                "RAW-REJECTED-TEXT"):
        assert bad not in out
    assert "https://doi.org/10.0000/demo.1" in out and "and/or" in out and svc.WITHHELD in out


def test_no_key_env_values_paths_or_rejected_output_in_demo_display(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", FAKE_XAI_KEY)
    monkeypatch.setenv("NCBI_API_KEY", "ncbi-fake-key-123456")
    result = svc.run_web_investigation(svc.InvestigationRequest(question=DEMO_QUESTION))
    text = result.displayed_text()
    for bad in (FAKE_XAI_KEY, "ncbi-fake-key-123456", svc.TEMP_PREFIX, tempfile.gettempdir() + "/",
                str(Path.home()) + "/", str(Path(__file__).resolve().parents[1]), *REJECTED_DEMO_STRINGS):
        assert bad not in text, bad
    assert "xai-" not in text


def test_question_containing_a_key_is_redacted_in_display():
    result = demo(question=f"Does preregistration matter? {FAKE_XAI_KEY}")
    assert result.ok and FAKE_XAI_KEY not in result.displayed_text()
    assert "[REDACTED]" in result.question


def test_unexpected_failure_shows_generic_message_only(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError(f"secret {FAKE_XAI_KEY} at /home/someone/file")

    monkeypatch.setattr(svc, "run_model_investigation", boom)
    result = demo()
    assert not result.ok and result.errors == ["The investigation failed unexpectedly (RuntimeError)."]
