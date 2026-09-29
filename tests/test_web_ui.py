"""SciForge web app: Streamlit UI tests with streamlit.testing.v1.AppTest (offline; network guard applies)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from conftest import FAKE_XAI_KEY, FAKE_XAI_MODEL  # noqa: E402
from sciforge import app_service as svc  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")
TABS = ["Overview", "Evidence", "Conflicts", "Research Gaps", "Hypotheses", "Sources", "Validation"]
STAGE_LABELS = [label for _, label in svc.PROGRESS_STAGES]


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("SCIFORGE_DEBUG_KEEP_REJECTED_RAW", "SCIFORGE_MAX_SOURCE_CHARS", "SCIFORGE_LIVE_ENABLED"):
        monkeypatch.delenv(name, raising=False)


def app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    return at


def rendered_text(at: AppTest) -> str:
    """Every visible string the test harness can reach."""
    parts: list[str] = []
    for kind in ("title", "subheader", "markdown", "caption", "info", "warning", "error", "success", "text",
                 "code", "metric", "json", "exception"):
        for el in getattr(at, kind, []):
            for attr in ("value", "label", "body"):
                v = getattr(el, attr, None)
                if v is not None:
                    parts.append(str(v))
    for df in at.dataframe:
        parts.append(df.value.to_csv())
    for w in [*at.text_area, *at.radio, *at.slider, *at.checkbox, *at.number_input, *at.button]:
        parts.append(str(w.label))
        parts.append(str(getattr(w, "help", "") or ""))
    return "\n".join(parts)


def test_app_imports_and_renders_header_and_controls():
    at = app()
    assert [t.value for t in at.title] == ["SciForge"]
    assert [s.value for s in at.subheader] == ["AI for Scientific Discovery"]
    assert at.text_area(key="question").label == "Research question"
    assert at.radio(key="mode").options == ["Demo", "Live"] and at.radio(key="mode").value == "Demo"
    assert at.slider(key="max_sources").label == "Maximum sources"
    assert at.checkbox(key="use_dates").value is False
    assert at.button(key="investigate").label == "Investigate"


def test_live_mode_disabled_by_default():
    at = app()
    assert at.radio(key="mode").disabled is True
    assert any("Live Mode is disabled for this deployment" in c.value for c in at.caption)


@pytest.mark.parametrize("flag", [None, "false", "yes", "1"])
def test_live_mode_unavailable_when_flag_not_true_even_with_credentials(monkeypatch, flag):
    monkeypatch.setenv("XAI_API_KEY", FAKE_XAI_KEY)
    monkeypatch.setenv("XAI_MODEL", FAKE_XAI_MODEL)
    if flag is not None:
        monkeypatch.setenv("SCIFORGE_LIVE_ENABLED", flag)
    at = app()
    assert at.radio(key="mode").disabled is True
    assert any("Live Mode is disabled for this deployment" in c.value for c in at.caption)
    assert FAKE_XAI_KEY not in rendered_text(at)


def test_live_mode_unavailable_with_flag_true_but_missing_model(monkeypatch):
    monkeypatch.setenv("SCIFORGE_LIVE_ENABLED", "true")
    monkeypatch.setenv("XAI_API_KEY", FAKE_XAI_KEY)
    at = app()
    assert at.radio(key="mode").disabled is True
    assert any("XAI_MODEL is not configured" in c.value for c in at.caption)


def test_demo_run_in_ui_makes_no_xai_call(monkeypatch):
    import sciforge.llm.xai as xai

    def boom(*a, **k):
        raise AssertionError("XAIClient must not be used in Demo Mode")

    monkeypatch.setattr(xai.XAIClient, "__init__", boom)
    monkeypatch.setattr(xai.XAIClient, "complete", boom)
    at = app()
    at.button(key="investigate").click().run()
    assert not at.exception and not at.error and [t.label for t in at.tabs] == TABS


def test_live_mode_enabled_with_flag_and_credentials_and_key_never_rendered(monkeypatch):
    monkeypatch.setenv("SCIFORGE_LIVE_ENABLED", "true")
    monkeypatch.setenv("XAI_API_KEY", FAKE_XAI_KEY)
    monkeypatch.setenv("XAI_MODEL", FAKE_XAI_MODEL)
    at = app()
    assert at.radio(key="mode").disabled is False
    at.button(key="investigate").click().run()          # still Demo Mode by default
    assert not at.exception
    text = rendered_text(at)
    assert FAKE_XAI_KEY not in text and "xai-" not in text
    assert FAKE_XAI_MODEL not in text


def test_demo_mode_end_to_end_in_ui():
    at = app()
    at.button(key="investigate").click().run()
    assert not at.exception and not at.error
    assert [t.label for t in at.tabs] == TABS
    md = [m.value for m in at.markdown]
    for label in STAGE_LABELS:
        assert any(f"✅ **{label}**" in m for m in md), label
    assert any("SYNTHETIC DEMO DATA" in w.value for w in at.warning)
    text = rendered_text(at)
    assert "Hypothesis: Strict measurement standards may reduce" in text
    assert "[UNRESOLVED CITATION: rec_" not in text


def test_ui_citations_are_the_rendered_v02_citations():
    at = app()
    at.button(key="investigate").click().run()
    result = svc.run_web_investigation(svc.InvestigationRequest(question=at.text_area(key="question").value),
                                       environ={})
    md = "\n".join(m.value for m in at.markdown)
    for s in result.sources:
        assert s["citation"] in md
    assert "10.0000/demo.1" in md and "Demo Author A" in md


@pytest.mark.parametrize("question, fragment", [
    ("", "Please enter a research question"),
    ("short", "too short"),
    ("x" * (svc.MAX_QUESTION_CHARS + 1), "too long"),
])
def test_ui_question_validation(question, fragment):
    at = app()
    at.text_area(key="question").input(question)
    at.button(key="investigate").click().run()
    assert not at.exception
    assert any(fragment in e.value for e in at.error)
    assert not at.tabs


def test_ui_invalid_date_range():
    at = app()
    at.checkbox(key="use_dates").check().run()
    at.number_input(key="from_year").set_value(2025)
    at.number_input(key="to_year").set_value(2020)
    at.button(key="investigate").click().run()
    assert any("Invalid date range" in e.value for e in at.error)
    assert not at.tabs


def test_ui_valid_date_range_and_max_sources():
    at = app()
    at.checkbox(key="use_dates").check().run()
    at.number_input(key="from_year").set_value(2021)
    at.number_input(key="to_year").set_value(2021)
    at.slider(key="max_sources").set_value(1)
    at.button(key="investigate").click().run()
    assert not at.exception and not at.error
    assert len(at.dataframe[0].value) == 1                    # one evidence row (one synthetic source in 2021)


def test_ui_max_sources_slider_bounds():
    at = app()
    slider = at.slider(key="max_sources")
    assert slider.min == svc.MIN_SOURCES and slider.max == svc.MAX_SOURCES_LIMIT


def test_ui_privacy_no_secrets_paths_or_rejected_output(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", FAKE_XAI_KEY)
    monkeypatch.setenv("NCBI_API_KEY", "ncbi-fake-key-123456")
    at = app()
    at.button(key="investigate").click().run()
    text = rendered_text(at)
    for bad in (FAKE_XAI_KEY, "ncbi-fake-key-123456", svc.TEMP_PREFIX, str(Path.home()) + "/",
                str(Path(APP).parent), "DEMO-REJECTED-CLAIM", "DEMO-REJECTED-QUOTE"):
        assert bad not in text, bad
