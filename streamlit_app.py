"""SciForge web app (Streamlit). Thin UI only: all logic lives in ``sciforge.app_service``.

Run locally:  streamlit run streamlit_app.py
Demo Mode (default) is fully offline and uses SYNTHETIC data. Live Mode is enabled
only when SCIFORGE_LIVE_ENABLED=true AND XAI_API_KEY and XAI_MODEL are configured
(values are never displayed).
"""

from __future__ import annotations

import sys
from pathlib import Path

try:  # installed package (pip install -e ".[web]" or requirements.txt)
    import sciforge  # noqa: F401
except ImportError:  # running from a plain checkout: the package lives in app/
    sys.path.insert(0, str(Path(__file__).resolve().parent / "app"))

import streamlit as st

from sciforge import app_service as svc
from sciforge.demo_data import DEMO_LABEL, DEMO_QUESTION, DEMO_TOPIC

MODES = {"Demo": svc.MODE_DEMO, "Live": svc.MODE_LIVE}
ICONS = {"pending": "⚪", "running": "⏳", "done": "✅", "skipped": "⏭️", "failed": "❌"}
TAB_NAMES = ["Overview", "Evidence", "Conflicts", "Research Gaps", "Hypotheses", "Sources", "Validation"]


def _secrets_view() -> dict[str, str]:
    """Only the Live Mode gate + credential names from st.secrets (kept in memory, never rendered)."""
    try:
        return {n: st.secrets[n] for n in svc.LIVE_SETTING_NAMES if n in st.secrets}
    except Exception:  # noqa: BLE001 - no secrets file configured
        return {}


def _progress_line(label: str, state: str, detail: str = "") -> str:
    return f"{ICONS.get(state, '⚪')} **{label}**" + (f" — {detail}" if detail else "")


def _section(result: svc.WebInvestigationResult, title: str) -> None:
    body = result.sections.get(title)
    st.markdown(body if body else "_Not available._")


def _render_result(result: svc.WebInvestigationResult) -> None:
    if result.demo:
        st.warning(f"**{DEMO_LABEL}.** All sources, abstracts, findings, gaps and hypotheses below are synthetic "
                   "and only illustrate how SciForge works.")
    for notice in result.notices:
        st.info(notice)
    tabs = st.tabs(TAB_NAMES)
    with tabs[0]:
        st.markdown(f"**Status:** `{result.status}`")
        _section(result, "A. Research Question")
        st.markdown("#### Key findings")
        _section(result, "C. Key Findings")
        st.markdown("#### Proposed next steps")
        _section(result, "I. Proposed Next Steps")
        st.download_button("Download report (Markdown)", data=result.report_markdown,
                           file_name="sciforge-report.md", mime="text/markdown")
        with st.expander("Full report"):
            st.markdown(result.report_markdown)
    with tabs[1]:
        st.caption("Accepted evidence only: every quote is an exact substring of the source abstract. "
                   "Source refs point to the Sources tab.")
        if result.evidence:
            st.dataframe(result.evidence, width="stretch", hide_index=True)
        else:
            st.markdown("No evidence passed validation.")
    with tabs[2]:
        _section(result, "E. Conflicting Evidence")
        if result.conflicts:
            st.dataframe(result.conflicts, width="stretch", hide_index=True)
    with tabs[3]:
        st.caption("Research gaps are the pipeline's inference from validated evidence, not source statements.")
        _section(result, "G. Research Gaps")
    with tabs[4]:
        st.info("Hypotheses are untested proposals, not findings.")
        _section(result, "H. Candidate Hypotheses")
    with tabs[5]:
        st.caption("Citations are rendered by code from v0.2 records only; the model cannot create citations. "
                   "Unresolvable ids are shown as [UNRESOLVED CITATION: id].")
        cited = [s for s in result.sources if s["cited"]]
        other = [s for s in result.sources if not s["cited"]]
        st.markdown("#### Cited sources")
        st.markdown("\n".join(s["citation"] for s in cited) if cited else "No sources cited.")
        if other:
            st.markdown("#### Other retrieved records (not used as evidence)")
            st.markdown("\n".join(s["citation"] for s in other))
    with tabs[6]:
        v = result.validation
        cols = st.columns(4)
        cols[0].metric("Report validation", str(v.get("report_status")))
        cols[1].metric("Evidence accepted", v.get("evidence", {}).get("accepted", 0))
        cols[2].metric("Evidence rejected", v.get("evidence", {}).get("rejected", 0))
        cols[3].metric("Unresolved citations", v.get("citations", {}).get("unresolved", 0))
        st.markdown("#### Limitations")
        st.markdown("\n".join(f"- {item}" for item in result.limitations))
        _section(result, "F. Limitations")
        st.markdown("#### Validation details")
        st.json(v, expanded=False)


def main() -> None:
    st.set_page_config(page_title="SciForge", page_icon="🔬", layout="wide")
    st.title("SciForge")
    st.subheader("AI for Scientific Discovery")

    secrets = _secrets_view()
    availability = svc.live_availability(secrets=secrets)

    question = st.text_area("Research question", value=DEMO_QUESTION, height=120, key="question",
                            help=f"{svc.MIN_QUESTION_CHARS}–{svc.MAX_QUESTION_CHARS} characters.")
    col_mode, col_sources = st.columns(2)
    with col_mode:
        mode_label = st.radio("Mode", list(MODES), index=0, horizontal=True, key="mode",
                              disabled=not availability.available)
        if not availability.available:
            st.caption(availability.message + " Demo Mode runs offline on synthetic data.")
        else:
            st.caption("Live Mode is experimental and not yet validated against the real xAI API.")
    with col_sources:
        max_sources = st.slider("Maximum sources", svc.MIN_SOURCES, svc.MAX_SOURCES_LIMIT, svc.DEFAULT_MAX_SOURCES,
                                key="max_sources")
    from_year = to_year = None
    if st.checkbox("Limit by publication year (optional)", key="use_dates"):
        col_from, col_to = st.columns(2)
        from_year = int(col_from.number_input("Start year", svc.MIN_YEAR, svc.MAX_YEAR, 2015, key="from_year"))
        to_year = int(col_to.number_input("End year", svc.MIN_YEAR, svc.MAX_YEAR, 2026, key="to_year"))
    mode = MODES.get(mode_label, svc.MODE_DEMO) if availability.available else svc.MODE_DEMO
    if mode == svc.MODE_DEMO:
        st.caption(f"Demo Mode example investigation (synthetic): {DEMO_TOPIC}.")

    if st.button("Investigate", type="primary", key="investigate"):
        request = svc.InvestigationRequest(question=question, from_year=from_year, to_year=to_year,
                                           max_sources=int(max_sources), mode=mode)
        errors = svc.validate_request(request)
        if errors:
            st.session_state.pop("result", None)
            for error in errors:
                st.error(error)
        else:
            st.markdown("#### Progress")
            slots = {key: st.empty() for key, _ in svc.PROGRESS_STAGES}
            labels = dict(svc.PROGRESS_STAGES)
            for key, label in svc.PROGRESS_STAGES:
                slots[key].markdown(_progress_line(label, "pending"))

            def on_progress(event: svc.ProgressEvent) -> None:
                if event.stage in slots:
                    slots[event.stage].markdown(_progress_line(labels[event.stage], event.state, event.detail))

            with st.spinner("Investigating..."):
                result = svc.run_web_investigation(request, progress=on_progress, secrets=secrets)
            st.session_state["result"] = result

    result = st.session_state.get("result")
    if result is not None:
        if not result.ok:
            for error in result.errors:
                st.error(error)
        else:
            _render_result(result)


main()
