"""SciForge web app (Streamlit). Thin UI only: all logic lives in ``sciforge.app_service``.

Run locally:  streamlit run streamlit_app.py
Demo Mode (default) is fully offline, public and uses SYNTHETIC data. Live Mode is enabled
only when SCIFORGE_LIVE_ENABLED=true AND XAI_API_KEY and XAI_MODEL are configured
(values are never displayed) AND — unless SCIFORGE_LIVE_REQUIRE_AUTH is exactly "false" —
the user is signed in via Streamlit OIDC (st.login, [auth] in secrets.toml, Authlib installed)
with an email on SCIFORGE_LIVE_ALLOWED_EMAILS. The service layer re-checks all of this.
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
from sciforge import evidence_graph as eg
from sciforge.demo_data import DEMO_LABEL, DEMO_QUESTION, DEMO_TOPIC

MODES = {"Demo": svc.MODE_DEMO, "Live": svc.MODE_LIVE}
ICONS = {"pending": "⚪", "running": "⏳", "done": "✅", "skipped": "⏭️", "failed": "❌"}
TAB_NAMES = ["Overview", "Evidence", "Conflicts", "Research Gaps", "Hypotheses", "Evidence Graph", "Sources",
             "Validation"]


def _secrets_view() -> dict[str, object]:
    """Only the Live Mode gate/credential/auth-policy names from st.secrets (kept in memory, never rendered)."""
    try:
        view: dict[str, object] = {n: st.secrets[n] for n in svc.LIVE_SETTING_NAMES if n in st.secrets}
    except Exception:  # noqa: BLE001 - no secrets file configured
        return {}
    return view


def _auth_configured() -> bool:
    """Streamlit OIDC usable: complete [auth] section (presence only) and Authlib installed."""
    try:
        section = st.secrets.get("auth") if "auth" in st.secrets else None
    except Exception:  # noqa: BLE001
        return False
    return svc.streamlit_auth_configured(section)


def _quota_connection_configured(secrets: dict[str, object]) -> bool:
    """[connections.<name>] present for the postgres usage-limit backend (presence only; values never read out)."""
    name = svc.quota_connection_name(secrets=secrets)
    try:
        connections = st.secrets.get("connections") if "connections" in st.secrets else None
        section = connections.get(name) if connections is not None else None
    except Exception:  # noqa: BLE001 - no secrets file configured
        return False
    return svc.sql_connection_configured(section)


def _quota_engine(name: str):
    """SQLAlchemy engine of the Streamlit SQL connection used by the postgres usage-limit backend."""
    return st.connection(name, type="sql").engine


def _identity(auth_configured: bool) -> svc.LiveIdentity:
    """Signed-in identity from st.user (is_logged_in/email/email_verified only; tokens are never read)."""
    if not auth_configured:
        return svc.ANONYMOUS
    try:
        return svc.identity_from_user(st.user)
    except Exception:  # noqa: BLE001
        return svc.ANONYMOUS


def _progress_line(label: str, state: str, detail: str = "") -> str:
    return f"{ICONS.get(state, '⚪')} **{label}**" + (f" — {detail}" if detail else "")


def _section(result: svc.WebInvestigationResult, title: str) -> None:
    body = result.sections.get(title)
    st.markdown(body if body else "_Not available._")


def _render_graph(result: svc.WebInvestigationResult) -> None:
    """Evidence-to-hypothesis graph: DOT rendered client-side by st.graphviz_chart + selector navigation."""
    graph = result.evidence_graph
    if not graph or not graph.get("nodes"):
        st.markdown("No evidence graph is available for this run.")
        return
    st.warning(f"**{svc.HYPOTHESIS_DISCLAIMER}**")
    v = graph.get("validation") or {}
    st.caption(f"Built by code from validated outputs only (no model call). Graph {graph.get('graph_version')}; "
               f"validation: {v.get('status')}; contradiction edges only where a research gap records the "
               "conflict. Graph nodes are not clickable: pick a node below to see its details.")
    st.graphviz_chart(eg.graph_to_dot(graph), width="stretch")
    st.markdown("**Legend** — " + "; ".join(f"`{t}`: {eg.LEGEND[t]}" for t in eg.EDGE_TYPES))
    options = eg.node_options(graph)
    labels = {nid: label for nid, label in options}
    selected = st.selectbox("Node", [nid for nid, _ in options], format_func=lambda nid: labels[nid],
                            key="graph_node")
    details = eg.node_details(graph, selected) if selected else None
    if details:
        st.markdown(f"#### {details['type_label']} `{details['id']}`")
        st.dataframe([{"Field": k, "Value": str(val)} for k, val in details["fields"].items()],
                     width="stretch", hide_index=True)
        rel = [{"Relation": k, "Nodes": ", ".join(val) or "none"} for k, val in details["related"].items()]
        if rel:
            st.dataframe(rel, width="stretch", hide_index=True)
    if graph.get("excluded"):
        st.caption(f"{len(graph['excluded'])} item(s) excluded from the graph (reasons recorded in "
                   "evidence_graph.json).")


def _render_result(result: svc.WebInvestigationResult) -> None:
    if result.demo:
        st.warning(f"**{DEMO_LABEL}.** All sources, abstracts, findings, gaps and hypotheses below are synthetic "
                   "and only illustrate how SciForge works (no real literature search, no xAI calls).")
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
                   f"{svc.DETERMINISTIC_VALIDATION_NOTE} {svc.SEMANTIC_ENTAILMENT_NOTE} "
                   "Source refs point to the Sources tab; the Source type column comes from bibliographic metadata.")
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
        st.warning(f"**{svc.HYPOTHESIS_DISCLAIMER}**")
        st.caption("Each hypothesis was generated, reviewed by a separate critic call and revised only when needed; "
                   "deterministic validation (evidence links, claim level, prediction, alternative, falsification "
                   "test, citations) outranks the model and the critic. Confidence is qualitative and capped by "
                   "code. 'Not stress-tested' means the critic did not run.")
        if result.hypotheses:
            st.dataframe([{"ID": h["hypothesis_id"], "Label": h["label"],
                           "Claim level": h["mechanistic_claim_level"].replace("_", " "),
                           "Confidence": h["confidence"], "Critic": h["critic_status"],
                           "Revision": h["revision_status"]} for h in result.hypotheses],
                         width="stretch", hide_index=True)
        _section(result, "H. Candidate Hypotheses")
    with tabs[5]:
        _render_graph(result)
    with tabs[6]:
        st.caption("Citations are rendered by code from v0.2 records only; the model cannot create citations. "
                   "Unresolvable ids are shown as [UNRESOLVED CITATION: id]. Each source starts with its source "
                   f"type from bibliographic metadata. {svc.PREPRINT_NOTE} 'Peer-reviewed journal article' is a "
                   "metadata label, not a guarantee of peer review.")
        cited = [s for s in result.sources if s["cited"]]
        other = [s for s in result.sources if not s["cited"]]
        if cited:
            st.dataframe([{"Source type": s.get("source_type") or "Source type unknown", "Ref": s["ref"],
                           "Verification": s.get("verification_status") or "n/a"} for s in cited],
                         width="stretch", hide_index=True)
        st.markdown("#### Cited sources")
        st.markdown("\n".join(s["citation"] for s in cited) if cited else "No sources cited.")
        if other:
            st.markdown("#### Other retrieved records (not used as evidence)")
            st.markdown("\n".join(s["citation"] for s in other))
    with tabs[7]:
        v = result.validation
        cols = st.columns(4)
        cols[0].metric("Report validation", str(v.get("report_status")))
        cols[1].metric("Evidence accepted", v.get("evidence", {}).get("accepted", 0))
        cols[2].metric("Evidence rejected", v.get("evidence", {}).get("rejected", 0))
        cols[3].metric("Unresolved citations", v.get("citations", {}).get("unresolved", 0))
        st.markdown(f"**{svc.DETERMINISTIC_VALIDATION_NOTE}**  \n**{svc.SEMANTIC_ENTAILMENT_NOTE}**")
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
    auth_configured = _auth_configured()
    identity = _identity(auth_configured)
    quota_connection_configured = _quota_connection_configured(secrets)
    availability = svc.live_availability(secrets=secrets, identity=identity, auth_configured=auth_configured,
                                         quota_connection_configured=quota_connection_configured)

    question = st.text_area("Research question", value=DEMO_QUESTION, height=120, key="question",
                            help=f"{svc.MIN_QUESTION_CHARS}–{svc.MAX_QUESTION_CHARS} characters.")
    col_mode, col_sources = st.columns(2)
    with col_mode:
        mode_label = st.radio("Mode", list(MODES), index=0, horizontal=True, key="mode",
                              disabled=not availability.available)
        st.caption(svc.DEMO_MODE_DESCRIPTION)
        st.caption(svc.live_mode_description(svc.live_require_auth(secrets=secrets)))
        if not availability.available:
            st.caption(availability.message + " Demo Mode runs offline on synthetic data and needs no sign-in.")
            if availability.can_login and st.button("Log in for Live Mode", key="login"):
                st.login()
        else:
            st.caption("Live Mode is available to you. It is experimental and each run costs money.")
        if auth_configured and identity.is_logged_in and st.button("Log out", key="logout"):
            st.logout()
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
        st.caption(f"Demo Mode example investigation (synthetic, offline): {DEMO_TOPIC}.")

    if st.button("Investigate", type="primary", key="investigate"):
        request = svc.InvestigationRequest(question=question, from_year=from_year, to_year=to_year,
                                           max_sources=int(max_sources), mode=mode,
                                           identity=identity if mode == svc.MODE_LIVE else None)
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
                result = svc.run_web_investigation(request, progress=on_progress, secrets=secrets,
                                                   auth_configured=auth_configured,
                                                   quota_connection_factory=_quota_engine,
                                                   quota_connection_configured=quota_connection_configured)
            st.session_state["result"] = result

    result = st.session_state.get("result")
    if result is not None:
        if not result.ok:
            for error in result.errors:
                st.error(error)
        else:
            _render_result(result)


main()
