"""Truthful claim-check reporting and stale-wording regressions (offline only)."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from conftest import api_handler, mock_client
from sciforge import app_service as svc
from sciforge.cli import format_summary
from sciforge.config import ModelSettings
from sciforge.demo_data import DEMO_QUESTION
from sciforge.pipeline import run_investigation
from sciforge.stages.claim_checks import SEMANTIC_STATUS, claim_check_lines, claim_check_summary
from sciforge.stages.report import build_report

FIXED = datetime(2026, 9, 28, 23, 41, 0, tzinfo=timezone.utc)
Q = "What is the role of lipid-related changes in shear-mediated platelet activation?"
REPO = Path(__file__).resolve().parents[1]

# Wording that would claim (or imply) that a semantic / model-based claim check ran.
SEMANTIC_CLAIMS = re.compile(
    r"(?i)(semantic(?: \(model-based\))? claim check(?:ing)?\s*(?::|is|was)?\s*(?:on|enabled|passed|ran|done|complete)"
    r"|entailment check (?:passed|ran|on|enabled)|claims? (?:were|was) semantically (?:verified|checked)"
    r"|model entailment check on|claim checking is on)")
NEGATIONS = ("no semantic claim check ran", "no semantic check ran")


def claims_semantic(text: str) -> bool:
    low = text.lower()
    for phrase in NEGATIONS:
        low = low.replace(phrase, "")
    return bool(SEMANTIC_CLAIMS.search(low))


STALE = ("max results per source", "Max results per source", "searched verbatim only",
         "searches the question verbatim only", "only the question verbatim is searched",
         "Claim support is deterministic only (no model entailment check", "does not read abstracts",
         "scored from retrieved titles and query provenance only")


def test_claim_check_summary_counts_by_check_family():
    rejected = [{"reason_codes": ["quote_not_in_source"]}, {"reason_codes": ["number_not_in_source"]},
                {"reason_codes": ["unknown_record_id"]}, {"reason_codes": ["malformed_item"]}]
    s = claim_check_summary(evidence_accepted=5, evidence_rejected=rejected, synthesis_accepted=2,
                            synthesis_rejected=[{"reason_codes": ["unit_mismatch"]}], citations_resolved=4,
                            citations_unresolved=1)
    det = s["deterministic_checks"]
    assert det["exact_quote"] == {"checked": 8, "passed": 7, "failed": 1, "scope": "evidence items", "ran": True}
    assert (det["numeric_consistency"]["checked"], det["numeric_consistency"]["failed"]) == (11, 2)
    assert det["citation_validation"]["failed"] == 1 and det["citation_validation"]["citations_unresolved"] == 1
    assert s["semantic_claim_check"]["ran"] is False and s["semantic_claim_check"]["status"] == SEMANTIC_STATUS
    lines = claim_check_lines(s)
    assert lines[-1].startswith("Semantic (model-based) claim check: not implemented")
    assert not any(claims_semantic(line) for line in lines)


def test_claim_check_lines_never_invent_results_when_nothing_ran():
    lines = claim_check_lines(claim_check_summary(evidence_accepted=0, evidence_rejected=[]))
    assert all("not run (no items to check)" in line for line in lines[:2])
    assert "not implemented" in lines[-1]


def test_entailment_flag_is_reserved_and_off():
    assert ModelSettings(api_key="k" * 20, model="m", max_spend_usd=None).entailment is False


def _pipeline_summary(tmp_path, **kw):
    return run_investigation(Q, output_dir=tmp_path / "runs", client=mock_client(api_handler),
                             sleep=lambda s: None, now=lambda: FIXED, **kw).summary


def _fallback_report(summary):
    return build_report(question=Q, question_definition=None, search_summary=summary, source_texts={"sources": []},
                        evidence={"accepted": [], "rejected": []}, gaps=None, hypotheses=None, narrative=None,
                        records=[], verification=[], stage_notes=[])


def test_fallback_report_is_truthful_about_claim_checks_and_retrieval(tmp_path, settings):
    summary = run_investigation(Q, output_dir=tmp_path / "runs", settings=settings, client=mock_client(api_handler),
                                sleep=lambda s: None, now=lambda: FIXED).summary
    report, validation = _fallback_report(summary)
    assert "Semantic (model-based) claim check: not implemented" in report
    assert validation["claim_checks"]["semantic_claim_check"]["ran"] is False
    assert not claims_semantic(report)
    for phrase in STALE:
        assert phrase not in report, phrase
    b = report.split("## B.")[1].split("## C.")[0]
    for phrase in ("Query expansion:", "Candidate pool:", "Abstract enrichment", "Selection:", "Source policy:"):
        assert phrase in b, phrase
    cli = format_summary(summary, "runs/x")
    assert "Source policy: peer_reviewed_preferred" in cli
    for phrase in STALE:
        assert phrase not in cli


def test_web_demo_outputs_never_claim_a_semantic_check():
    result = svc.run_web_investigation(svc.InvestigationRequest(question=DEMO_QUESTION), environ={})
    assert result.ok
    text = result.displayed_text()
    assert not claims_semantic(text)
    for phrase in STALE:
        assert phrase not in text, phrase
    assert result.validation["claim_checks"]["semantic_claim_check"] == {
        "status": "not implemented", "ran": False,
        "note": "Semantic (model-based) claim checking is not implemented in this build; no semantic check ran. "
                "Only the deterministic checks listed here were applied."}
    det = result.validation["claim_checks"]["deterministic_checks"]
    assert det["exact_quote"]["ran"] and det["exact_quote"]["failed"] == 1       # the demo's rejected quote
    assert any("semantic (model-based) claim checking is not implemented" in item for item in result.limitations)
    # Demo records carry no bibliographic type metadata: labelled unknown, never invented
    assert {s["source_status"] for s in result.sources if s["cited"]} == {"unknown"}


def test_progress_messages_describe_deterministic_checks_only():
    events = []
    svc.run_web_investigation(svc.InvestigationRequest(question=DEMO_QUESTION), environ={},
                              progress=events.append)
    labels = dict(svc.PROGRESS_STAGES)
    assert labels["check"] == "Checking evidence (deterministic checks)"
    check = [e.detail for e in events if e.stage == "check" and e.state == "done"]
    assert check and "semantic claim check: not implemented" in check[-1]
    for e in events:
        assert not claims_semantic(e.detail)
        for phrase in STALE:
            assert phrase not in e.detail


@pytest.mark.parametrize("path", ["README.md", "docs/web-app.md", "docs/v0.2-retrieval-engine.md",
                                  "docs/v0.3-model-layer.md", "docs/product_spec.md", ".env.example",
                                  "streamlit_app.py", "app/sciforge/cli.py", "SECURITY.md"])
def test_docs_and_ui_have_no_stale_or_semantic_claims(path):
    text = (REPO / path).read_text(encoding="utf-8")
    assert not claims_semantic(text), path
    for phrase in STALE:
        assert phrase not in text, (path, phrase)
    assert "Model entailment check of extracted claims (default true)" not in text
    assert "D4** Model entailment check on by default" not in text
