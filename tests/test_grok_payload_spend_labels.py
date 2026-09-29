"""grok-4.7 request payloads per stage, reported-cost spend guard, source-type labels and UI/report wording."""

from __future__ import annotations

import json
import re
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from conftest import FAKE_XAI_KEY, Sleeper, json_response, mock_client, responses_payload
from m2_support import no_throttle, record, verified
from m3_support import FIXED, QUESTION, SEARCH_SUMMARY, api, critic_review, full_script, records
from sciforge import app_service as svc
from sciforge.config import ModelSettings, Settings
from sciforge.demo_data import DEMO_QUESTION
from sciforge.investigation_pipeline import run_model_investigation
from sciforge.llm.budget import BudgetLimits, BudgetTracker, PriceTable, budgeted_call
from sciforge.llm.client import BudgetExhausted, ModelMessage, ModelRequest, ModelUsage
from sciforge.llm.fake import FakeModelClient
from sciforge.llm.parsing import parse_usage
from sciforge.llm.xai import XAIClient, build_request_body
from sciforge.stages.report import (
    HYPOTHESIS_DISCLAIMER,
    PREPRINT_BADGE,
    build_report,
    render_citation,
    source_type_badge,
)

REPO = Path(__file__).resolve().parents[1]
STAGE_ENV = {
    "XAI_API_KEY": FAKE_XAI_KEY, "XAI_MODEL": "grok-4.7", "SCIFORGE_MAX_SPEND_USD": "none",
    "SCIFORGE_MODEL_REASONING_EFFORT": "high", "SCIFORGE_MODEL_REASONING_EFFORT_EVIDENCE": "medium",
    "SCIFORGE_MODEL_REASONING_EFFORT_GAPS": "low", "SCIFORGE_MODEL_REASONING_EFFORT_HYPOTHESES": "xhigh",
    "SCIFORGE_MODEL_REASONING_EFFORT_REPORT": "medium", "SCIFORGE_MODEL_REASONING_EFFORT_HYPOTHESIS_CRITIC": "high",
    "SCIFORGE_MODEL_MAX_OUTPUT_TOKENS_HYPOTHESIS_CRITIC": "4000",
    "SCIFORGE_MODEL_MAX_OUTPUT_TOKENS_QUESTION": "1500", "SCIFORGE_MODEL_MAX_OUTPUT_TOKENS_EVIDENCE": "2500",
    "SCIFORGE_MODEL_MAX_OUTPUT_TOKENS_GAPS": "3000", "SCIFORGE_MODEL_MAX_OUTPUT_TOKENS_HYPOTHESES": "6000",
    "SCIFORGE_MODEL_MAX_OUTPUT_TOKENS_REPORT": "3500",
}
USAGE = {"input_tokens": 900, "output_tokens": 200, "total_tokens": 1500,
         "output_tokens_details": {"reasoning_tokens": 400}, "cost_in_usd_ticks": 50_000_000}   # $0.005


# ------------------------------------------------------------------ grok-4.7 payload per stage


def test_grok_47_payload_per_stage_end_to_end(tmp_path):
    """Full M3 pipeline through the real XAIClient over MockTransport: exact body per stage, store=false."""
    ms = ModelSettings.from_env(STAGE_ENV)
    recs, ver = records()
    script = list(full_script(recs, critic={"reviews": [critic_review("hyp_01")]}))
    bodies: list[dict] = []

    def xai(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/responses"
        bodies.append(json.loads(request.content))
        return json_response(responses_payload(json.dumps(script.pop(0)), usage=USAGE, model="grok-4.7"))

    client = XAIClient(ms, http=mock_client(xai))
    res = run_model_investigation(QUESTION, recs, ver, model_client=client, settings=Settings(), model_settings=ms,
                                  search_summary=SEARCH_SUMMARY, output_dir=tmp_path, http_client=mock_client(api()),
                                  pubmed_limiter=no_throttle(), crossref_limiter=no_throttle(), sleep=Sleeper(),
                                  now=lambda: FIXED)
    assert script == [] and len(bodies) == 7
    expected = [("high", 1500), ("medium", 2500), ("medium", 2500), ("low", 3000), ("xhigh", 6000),
                ("high", 4000), ("medium", 3500)]       # question, evidence x2, gaps, hypotheses, critic, report
    for body, (effort, cap) in zip(bodies, expected):
        assert body["model"] == "grok-4.7" and body["store"] is False
        assert body["reasoning"] == {"effort": effort} and body["max_output_tokens"] == cap
        assert "previous_response_id" not in body and FAKE_XAI_KEY not in json.dumps(body)
    # actual cost: cost_in_usd_ticks is the recorded spend for every attempt (7 x $0.005)
    used = res.budget["used"]
    assert Decimal(used["spend_usd"]) == Decimal("0.035") and Decimal(used["reported_cost_usd"]) == Decimal("0.035")
    assert used["reasoning_tokens"] == 7 * 400 and used["output_side_tokens_charged"] == 7 * 600


def test_request_body_unit_for_every_stage():
    ms = ModelSettings.from_env(STAGE_ENV)
    for stage in ("question", "evidence", "gaps", "hypotheses", "report"):
        req = ModelRequest(messages=(ModelMessage("user", "x"),), max_output_tokens=ms.max_output_tokens_for(stage),
                           stage=stage, reasoning_effort=ms.reasoning_effort_for(stage))
        body = build_request_body("grok-4.7", req)
        assert body["store"] is False and body["reasoning"] == {"effort": ms.reasoning_effort_for(stage)}
        assert body["max_output_tokens"] == ms.max_output_tokens_for(stage)


# ------------------------------------------------------------------ accounting + spend guard


def test_output_side_tokens_are_output_plus_reasoning():
    u = parse_usage({"usage": USAGE})
    assert u.billable_output_tokens == 600 and u.cost_usd_reported == Decimal("0.005")
    assert ModelUsage(input_tokens=10, output_tokens=5).billable_output_tokens == 5


def test_reported_cost_accumulates_and_blocks_the_next_call_when_remaining_budget_is_insufficient():
    prices = PriceTable(input_per_mtok=3.0, output_per_mtok=60.0)
    tracker = BudgetTracker(BudgetLimits(max_spend_usd=2), prices)
    req = ModelRequest(messages=(ModelMessage("user", "x" * 3000),), max_output_tokens=2000, stage="evidence")
    ticks = [15_000_000_000, 4_000_000_000]                     # $1.50 then $0.40 reported by xAI
    client = FakeModelClient([{"ok": 1}, {"ok": 2}, {"ok": 3}])
    usages = iter(ticks)

    class Reporting(FakeModelClient):
        def complete(self, request):
            resp = client.complete(request)
            from dataclasses import replace
            return replace(resp, usage=parse_usage({"usage": {"input_tokens": 1000, "output_tokens": 100,
                                                              "output_tokens_details": {"reasoning_tokens": 50},
                                                              "cost_in_usd_ticks": next(usages)}}))

    reporting = Reporting([])
    budgeted_call(reporting, req, tracker, sleep=lambda s: None)
    budgeted_call(reporting, req, tracker, sleep=lambda s: None)
    s = tracker.summary()
    assert Decimal(s["used"]["spend_usd"]) == Decimal("1.9") and s["used"]["reported_cost_attempts"] == 2
    # worst case of one more attempt: ~1000 input tok * $3/M + 2000 output tok * $60/M = ~$0.123 > remaining $0.10
    with pytest.raises(BudgetExhausted) as info:
        budgeted_call(reporting, req, tracker, sleep=lambda s: None)
    assert info.value.limit == "max_spend_usd" and tracker.attempts == 2           # no new call was sent


def test_reported_cost_reaching_the_cap_stops_all_further_calls():
    tracker = BudgetTracker(BudgetLimits(max_spend_usd=2), PriceTable(input_per_mtok=1.0, output_per_mtok=1.0))
    req = ModelRequest(messages=(ModelMessage("user", "x"),), max_output_tokens=100)
    r = tracker.reserve(req)
    tracker.record(r, ModelUsage(input_tokens=10, output_tokens=10, reasoning_tokens=5,
                                 cost_usd_reported=Decimal("2.0"), cost_source="reported_ticks"))
    assert tracker.exhausted_by == "max_spend_usd"
    with pytest.raises(BudgetExhausted):
        tracker.reserve(req)


# ------------------------------------------------------------------ source-type labels


def test_badges_and_citation_order():
    assert source_type_badge("preprint") == PREPRINT_BADGE == "Preprint — not peer-reviewed"
    assert source_type_badge(None) == source_type_badge("weird") == "Source type unknown"
    rec = record()
    pre = render_citation("S1", rec, verified(rec), None, "preprint")
    assert pre.startswith("- **[S1]** **[Preprint — not peer-reviewed]** ")
    assert "journal article" not in pre.lower()
    art = render_citation("S2", rec, verified(rec), None, "peer-reviewed journal article")
    assert art.startswith("- **[S2]** **[Peer-reviewed journal article (metadata label)]** ")
    assert "not a guarantee of peer review" in art


def _report(statuses):
    recs, ver = records()
    ev = [{"evidence_id": "ev_0001", "claim": "c", "quote": "q", "evidence_category": "established",
           "confidence": "moderate", "access_level": "pubmed_abstract", "source_record_id": recs[0].record_id},
          {"evidence_id": "ev_0002", "claim": "d", "quote": "r", "evidence_category": "inference",
           "confidence": "low", "access_level": "pubmed_abstract", "source_record_id": recs[1].record_id}]
    summary = {**SEARCH_SUMMARY, "source_classification": {recs[0].record_id: statuses[0],
                                                           recs[1].record_id: statuses[1]}}
    report, _ = build_report(question=QUESTION, question_definition=None, search_summary=summary,
                             source_texts={"sources": []}, evidence={"accepted": ev}, gaps=None, hypotheses=None,
                             narrative=None, records=recs, verification=ver, stage_notes=[])
    return report


def test_report_source_type_column_and_references():
    report = _report(["preprint", "peer-reviewed journal article"])
    d = report.split("## D.")[1].split("## E.")[0]
    assert "| Source | Source type |" in d
    assert "| [S1] | **Preprint — not peer-reviewed** |" in d
    assert "| [S2] | **Peer-reviewed journal article (metadata label)** |" in d
    j = report.split("## J.")[1]
    assert "**Preprint — not peer-reviewed** 1" in j
    lines = [ln for ln in j.splitlines() if ln.startswith("- **[S")]
    assert lines[0].startswith("- **[S1]** **[Preprint — not peer-reviewed]**")
    assert lines[1].startswith("- **[S2]** **[Peer-reviewed journal article (metadata label)]**")
    # a preprint is never presented as a peer-reviewed journal article
    assert "Peer-reviewed" not in lines[0]


def test_report_hypotheses_are_labelled_unvalidated():
    report = _report(["unknown", "unknown"])
    h = report.split("## H.")[1].split("## I.")[0]
    assert HYPOTHESIS_DISCLAIMER in h and "not validated discoveries" in h
    assert "No candidate hypotheses passed the deterministic checks." in h


# ------------------------------------------------------------------ wording


CLAIMS = re.compile(r"(?i)(?<!not )(?<!not a )validated discover|proven hypothes|confirmed discover|"
                    r"semantic(?:ally)? (?:claim )?(?:entailment|check) (?:passed|verified|on\b)")


def test_service_wording_constants():
    assert "no real literature search and no xAI calls" in svc.DEMO_MODE_DESCRIPTION
    assert "synthetic" in svc.DEMO_MODE_DESCRIPTION and "offline" in svc.DEMO_MODE_DESCRIPTION
    live = svc.live_mode_description(True)
    for phrase in ("real literature retrieval", "real xAI model calls", "costs money", "Sign-in"):
        assert phrase in live
    assert "disabled" in svc.live_mode_description(False)
    assert svc.PREPRINT_NOTE == "Preprint: not peer-reviewed."
    assert svc.DETERMINISTIC_VALIDATION_NOTE == "Deterministic validation: exact quote, numeric and citation checks."
    assert svc.SEMANTIC_ENTAILMENT_NOTE == "Semantic claim entailment: not implemented."
    assert HYPOTHESIS_DISCLAIMER.startswith("Unvalidated, AI-generated hypotheses for further investigation")


def test_demo_result_labels_hypotheses_and_never_claims_discovery():
    r = svc.run_web_investigation(svc.InvestigationRequest(question=DEMO_QUESTION), environ={})
    assert r.ok and all(h["status"] == HYPOTHESIS_DISCLAIMER for h in r.hypotheses)
    assert HYPOTHESIS_DISCLAIMER in r.sections["H. Candidate Hypotheses"]
    assert all(e["source_type"] == "Source type unknown" for e in r.evidence)
    assert not CLAIMS.search(r.displayed_text())


def _app(monkeypatch):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(REPO / "streamlit_app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    return at


def _text(at):
    parts = []
    for kind in ("caption", "markdown", "info", "warning", "error", "subheader", "title"):
        parts += [str(e.value) for e in getattr(at, kind, [])]
    for df in at.dataframe:
        parts.append(df.value.to_csv())
    return "\n".join(parts)


def test_ui_wording_before_and_after_a_demo_run(monkeypatch):
    pytest.importorskip("streamlit")
    at = _app(monkeypatch)
    before = _text(at)
    assert svc.DEMO_MODE_DESCRIPTION in before and "real xAI model calls" in before and "costs money" in before
    at.button(key="investigate").click().run()
    after = _text(at)
    for phrase in (svc.DETERMINISTIC_VALIDATION_NOTE, svc.SEMANTIC_ENTAILMENT_NOTE, svc.PREPRINT_NOTE,
                   HYPOTHESIS_DISCLAIMER, "Source type"):
        assert phrase in after, phrase
    assert not CLAIMS.search(after)


@pytest.mark.parametrize("path", ["streamlit_app.py", "README.md", "docs/web-app.md", "app/sciforge/app_service.py",
                                  "app/sciforge/stages/report.py", "app/sciforge/demo_data.py"])
def test_no_discovery_or_semantic_claims_in_sources(path):
    assert not CLAIMS.search((REPO / path).read_text(encoding="utf-8"))
