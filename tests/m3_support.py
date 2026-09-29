"""Shared builders for the v0.3 Milestone 3 tests (offline only; FakeModelClient + httpx.MockTransport)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from conftest import Sleeper, mock_client
from m2_support import BIB, PLAIN_ABSTRACT_TEXT, STRUCTURED_ABSTRACT_XML, Api, item, no_throttle, question_output, \
    record, tracker, verified
from sciforge.investigation_pipeline import run_model_investigation
from sciforge.llm.fake import FakeModelClient

FIXED = datetime(2026, 9, 28, 12, 0, 0, tzinfo=timezone.utc)
QUESTION = "Does high shear stress directly activate human platelets?"
QUOTE_1 = "Shear exposure increased P-selectin expression by 40% compared with static controls."
QUOTE_2 = "Von Willebrand factor unfolds under elongational flow"

BIB2 = {"title": "Elongational Flow Unfolding of Ultralarge Multimers", "authors": ["Isolde Marchetti-Oyelaran"],
        "journal": "Quarterly Review of Imaginary Vascular Mechanics", "year": 1993,
        "doi": "10.5555/qrivm.1993.777", "pmid": "27182818"}


def records():
    r1 = record()
    r2 = record(pmid=BIB2["pmid"], doi=BIB2["doi"], title=BIB2["title"], authors=BIB2["authors"],
                journal=BIB2["journal"], year=BIB2["year"])
    return [r1, r2], [verified(r1), verified(r2)]


def api() -> Api:
    return Api(pubmed={BIB["pmid"]: STRUCTURED_ABSTRACT_XML,
                       BIB2["pmid"]: f"<AbstractText>{PLAIN_ABSTRACT_TEXT}</AbstractText>"})


def evidence_items(recs) -> list[dict[str, Any]]:
    r1, r2 = recs
    return [{"items": [item(r1.record_id, QUOTE_1)]},
            {"items": [item(r2.record_id, QUOTE_2, claim="Von Willebrand factor unfolds under elongational flow.",
                            finding="Unfolded factor binds platelet GPIb more strongly.", methods=None,
                            limitations="Abstract only.", relevance="Mechanism of shear sensing.")]}]


def gap(**overrides: Any) -> dict[str, Any]:
    base = {
        "gap_id": "gap_01",
        "gap_statement": "Whether the 40% P-selectin increase under shear depends on GPIb binding is untested.",
        "supporting_evidence_ids": ["ev_0001", "ev_0002"],
        "conflicting_evidence_ids": [],
        "why_unresolved": "The abstracts report the shear effects separately (ev_0001, ev_0002).",
        "confidence": "moderate",
    }
    base.update(overrides)
    return base


def falsification(**overrides: Any) -> dict[str, Any]:
    base = {"manipulated_or_compared": "Platelets under high shear with versus without GPIb blockade.",
            "measured": "P-selectin expression on the platelet surface.",
            "weakening_result": "P-selectin expression is unchanged by GPIb blockade under shear.",
            "supporting_result": "P-selectin expression falls when GPIb is blocked during shear exposure."}
    base.update(overrides)
    return base


def alternative(**overrides: Any) -> dict[str, Any]:
    base = {"explanation": "Shear might activate platelets through a GPIb-independent route, such as membrane "
                           "stretch.",
            "basis": "inference", "evidence_ids": []}
    base.update(overrides)
    return base


def hypothesis(**overrides: Any) -> dict[str, Any]:
    """A v0.4 model hypothesis that passes every deterministic check against the M3 fixtures."""
    base = {
        "hypothesis_id": "h1",
        "hypothesis": "Blocking GPIb may reduce the shear-induced P-selectin increase.",
        "evidence_ids": ["ev_0001", "ev_0002"],
        "research_gap_id": "gap_01",
        "mechanistic_claim_level": "mechanistic_support",
        "rationale": "Both effects occur under high shear, and unfolded factor binds GPIb more strongly.",
        "prediction": "P-selectin expression under shear is lower when GPIb is blocked than without blockade.",
        "alternative_explanation": alternative(),
        "falsification_test": falsification(),
        "assumptions": ["GPIb binding precedes P-selectin exposure."],
        "evidence_limitations": ["Both items come from single abstracts."],
        "confidence": "low",
    }
    base.update(overrides)
    return base


CRITIC_CHECKS = ("evidence_supports_mechanism", "causal_language_exceeds_evidence", "distinct_from_evidence",
                 "prediction_measurable", "prediction_discriminates", "falsification_meaningful",
                 "ignored_contradictions_or_missing_evidence", "confidence_consistent")


def critic_review(hid: str, *, fail: tuple[str, ...] = (), uncertain: tuple[str, ...] = (),
                  problems: tuple[str, ...] = (), unsupported: tuple[str, ...] = (),
                  explanation: str = "Looks consistent with the evidence.") -> dict[str, Any]:
    checks = {}
    for name in CRITIC_CHECKS:
        verdict = "fail" if name in fail else "uncertain" if name in uncertain else "pass"
        checks[name] = {"verdict": verdict, "explanation": explanation if verdict == "pass"
                        else f"Problem with {name.replace('_', ' ')}."}
    return {"hypothesis_id": hid, "checks": checks, "unsupported_evidence_ids": list(unsupported),
            "substantive_problems": list(problems)}


def candidate_ids(request) -> list[str]:
    payload = json.loads(request.messages[-1].content)
    return [c["hypothesis_id"] for c in payload.get("candidate_hypotheses", [])]


def auto_critic(request) -> dict[str, Any]:
    """Critic responder that passes every candidate named in the request."""
    return {"reviews": [critic_review(hid) for hid in candidate_ids(request)]}


def paragraph(**overrides: Any) -> dict[str, Any]:
    base = {"section": "key_findings", "label": "established",
            "text": "High shear increased P-selectin expression by 40% [ev_0001].",
            "evidence_ids": ["ev_0001"], "source_ids": []}
    base.update(overrides)
    return base


def narrative(*extra: dict[str, Any]) -> dict[str, Any]:
    return {"paragraphs": [
        paragraph(),
        paragraph(section="key_findings", label="inference",
                  text="Elongational flow may couple factor unfolding to platelet capture [ev_0002].",
                  evidence_ids=["ev_0002"]),
        paragraph(section="limitations", label="inference", text="Both findings rest on single abstracts.",
                  evidence_ids=[]),
        paragraph(section="next_steps", label="hypothesis", text="Test hyp_01 in a flow chamber with GPIb blockade.",
                  evidence_ids=[]),
        *extra]}


def full_script(recs, *, gaps=None, hyps=None, critic=None, revision=None, narr=None) -> list[Any]:
    """Scripted outputs in pipeline order. ``critic=False`` omits the critic call (e.g. no candidates);
    ``revision`` is only appended when given (the revision call only happens when something needs revising)."""
    script = [question_output(), *evidence_items(recs),
              gaps if gaps is not None else {"gaps": [gap()]},
              hyps if hyps is not None else {"hypotheses": [hypothesis()]}]
    if critic is not False:
        script.append(critic if critic is not None else auto_critic)
    if revision is not None:
        script.append(revision)
    script.append(narr if narr is not None else narrative())
    return script


def run(tmp_path, settings, script, *, recs=None, ver=None, **kw):
    if recs is None:
        recs, ver = records()
    client = FakeModelClient(script)
    kw.setdefault("tracker", tracker())
    kw.setdefault("search_summary", SEARCH_SUMMARY)
    result = run_model_investigation(QUESTION, recs, ver, model_client=client, settings=settings,
                                     output_dir=tmp_path, http_client=mock_client(kw.pop("api", None) or api()),
                                     pubmed_limiter=no_throttle(), crossref_limiter=no_throttle(), sleep=Sleeper(),
                                     now=lambda: FIXED, **kw)
    return result, client


def load(result, name):
    return json.loads(result.files[name].read_text(encoding="utf-8"))


SEARCH_SUMMARY = {
    "query_used": QUESTION,
    "query_generation": "none: v0.2 uses the research question verbatim as the search query",
    "parameters": {"from_year": 1980, "to_year": 2026, "max_results_per_source": 20},
    "started_at": "2026-09-28T11:59:00Z", "finished_at": "2026-09-28T11:59:30Z",
    "databases_queried": ["pubmed", "crossref"],
    "search_status": {"pubmed": "ok", "crossref": "ok"},
    "total_hits_reported": {"pubmed": 42, "crossref": 1234},
    "retrieved_per_source": {"pubmed": 2, "crossref": 1},
    "unique_records": 2,
    "verification_counts": {"verified": 2, "partially_verified": 0, "not_verified": 0},
}
