"""v0.3 M3: research-gap stage — deterministic validation of model gaps (offline)."""

from __future__ import annotations

import json

import pytest

from conftest import Sleeper
from m2_support import BIB, tracker
from m3_support import gap
from sciforge.llm.audit import ModelCallAudit
from sciforge.llm.budget import RetryPolicy
from sciforge.llm.fake import FakeModelClient
from sciforge.stages.common import CallContext
from sciforge.stages.gaps import GAP_FIELDS, run_gaps_stage, validate_gap
from sciforge.stages.synthesis_checks import model_evidence_view

R1, R2 = "rec_" + "a" * 16, "rec_" + "b" * 16
EVIDENCE = [
    {"evidence_id": "ev_0001", "source_record_id": R1, "claim": "High shear increases P-selectin.",
     "finding": "P-selectin expression rose by 40% versus static controls.", "quote": "by 40% compared",
     "evidence_category": "established", "confidence": "moderate", "access_level": "pubmed_abstract",
     "methods": "Exposed to 50 dyn/cm2 for 10 minutes."},
    {"evidence_id": "ev_0002", "source_record_id": R2, "claim": "Von Willebrand factor unfolds under flow.",
     "finding": None, "quote": "unfolds", "evidence_category": "conflicting", "confidence": "low",
     "access_level": "pubmed_abstract"},
]
BY_ID = {e["evidence_id"]: e for e in EVIDENCE}


def check(raw):
    return validate_gap(raw, evidence_by_id=BY_ID, known_sources={R1, R2}, next_id="gap_01")


def codes(raw):
    rec, reasons = check(raw)
    assert rec is None
    return [r["code"] for r in reasons]


def ctx(script):
    client = FakeModelClient(script)
    return CallContext(client=client, tracker=tracker(), audit=ModelCallAudit(), retry=RetryPolicy(), sleep=Sleeper()), client


def test_valid_gap_accepted_with_code_ids_and_label():
    rec, reasons = check(gap(conflicting_evidence_ids=["ev_0002"], gap_id="whatever the model said"))
    assert reasons == []
    assert rec["gap_id"] == "gap_01" and rec["label"] == "inference"
    assert rec["model_gap_id"]["redacted"] is True            # model id not echoed unless in opaque form
    assert rec["supporting_evidence_ids"] == ["ev_0001", "ev_0002"] and rec["conflicting_evidence_ids"] == ["ev_0002"]
    assert rec["source_record_ids"] == [R1, R2]
    assert rec["support"]["final_status"] == "accepted"


def test_gap_numbers_supported_by_referenced_evidence_only():
    assert check(gap(supporting_evidence_ids=["ev_0001"]))[1] == []
    # 40% lives in ev_0001's finding: citing only ev_0002 makes it unsupported ...
    c = codes(gap(supporting_evidence_ids=["ev_0002"], why_unresolved="Reported separately."))
    assert c == ["unsupported_claim"]
    # ... unless ev_0001 is referenced inline (inline ids count as references)
    assert check(gap(supporting_evidence_ids=["ev_0002"], why_unresolved="Reported in ev_0001 only."))[1] == []


@pytest.mark.parametrize("statement,subtype", [
    ("The 45% increase under shear lacks a mechanism.", "number_not_in_evidence"),
    ("A 40 mg dose effect is unexplained.", "unit_mismatch"),
    ("Effects in 12 cohorts (p < 0.05) are unexplained.", "number_not_in_evidence"),
])
def test_unsupported_gap_rejected(statement, subtype):
    rec, reasons = check(gap(gap_statement=statement))
    assert rec is None and reasons[0]["code"] == "unsupported_claim" and reasons[0]["subtype"] == subtype
    assert reasons[0]["field"] == "gap_statement"


def test_numbers_in_methods_do_not_support_gaps():
    # the model never sees methods, so 50 dyn/cm2 is not in the referenced evidence
    assert codes(gap(gap_statement="Whether 50 dyn/cm2 is a threshold is unknown.")) == ["unsupported_claim"]


def test_gap_with_unknown_evidence_id():
    rec, reasons = check(gap(supporting_evidence_ids=["ev_0001", "ev_0099"]))
    assert rec is None and [r["code"] for r in reasons] == ["unknown_evidence_id"]
    assert reasons[0]["value"] == "ev_0099"
    assert codes(gap(conflicting_evidence_ids=["ev_9"])) == ["unknown_evidence_id"]
    assert codes(gap(why_unresolved="See ev_0042.")) == ["unknown_evidence_id"]


def test_gap_missing_evidence():
    assert codes(gap(supporting_evidence_ids=[])) == ["missing_evidence_reference"]
    assert "missing_evidence_reference" in codes({k: v for k, v in gap().items() if k != "supporting_evidence_ids"})


def test_fabricated_source_id_in_gap_text():
    c = codes(gap(why_unresolved="Only rec_ffffffffffffffff examined it."))
    assert c == ["unknown_source_id"]
    assert check(gap(why_unresolved=f"Only {R1} examined it."))[1] == []


def test_fabricated_bibliographic_field_rejected_values_not_recorded():
    raw = gap(title="Invented Title Zeta", doi="10.1234/fake.99", authors=["Nobody Q"])
    rec, reasons = check(raw)
    assert rec is None and reasons[0]["code"] == "fabricated_bibliographic_field"
    assert reasons[0]["keys"] == ["authors", "doi", "title"]
    assert "Invented Title Zeta" not in json.dumps(reasons) and "10.1234" not in json.dumps(reasons)


@pytest.mark.parametrize("text,code,kind", [
    ("Resolved in doi 10.1234/abc.5 perhaps.", "identifier_in_text", "doi"),
    ("See https://example.org/paper for details.", "identifier_in_text", "url"),
    ("As PMID: 12345678 suggests.", "identifier_in_text", "pmid"),
    ("As Smith et al. noted, it is unclear.", "bibliographic_text", "et_al"),
    ("Unclear (Jones, 2019) and unresolved.", "bibliographic_text", "author_year"),
    ("Reported in the Journal of Thrombosis only.", "bibliographic_text", "journal_name"),
    ("Title: something unresolved.", "bibliographic_text", "field_label"),
])
def test_identifier_and_citation_patterns_in_text_rejected(text, code, kind):
    rec, reasons = check(gap(why_unresolved=text))
    assert rec is None and any(r["code"] == code and r["type"] == kind for r in reasons)


def test_invalid_confidence_and_unexpected_field():
    rec, reasons = check(gap(confidence="certain"))
    assert rec is None and reasons == [{"code": "invalid_confidence", "detail": "confidence must be one of "
                                        "['high', 'moderate', 'low']", "field": "confidence", "value": "certain"}]
    assert codes(gap(confidence="Totally sure, per the literature")) == ["invalid_confidence"]
    assert check(gap(confidence="Totally sure, per the literature"))[1][0]["value"] == "[redacted]"
    assert codes(gap(extra_note="x")) == ["unexpected_field"]
    assert codes("not an object") == ["malformed_item"]
    assert codes(gap(gap_statement="  ")) == ["empty_text"]


def test_stage_keeps_valid_rejects_invalid_and_sends_only_allowlisted_evidence():
    c, client = ctx([{"gaps": [gap(), gap(gap_statement="A 99% effect."), gap(supporting_evidence_ids=["ev_0002"],
                                                                                 gap_statement="Unfolding is unexplained.")]}])
    res = run_gaps_stage(c, {"research_question": "Q?"}, EVIDENCE, {R1, R2})
    assert res.status == "ok" and [g["gap_id"] for g in res.accepted] == ["gap_01", "gap_02"]
    assert len(res.rejected) == 1 and res.rejected[0]["reason_codes"] == ["unsupported_claim"]
    assert res.rejected[0]["item_index"] == 1 and res.rejected[0]["raw_output_stored"] is False
    assert "A 99% effect." not in json.dumps(res.rejected)
    assert res.redaction_plan and res.redaction_plan["audit_indices"] == [0]
    payload = json.loads(client.requests[0].messages[-1].content)
    assert set(payload) == {"question_definition", "evidence"}
    assert all(set(e) == {"evidence_id", "source_record_id", "claim", "finding", "evidence_category", "confidence"}
               for e in payload["evidence"])
    assert "quote" not in client.requests[0].messages[-1].content and "methods" not in json.dumps(payload["evidence"])
    for value in (BIB["title"], BIB["doi"], BIB["pmid"], BIB["journal"]):
        assert value not in json.dumps(client.requests[0].__dict__, default=str)


def test_model_evidence_view_redacts_identifiers():
    view = model_evidence_view([{**EVIDENCE[0], "claim": "See doi:10.1234/x.y for details."}])
    assert "10.1234" not in view[0]["claim"]


def test_structural_failure_repaired_then_failure_recorded():
    c, _ = ctx(["not json", {"wrong": []}])
    res = run_gaps_stage(c, {"research_question": "Q?"}, EVIDENCE, {R1, R2})
    assert res.status == "failed" and res.accepted == [] and res.call.repaired
    assert res.redaction_plan["audit_indices"] == [0, 1]


def test_gap_fields_constant_matches_spec():
    assert GAP_FIELDS == ("gap_id", "gap_statement", "supporting_evidence_ids", "conflicting_evidence_ids",
                          "why_unresolved", "confidence")
