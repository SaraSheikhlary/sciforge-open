"""v0.3 M3: candidate-hypothesis stage — deterministic validation (offline)."""

from __future__ import annotations

import json

import pytest

from conftest import Sleeper
from m2_support import tracker
from m3_support import hypothesis
from sciforge.llm.audit import ModelCallAudit
from sciforge.llm.budget import RetryPolicy
from sciforge.llm.fake import FakeModelClient
from sciforge.stages.common import CallContext
from sciforge.stages.hypotheses import run_hypotheses_stage, validate_hypothesis
from test_gaps_stage import BY_ID, EVIDENCE, R1, R2

GAPS = [{"gap_id": "gap_01", "gap_statement": "Mechanism untested.", "supporting_evidence_ids": ["ev_0001"],
         "conflicting_evidence_ids": [], "why_unresolved": "Separate reports.", "confidence": "moderate"}]
GAPS_BY_ID = {g["gap_id"]: g for g in GAPS}


def check(raw):
    return validate_hypothesis(raw, evidence_by_id=BY_ID, known_sources={R1, R2}, gaps_by_id=GAPS_BY_ID,
                               next_id="hyp_01")


def codes(raw):
    rec, reasons = check(raw)
    assert rec is None
    return [r["code"] for r in reasons]


def test_valid_hypothesis():
    rec, reasons = check(hypothesis(statement="A 40% rise may require GPIb binding."))
    assert reasons == []
    assert rec["hypothesis_id"] == "hyp_01" and rec["label"] == "hypothesis"
    assert rec["research_gap_ids"] == ["gap_01"] and rec["supporting_evidence_ids"] == ["ev_0001", "ev_0002"]
    assert rec["assumptions"] == ["GPIb binding precedes P-selectin exposure."]
    assert rec["support"]["final_status"] == "accepted" and rec["source_record_ids"] == [R1, R2]


def test_hypothesis_without_evidence():
    assert codes(hypothesis(supporting_evidence_ids=[])) == ["hypothesis_missing_evidence"]
    assert codes(hypothesis(supporting_evidence_ids=["ev_0404"])) == ["unknown_evidence_id"]


def test_hypothesis_without_gap():
    assert codes(hypothesis(research_gap_ids=[])) == ["hypothesis_missing_gap"]
    rec, reasons = check(hypothesis(research_gap_ids=["gap_07"]))
    assert rec is None and reasons[0]["code"] == "unknown_gap_id" and reasons[0]["value"] == "gap_07"
    assert codes(hypothesis(rationale="Follows from gap_09.")) == ["unknown_gap_id"]


@pytest.mark.parametrize("label", ["established", "Hypothesis", "", None, "theory"])
def test_label_enforced(label):
    assert codes(hypothesis(label=label)) == ["invalid_label"]


@pytest.mark.parametrize("statement", [
    "It is established that GPIb blockade prevents activation.",
    "It has been proven that shear activates platelets.",
    "It is well-known that GPIb mediates this.",
    "This demonstrates that GPIb is required.",
    "GPIb blockade definitively prevents activation.",
    "Evidence clearly shows GPIb dependence.",
])
def test_statement_phrased_as_fact_rejected(statement):
    assert codes(hypothesis(statement=statement)) == ["hypothesis_asserted_as_fact"]


def test_same_bibliographic_identifier_and_number_checks():
    assert codes(hypothesis(doi="10.1/x"))[0] == "fabricated_bibliographic_field"
    assert codes(hypothesis(rationale="Per Smith et al., likely.")) == ["bibliographic_text"]
    assert codes(hypothesis(predicted_observable_outcome="See https://x.org/y.")) == ["identifier_in_text"]
    assert codes(hypothesis(predicted_observable_outcome="A 75% reduction in P-selectin.")) == ["unsupported_claim"]
    assert codes(hypothesis(assumptions=["Donors aged 18 or older."])) == ["unsupported_claim"]
    assert codes(hypothesis(statement="rec_0000000000000000 suggests GPIb may matter.")) == ["unknown_source_id"]
    assert codes(hypothesis(assumptions="not a list")) == ["schema_violation"]
    assert codes(hypothesis(confidence="very_high")) == ["invalid_confidence"]


def test_stage_payload_and_mixed_batch():
    client = FakeModelClient([{"hypotheses": [hypothesis(), hypothesis(research_gap_ids=[]),
                                              hypothesis(statement="GPIb may matter too.")]}])
    ctx = CallContext(client=client, tracker=tracker(), audit=ModelCallAudit(), retry=RetryPolicy(), sleep=Sleeper())
    res = run_hypotheses_stage(ctx, {"research_question": "Q?"}, EVIDENCE, GAPS, {R1, R2})
    assert [h["hypothesis_id"] for h in res.accepted] == ["hyp_01", "hyp_02"]
    assert res.rejected[0]["reason_codes"] == ["hypothesis_missing_gap"]
    payload = json.loads(client.requests[0].messages[-1].content)
    assert set(payload) == {"question_definition", "evidence", "research_gaps"}
    assert payload["research_gaps"][0]["gap_id"] == "gap_01"
