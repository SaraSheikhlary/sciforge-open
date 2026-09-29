"""v0.3 M2 precedence contract: deterministic validation precedes and overrides model support labels."""

from __future__ import annotations

import pytest

from m2_support import STRUCTURED_ABSTRACT_TEXT, item, record
from sciforge.stages.validation import (
    DETERMINISTIC_FAILURE_CODES, MODEL_SUPPORT_LABELS, DeterministicResult, combine_support, validate_item,
)

REC = record()
QUOTE = "Shear exposure increased P-selectin expression by 40% compared with static controls."

FAILURES = {
    "unknown_record_id": item("rec_deadbeefdeadbeef", QUOTE),
    "quote_not_in_source": item(REC.record_id, QUOTE.replace("increased", "raised")),
    "bibliographic_field": item(REC.record_id, QUOTE, doi="10.9999/x"),
    "unsupported_evidence_category": item(REC.record_id, QUOTE, evidence_category="proven"),
    "number_not_in_quote": item(REC.record_id, QUOTE, claim="P-selectin rose by 45%."),
    "unit_mismatch": item(REC.record_id, QUOTE, claim="P-selectin rose by 40 mg."),
}


def deterministic(raw) -> DeterministicResult:
    _, reasons, _ = validate_item(raw, request_texts={REC.record_id: STRUCTURED_ABSTRACT_TEXT},
                                  supplied_ids={REC.record_id})
    return DeterministicResult(passed=not reasons, reasons=tuple(r["code"] for r in reasons))


@pytest.mark.parametrize("code", sorted(FAILURES))
@pytest.mark.parametrize("label", ["supported", "partially_supported", None, "bogus"])
def test_deterministic_failure_is_final_whatever_the_model_says(code, label):
    det = deterministic(FAILURES[code])
    assert not det.passed and code in det.reasons
    out = combine_support(det, label)
    assert out["final_status"] == "rejected" and out["final_label"] == "rejected_deterministic"
    assert out["decided_by"] == "deterministic" and out["model_label_applied"] is False


def test_passed_item_without_model_label():
    det = deterministic(item(REC.record_id, QUOTE))
    assert det.passed
    assert combine_support(det)["final_label"] == "passed_deterministic"
    assert combine_support(det)["final_status"] == "accepted"


def test_supported_keeps_passed_item_accepted():
    out = combine_support(DeterministicResult(passed=True), "supported")
    assert (out["final_status"], out["final_label"], out["model_label_applied"]) == ("accepted", "supported", True)


@pytest.mark.parametrize("label", ["partially_supported", "unverifiable", "not_supported"])
def test_model_can_only_downgrade(label):
    out = combine_support(DeterministicResult(passed=True), label)
    assert out["final_status"] == "downgraded" and out["final_label"] == label


def test_unknown_model_label_downgrades_never_upgrades():
    out = combine_support(DeterministicResult(passed=True), "definitely_true")
    assert out["final_status"] == "downgraded" and out["final_label"] == "unverifiable"
    assert out["model_label"] == "invalid"


def test_contract_invariants():
    with pytest.raises(ValueError):
        DeterministicResult(passed=True, reasons=("quote_not_in_source",))
    with pytest.raises(ValueError):
        DeterministicResult(passed=False)
    assert MODEL_SUPPORT_LABELS[0] == "supported"
    assert set(FAILURES) <= DETERMINISTIC_FAILURE_CODES
    for label in (*MODEL_SUPPORT_LABELS, None):
        assert combine_support(DeterministicResult(False, ("empty_claim",)), label)["final_status"] == "rejected"
