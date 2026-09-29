"""v0.4 hypothesis engine: deterministic validation + generation -> critic -> revision stage (offline).

Replaces the v0.3 M3 hypothesis-stage tests (the v0.3 ``validate_hypothesis`` / ``statement`` schema is gone).
"""

from __future__ import annotations

import json
import re

import pytest

from conftest import Sleeper
from m2_support import tracker
from m3_support import alternative, critic_review, falsification, hypothesis
from sciforge.hypothesis_validation import (
    CLAIM_LEVELS,
    HYPOTHESIS_LABEL,
    HYPOTHESIS_NOTICE,
    MEASURABLE_PATTERN,
    SIMILARITY_THRESHOLD,
    ValidationContext,
    causality_statement,
    confidence_ceiling,
    evidence_claim_level,
    final_confidence,
    jaccard,
    protocol_detail_types,
    revision_link_reasons,
    source_quality_summary,
    supported_claim_level,
    unhedged_causal_spans,
    validate_hypothesis_fields,
)
from sciforge.llm.audit import ModelCallAudit
from sciforge.llm.budget import RetryPolicy
from sciforge.llm.fake import FakeModelClient
from sciforge.stages.common import CallContext
from sciforge.stages.hypotheses import (
    CRITIC_STAGE,
    REVISION_STAGE,
    STAGE,
    WITHHELD_EXPLANATION,
    hypotheses_json,
    run_hypotheses_stage,
)

R1, R2, R3 = "rec_" + "a" * 16, "rec_" + "b" * 16, "rec_" + "c" * 16
JOURNAL, PREPRINT = "peer-reviewed journal article", "preprint"
Q1 = "Shear exposure increased P-selectin expression compared with static controls."
Q2 = "Unfolded factor binds platelet GPIb more strongly under elongational flow"
Q3 = "GPIb blockade abolished the shear-induced rise in P-selectin"
Q4 = "Antibody blockade of GPIb prevented platelet activation under shear"
Q5 = "P-selectin levels did not differ between shear and static groups"


def ev(eid, rid, quote, category="established", **kw):
    base = {"evidence_id": eid, "source_record_id": rid, "claim": quote.rstrip(".") + ".", "quote": quote,
            "finding": None, "methods": None, "limitations": "Abstract only.", "relevance": "Shear mechanism.",
            "evidence_category": category, "confidence": "moderate", "access_level": "pubmed_abstract",
            "abstract_only": True}
    base.update(kw)
    return base


EVIDENCE = [ev("ev_0001", R1, Q1), ev("ev_0002", R2, Q2), ev("ev_0003", R3, Q3), ev("ev_0004", R1, Q4),
            ev("ev_0005", R2, Q5, "conflicting"), ev("ev_0006", R2, "Flow may matter", "inference")]
BY_ID = {e["evidence_id"]: e for e in EVIDENCE}
GAPS = [{"gap_id": "gap_01", "gap_statement": "Whether P-selectin exposure depends on GPIb is untested.",
         "supporting_evidence_ids": ["ev_0001", "ev_0002"], "conflicting_evidence_ids": [],
         "why_unresolved": "Separate reports.", "confidence": "moderate"},
        {"gap_id": "gap_02", "gap_statement": "Conflicting shear results are unexplained.",
         "supporting_evidence_ids": ["ev_0001"], "conflicting_evidence_ids": ["ev_0005"],
         "why_unresolved": "Different designs.", "confidence": "low"}]
TYPES = {R1: JOURNAL, R2: JOURNAL, R3: PREPRINT}
TEXTS = {R1: f"Background sentence. {Q1} {Q4}. More text.", R2: f"{Q2}. {Q5}.", R3: f"{Q3}."}


def vctx(types=None):
    return ValidationContext(evidence_by_id=BY_ID, gaps_by_id={g["gap_id"]: g for g in GAPS},
                             known_sources={R1, R2, R3}, source_types=TYPES if types is None else types,
                             source_texts=TEXTS)


def check(raw, types=None):
    return validate_hypothesis_fields(raw, vctx(types))


def codes(raw, types=None):
    return [r["code"] for r in check(raw, types).reasons]


def hard(raw):
    return [r["code"] for r in check(raw).hard]


def soft(raw):
    return [r["code"] for r in check(raw).soft]


# ================================================================== schema


def test_valid_hypothesis_passes_all_checks():
    c = check(hypothesis())
    assert c.passed and c.reasons == [] and c.evidence_ids == ["ev_0001", "ev_0002"]
    assert c.supported_level == "mechanistic_support"


def test_schema_missing_and_extra_fields_are_hard():
    raw = hypothesis()
    del raw["prediction"]
    assert "schema_violation" in hard(raw)
    assert "unexpected_field" in hard(hypothesis(label="validated discovery"))   # model may not set the label
    assert "unexpected_field" in hard(hypothesis(probability=0.8))
    assert "fabricated_bibliographic_field" in hard(hypothesis(doi="10.1234/abc"))
    assert "schema_violation" in hard(hypothesis(assumptions="not a list"))
    assert "schema_violation" in hard(hypothesis(falsification_test="just test it"))
    assert "schema_violation" in hard(hypothesis(alternative_explanation="something else"))
    assert "unexpected_field" in hard(hypothesis(falsification_test=falsification(protocol="step one")))
    assert "malformed_item" in hard("not an object") or "schema_violation" in hard("not an object")


@pytest.mark.parametrize("level", ["causal", "strong", "", None, "Causal_Claim"])
def test_invalid_claim_level_is_hard(level):
    assert "invalid_claim_level" in hard(hypothesis(mechanistic_claim_level=level))


@pytest.mark.parametrize("conf", ["85%", 0.85, "very high", "probable", None, "p=0.9"])
def test_confidence_must_be_qualitative(conf):
    assert "invalid_confidence" in hard(hypothesis(confidence=conf))


# ================================================================== evidence linkage


def test_evidence_ids_must_resolve_to_accepted_evidence():
    assert hard(hypothesis(evidence_ids=[])) == ["hypothesis_missing_evidence"]
    # ev_0404 = an id SciForge rejected (rejected evidence never receives an accepted id) -> cannot be cited
    assert hard(hypothesis(evidence_ids=["ev_0001", "ev_0404"])) == ["unknown_evidence_id"]
    assert "unknown_evidence_id" in hard(hypothesis(evidence_ids=["rec_" + "a" * 16]))
    assert "unknown_evidence_id" in hard(hypothesis(alternative_explanation=alternative(
        basis="evidence", evidence_ids=["ev_0999"])))
    assert "unknown_evidence_id" in hard(hypothesis(rationale="See ev_0777 for support."))


def test_research_gap_required_and_must_resolve():
    assert hard(hypothesis(research_gap_id="")) == ["hypothesis_missing_gap"]
    assert hard(hypothesis(research_gap_id=None)) == ["hypothesis_missing_gap"]
    assert hard(hypothesis(research_gap_id="gap_09")) == ["unknown_gap_id"]


@pytest.mark.parametrize("field,text,code", [
    ("hypothesis", "GPIb may matter (see doi 10.1182/blood.2019.123).", "identifier_in_text"),
    ("rationale", "Smith et al. reported that GPIb may reduce this.", "bibliographic_text"),
    ("rationale", "As shown by Jones (2019), GPIb may reduce this.", "bibliographic_text"),
    ("rationale", "Reported in PMID 27182818.", "identifier_in_text"),
    ("prediction", "Lower expression, see https://example.org/paper.", "identifier_in_text"),
    ("rationale", "A doi was reported for this work.", "bibliographic_text"),
    ("hypothesis", "Per rec_0000000000000000 GPIb may reduce it.", "unknown_source_id"),
])
def test_no_bibliographic_identity_in_model_text(field, text, code):
    assert code in hard(hypothesis(**{field: text}))


def test_bibliographic_text_in_nested_fields():
    assert "bibliographic_text" in hard(hypothesis(falsification_test=falsification(
        measured="P-selectin as in Lee et al.")))
    assert "bibliographic_text" in hard(hypothesis(alternative_explanation=alternative(
        explanation="Membrane stretch (2021) may explain it.")))
    assert "bibliographic_text" in hard(hypothesis(evidence_limitations=["Brown et al. used small samples."]))


# ================================================================== claim level / causal language


@pytest.mark.parametrize("evidence,expected", [
    (ev("x", R1, Q1), "association"),
    (ev("x", R1, Q2), "mechanistic_support"),
    (ev("x", R1, Q3), "causal_claim"),
    (ev("x", R1, "Platelets were studied under shear"), "observation"),
    (ev("x", R1, Q5, "conflicting"), "association"),
    (ev("x", R1, Q3, "conflicting"), "observation"),
    (ev("x", R1, Q3, "inference"), "observation"),
    (ev("x", R1, Q3, "hypothesis"), "observation"),
])
def test_evidence_claim_level_mapping(evidence, expected):
    assert evidence_claim_level(evidence) == expected


def test_supported_level_is_max_and_causal_needs_two_peer_reviewed_interventional_sources():
    assert supported_claim_level(["ev_0001"], BY_ID, TYPES)[0] == "association"
    assert supported_claim_level(["ev_0001", "ev_0002"], BY_ID, TYPES)[0] == "mechanistic_support"
    # R3 is a preprint -> only one peer-reviewed interventional source -> capped at mechanistic support
    level, reasons = supported_claim_level(["ev_0003", "ev_0004"], BY_ID, TYPES)
    assert level == "mechanistic_support" and any("capped" in r for r in reasons)
    both_journal = {**TYPES, R3: JOURNAL}
    assert supported_claim_level(["ev_0003", "ev_0004"], BY_ID, both_journal)[0] == "causal_claim"
    # a conflicting item among the cited evidence also blocks causal_claim
    assert supported_claim_level(["ev_0003", "ev_0004", "ev_0005"], BY_ID, both_journal)[0] == "mechanistic_support"
    # unknown source type (e.g. demo / unclassified) never supports causal_claim
    assert supported_claim_level(["ev_0003", "ev_0004"], BY_ID, {})[0] == "mechanistic_support"


def test_claim_level_above_evidence_is_flagged_not_upgraded():
    c = check(hypothesis(evidence_ids=["ev_0001"], mechanistic_claim_level="mechanistic_support"))
    flag = [r for r in c.soft if r["code"] == "claim_level_exceeds_evidence"][0]
    assert flag["proposed"] == "mechanistic_support" and flag["supported"] == "association"
    assert "claim_level_exceeds_evidence" in soft(hypothesis(mechanistic_claim_level="causal_claim"))
    # the lower level is fine (no silent upgrade in either direction)
    assert check(hypothesis(mechanistic_claim_level="association")).passed


def test_causal_claim_allowed_only_when_supported():
    raw = hypothesis(evidence_ids=["ev_0003", "ev_0004"], mechanistic_claim_level="causal_claim",
                     hypothesis="GPIb engagement drives shear-induced P-selectin exposure.")
    assert "claim_level_exceeds_evidence" in codes(raw)                     # preprint R3
    assert codes(raw, types={**TYPES, R3: JOURNAL}) == []                    # causal wording allowed at causal level


@pytest.mark.parametrize("text", [
    "GPIb binding causes P-selectin exposure.",
    "Shear drives platelet activation through GPIb.",
    "Unfolding leads to stronger platelet capture.",
    "GPIb is responsible for the P-selectin increase.",
    "Shear induces P-selectin exposure.",
    "GPIb mediates the shear response.",
    "Factor unfolding determines the platelet response.",
    "Shear triggers activation; GPIb may matter.",
])
def test_unhedged_causal_language_flagged_below_causal_level(text):
    assert unhedged_causal_spans(text) >= 1
    assert "unhedged_causal_language" in soft(hypothesis(hypothesis=text))


@pytest.mark.parametrize("text", [
    "GPIb binding may cause P-selectin exposure.",
    "Blocking GPIb might reduce the shear-induced increase.",
    "We hypothesize that shear drives activation through GPIb.",
    "Whether GPIb mediates the shear response is untested.",
    "Factor unfolding could lead to stronger platelet capture.",
])
def test_hedged_causal_language_passes(text):
    assert unhedged_causal_spans(text) == 0
    assert "unhedged_causal_language" not in soft(hypothesis(hypothesis=text))


def test_hedge_must_precede_causal_verb_in_same_sentence():
    assert unhedged_causal_spans("GPIb may matter. Shear causes activation.") == 1
    assert unhedged_causal_spans("Shear causes activation, which may matter.") == 1


def test_causality_statement_required_below_causal_level():
    for level in CLAIM_LEVELS[:-1]:
        s = causality_statement(level, "mechanistic_support")
        assert s.startswith("Causality is not established")
    assert "untested" in causality_statement("causal_claim", "causal_claim")


# ================================================================== prediction


def test_prediction_required_and_measurable():
    assert "prediction_missing" in soft(hypothesis(prediction="  "))
    for vague in ("GPIb is important for platelets.", "The mechanism will become clear.",
                  "Platelets will respond."):
        assert "prediction_not_measurable" in soft(hypothesis(prediction=vague)), vague
    for ok in ("P-selectin expression is lower with blockade.", "Activation rate increases under shear.",
               "Levels correlate with shear magnitude.", "A fold change compared with controls."):
        assert MEASURABLE_PATTERN.search(ok)
        assert "prediction_not_measurable" not in soft(hypothesis(prediction=ok)), ok


def test_prediction_restating_hypothesis_is_flagged():
    h = "Blocking GPIb may reduce the shear-induced P-selectin increase."
    assert jaccard(h, "Blocking GPIb would reduce the shear-induced P-selectin increase.") >= SIMILARITY_THRESHOLD
    assert "prediction_restates_hypothesis" in soft(hypothesis(
        prediction="Blocking GPIb would reduce the shear-induced P-selectin increase."))


def test_prediction_must_discriminate_from_alternative():
    same = "P-selectin expression under shear is lower when GPIb is blocked than without blockade."
    assert "prediction_not_discriminating" in soft(hypothesis(alternative_explanation=alternative(explanation=same)))
    assert "falsification_not_discriminating" in soft(hypothesis(falsification_test=falsification(
        weakening_result="P-selectin falls with blockade.", supporting_result="P-selectin falls with blockade.")))


# ================================================================== alternative / falsification / protocol


def test_alternative_explanation_required_and_basis_enforced():
    assert "alternative_missing" in soft(hypothesis(alternative_explanation=alternative(explanation="")))
    assert "alternative_basis_invalid" in soft(hypothesis(alternative_explanation=alternative(basis="guess")))
    assert "alternative_basis_invalid" in soft(hypothesis(alternative_explanation=alternative(basis="evidence")))
    assert "alternative_basis_invalid" in soft(hypothesis(alternative_explanation=alternative(
        basis="inference", evidence_ids=["ev_0005"])))
    ok = hypothesis(evidence_ids=["ev_0001", "ev_0002"], research_gap_id="gap_02",
                    alternative_explanation=alternative(basis="evidence", evidence_ids=["ev_0005"],
                                                        explanation="Static and shear groups might not differ."))
    c = check(ok)
    assert c.passed and c.alternative_evidence_ids == ["ev_0005"]


@pytest.mark.parametrize("part", ["manipulated_or_compared", "measured", "weakening_result", "supporting_result"])
def test_falsification_requires_all_four_parts(part):
    assert "falsification_incomplete" in soft(hypothesis(falsification_test=falsification(**{part: ""})))
    raw = hypothesis()
    del raw["falsification_test"][part]
    assert "schema_violation" in hard(raw) or "falsification_incomplete" in soft(raw)


@pytest.mark.parametrize("text,kind", [
    ("Treat platelets with 10 µM inhibitor.", "concentration"),
    ("Add 50 µL of antibody to each sample.", "volume"),
    ("Centrifuge at 1500 rpm before measuring.", "centrifugation"),
    ("Keep samples at 37 °C during flow.", "temperature"),
    ("Incubate platelets with antibody for 30 min.", "incubation"),
    ("1. Isolate platelets\n2. Apply shear\n3. Stain", "numbered_steps"),
    ("Step 2 applies shear.", "numbered_steps"),
])
def test_protocol_detail_guard(text, kind):
    assert kind in protocol_detail_types(text)
    assert "procedural_protocol_detail" in soft(hypothesis(falsification_test=falsification(
        manipulated_or_compared=text)))


def test_high_level_falsification_is_not_protocol():
    assert protocol_detail_types("Platelets under high shear with versus without GPIb blockade.") == []


# ================================================================== quotes / discovery wording / numbers


def test_quotes_must_be_traceable_to_verified_source_text():
    traceable = hypothesis(rationale='The abstract says "increased P-selectin expression compared with static".')
    assert check(traceable).passed
    assert "untraceable_quote" in soft(hypothesis(rationale='The abstract says "GPIb is the master switch".'))


@pytest.mark.parametrize("text", [
    "This discovery suggests GPIb may reduce the increase.",
    "Our finding is that GPIb may reduce the increase.",
    "A validated mechanism: GPIb may reduce the increase.",
    "It is proven that GPIb may reduce the increase.",
    "The data confirms GPIb may reduce the increase.",
    "This establishes that GPIb may reduce the increase.",
])
def test_self_labelled_discovery_is_flagged(text):
    c = codes(hypothesis(hypothesis=text))
    assert "self_labeled_discovery" in c or "hypothesis_asserted_as_fact" in c


def test_unsupported_numbers_flagged():
    assert "unsupported_claim" in soft(hypothesis(prediction="P-selectin expression drops by 75% with blockade."))


# ================================================================== confidence / source quality


def ceiling(ids, types=TYPES, level="mechanistic_support", gap=None):
    return confidence_ceiling(ids, BY_ID, types, level=level, gap=gap)


def test_confidence_ceiling_rules():
    assert ceiling(["ev_0001"])[0] == "low"                                  # one source
    c, reasons = ceiling(["ev_0001", "ev_0002"])
    assert c == "moderate" and any("abstract-only" in r for r in reasons)    # abstract-only caps at moderate
    full = {**BY_ID, "ev_0001": {**BY_ID["ev_0001"], "abstract_only": False},
            "ev_0002": {**BY_ID["ev_0002"], "abstract_only": False}}
    assert confidence_ceiling(["ev_0001", "ev_0002"], full, TYPES, level="association")[0] == "high"
    assert confidence_ceiling(["ev_0001", "ev_0002"], full, TYPES, level="causal_claim")[0] == "moderate"
    assert confidence_ceiling(["ev_0001", "ev_0002"], full, {}, level="association")[0] == "moderate"  # unknown
    assert ceiling(["ev_0001", "ev_0005"])[0] == "low"                       # conflicting cited
    assert ceiling(["ev_0001", "ev_0002"], gap=GAPS[1])[0] == "low"          # gap records conflicts
    assert ceiling(["ev_0006", "ev_0005"])[0] == "low"                       # no established item


def test_preprints_lower_the_ceiling():
    all_pre = {R1: PREPRINT, R2: PREPRINT, R3: PREPRINT}
    c, reasons = ceiling(["ev_0001", "ev_0002"], types=all_pre)
    assert c == "low" and any("preprints" in r for r in reasons)
    c, reasons = ceiling(["ev_0001", "ev_0003"])                             # R1 journal + R3 preprint
    assert c == "low" and any("lowered one step" in r for r in reasons)


def test_final_confidence_is_min_of_model_and_ceiling():
    assert final_confidence("high", "moderate") == "moderate"
    assert final_confidence("low", "high") == "low"
    assert final_confidence("moderate", "low") == "low"


def test_source_quality_summary_labels_preprints():
    s = source_quality_summary(["ev_0001", "ev_0003"], BY_ID, TYPES)
    assert "Preprint — not peer-reviewed" in s and "2 distinct source(s)" in s and "abstract-only" in s


def test_revision_link_rules():
    assert revision_link_reasons(["ev_0001", "ev_0002"], ["ev_0001", "ev_0002"], []) == []
    assert [r["code"] for r in revision_link_reasons(["ev_0001", "ev_0003"], ["ev_0001", "ev_0002"], [])] == [
        "revision_added_evidence", "revision_dropped_supported_evidence"]
    assert [r["code"] for r in revision_link_reasons(["ev_0001"], ["ev_0001", "ev_0002"], [])] == [
        "revision_dropped_supported_evidence"]
    assert revision_link_reasons(["ev_0001"], ["ev_0001", "ev_0002"], ["ev_0002"]) == []   # critic-flagged drop


# ================================================================== stage: generation -> critic -> revision


def ctx(script, max_attempts=30):
    client = FakeModelClient(script)
    return CallContext(client=client, tracker=tracker(max_attempts=max_attempts), audit=ModelCallAudit(),
                       retry=RetryPolicy(), sleep=Sleeper()), client


def run_stage(script, types=TYPES, max_attempts=30):
    c, client = ctx(script, max_attempts)
    res = run_hypotheses_stage(c, {"research_question": "Q?"}, EVIDENCE, GAPS, {R1, R2, R3}, source_types=types,
                               source_texts=TEXTS)
    return res, client


def payload(client, i):
    return json.loads(client.requests[i].messages[-1].content)


CAUSAL = "GPIb engagement drives the shear-induced P-selectin increase."
FIXED = "GPIb engagement may contribute to the shear-induced P-selectin increase."


def test_clean_path_no_revision_required():
    res, client = run_stage([{"hypotheses": [hypothesis(confidence="high")]},
                             {"reviews": [critic_review("hyp_01")]}])
    assert client.remaining == 0 and [r.stage for r in client.requests] == [STAGE, CRITIC_STAGE]
    assert res.critic_status == "completed" and res.revision_status == "not_required"
    h = res.accepted[0]
    assert h["label"] == HYPOTHESIS_LABEL and h["notice"] == HYPOTHESIS_NOTICE
    assert h["hypothesis_id"] == "hyp_01" and h["research_gap_id"] == "gap_01"
    assert h["stress_test"]["revision_status"] == "no revision required"
    assert h["stress_test"]["critic_status"] == "completed"
    assert set(h["stress_test"]["critic_findings"]) == {
        "evidence_supports_mechanism", "causal_language_exceeds_evidence", "distinct_from_evidence",
        "prediction_measurable", "prediction_discriminates", "falsification_meaningful",
        "ignored_contradictions_or_missing_evidence", "confidence_consistent"}
    assert h["causality_statement"].startswith("Causality is not established")
    # confidence: model proposed high, validator caps at moderate (abstract-only) and records both
    d = h["confidence_detail"]
    assert h["confidence"] == "moderate" and d["proposed_by_model"] == "high" and d["capped"] is True
    assert d["deterministic_ceiling"] == "moderate" and d["reasons"]
    assert "%" not in json.dumps(d) and not re.search(r"probab\w*\"\s*:", json.dumps(h))
    assert h["validation"]["status"] == "passed_deterministic_validation"
    assert h["source_quality_summary"].startswith("2 evidence item(s)")
    data = hypotheses_json(res)
    assert data["critic"]["status"] == "completed" and data["revision"]["status"] == "not_required"
    assert data["attempts_by_stage"] == {STAGE: 1, CRITIC_STAGE: 1, REVISION_STAGE: 0}


def test_generation_and_critic_payloads_are_opaque_and_separate():
    res, client = run_stage([{"hypotheses": [hypothesis()]}, {"reviews": [critic_review("hyp_01")]}])
    gen, crit = payload(client, 0), payload(client, 1)
    assert set(gen) == {"question_definition", "evidence", "research_gaps"}
    assert set(crit) == {"question_definition", "evidence", "research_gaps", "candidate_hypotheses"}
    assert client.requests[0].instructions != client.requests[1].instructions
    assert "critic" in client.requests[1].instructions.lower()
    assert crit["candidate_hypotheses"][0]["hypothesis_id"] == "hyp_01"
    assert crit["candidate_hypotheses"][0]["deterministic_flags"] == []
    assert crit["evidence"][0]["verified_quote"] == Q1
    for req in client.requests:
        blob = json.dumps([m.content for m in req.messages])
        for key in ('"title"', '"authors"', '"journal"', '"doi"', '"pmid"', '"url"', '"year"'):
            assert key not in blob


def test_critic_rejection_then_successful_revision():
    critic = {"reviews": [critic_review("hyp_01", fail=("confidence_consistent",),
                                        problems=("Confidence exceeds the evidence.",))]}
    revised = hypothesis(hypothesis_id="hyp_01", confidence="low")
    res, client = run_stage([{"hypotheses": [hypothesis(confidence="high")]}, critic, {"hypotheses": [revised]}])
    assert [r.stage for r in client.requests] == [STAGE, CRITIC_STAGE, REVISION_STAGE]
    rev = payload(client, 2)["hypotheses_to_revise"][0]
    assert rev["hypothesis"]["hypothesis_id"] == "hyp_01" and rev["critic"]["substantive_problems"]
    assert res.revision_status == "completed" and res.revised == 1
    h = res.accepted[0]
    assert h["stress_test"]["revision_status"] == "revised" and h["confidence"] == "low"
    assert "critic was not re-run" in h["stress_test"]["note"]


def test_deterministic_flag_forces_revision_even_when_critic_passes():
    res, client = run_stage([{"hypotheses": [hypothesis(hypothesis=CAUSAL)]},
                             {"reviews": [critic_review("hyp_01")]},
                             {"hypotheses": [hypothesis(hypothesis_id="hyp_01", hypothesis=FIXED)]}])
    assert payload(client, 1)["candidate_hypotheses"][0]["deterministic_flags"] == ["unhedged_causal_language"]
    assert res.accepted[0]["hypothesis"] == FIXED
    assert res.accepted[0]["validation"]["initial_flags"] == ["unhedged_causal_language"]


def test_revision_that_still_fails_is_rejected_not_accepted():
    res, _ = run_stage([{"hypotheses": [hypothesis(hypothesis=CAUSAL)]},
                        {"reviews": [critic_review("hyp_01", fail=("causal_language_exceeds_evidence",))]},
                        {"hypotheses": [hypothesis(hypothesis_id="hyp_01", hypothesis=CAUSAL)]}])
    assert res.accepted == []
    rej = res.rejected[0]
    assert rej["rejected_at"] == "revision" and "unresolved_after_revision" in rej["reason_codes"]
    assert "unhedged_causal_language" in rej["reason_codes"]
    assert CAUSAL not in json.dumps(rej)                                       # rejected text never kept
    assert rej["stress_test"]["critic_verdicts"]["causal_language_exceeds_evidence"] == "fail"


@pytest.mark.parametrize("change,code", [
    ({"evidence_ids": ["ev_0001", "ev_0002", "ev_0004"]}, "revision_added_evidence"),
    ({"evidence_ids": ["ev_0002"], "mechanistic_claim_level": "mechanistic_support"},
     "revision_dropped_supported_evidence"),
    ({"research_gap_id": "gap_02"}, "revision_changed_gap"),
    ({"evidence_ids": ["ev_0001", "ev_0404"]}, "unknown_evidence_id"),
    ({"rationale": "Consistent with Smith et al. and GPIb binding."}, "bibliographic_text"),
])
def test_revision_must_preserve_links_and_add_no_citations(change, code):
    res, _ = run_stage([{"hypotheses": [hypothesis(hypothesis=CAUSAL)]}, {"reviews": [critic_review("hyp_01")]},
                        {"hypotheses": [hypothesis(hypothesis_id="hyp_01", hypothesis=FIXED, **change)]}])
    assert res.accepted == [] and code in res.rejected[0]["reason_codes"]


def test_revision_may_drop_a_link_the_critic_flagged_unsupported():
    critic = {"reviews": [critic_review("hyp_01", fail=("evidence_supports_mechanism",),
                                        unsupported=("ev_0002",))]}
    revised = hypothesis(hypothesis_id="hyp_01", evidence_ids=["ev_0001"], mechanistic_claim_level="association",
                         hypothesis="GPIb engagement may be associated with the shear-induced P-selectin increase.")
    res, _ = run_stage([{"hypotheses": [hypothesis()]}, critic, {"hypotheses": [revised]}])
    assert res.accepted[0]["evidence_ids"] == ["ev_0001"]
    assert res.accepted[0]["stress_test"]["unsupported_evidence_ids"] == ["ev_0002"]


def test_revision_missing_item_rejects():
    res, _ = run_stage([{"hypotheses": [hypothesis(hypothesis=CAUSAL), hypothesis(hypothesis_id="h2")]},
                        {"reviews": [critic_review("hyp_01"), critic_review("hyp_02")]}, {"hypotheses": []}])
    assert [h["hypothesis_id"] for h in res.accepted] == ["hyp_02"]
    assert "revision_missing" in res.rejected[0]["reason_codes"]


def test_hard_failures_rejected_at_generation_and_never_sent_to_critic():
    gen = {"hypotheses": [hypothesis(evidence_ids=["ev_0404"]), hypothesis(research_gap_id="gap_07"),
                          hypothesis(rationale="Per doi 10.1000/xyz."), hypothesis()]}
    res, client = run_stage([gen, {"reviews": [critic_review("hyp_01")]}])
    assert [r["rejected_at"] for r in res.rejected] == ["generation"] * 3
    assert len(payload(client, 1)["candidate_hypotheses"]) == 1
    assert "10.1000/xyz" not in json.dumps(payload(client, 1))
    assert res.accepted[0]["hypothesis_id"] == "hyp_01"


def test_critic_explanations_are_sanitised():
    critic = {"reviews": [critic_review("hyp_01", explanation="Matches doi 10.1182/blood.123 by Smith et al.")]}
    res, _ = run_stage([{"hypotheses": [hypothesis()]}, critic])
    findings = res.accepted[0]["stress_test"]["critic_findings"]
    assert findings["evidence_supports_mechanism"]["explanation"] == WITHHELD_EXPLANATION
    assert "10.1182" not in json.dumps(res.accepted) and res.accepted[0]["stress_test"]["critic_output_sanitized"]
    assert res.redaction_plans and any(p["synthesis_stage"] == CRITIC_STAGE for p in res.redaction_plans)


def test_critic_review_for_unknown_candidate_ignored_and_incomplete():
    res, _ = run_stage([{"hypotheses": [hypothesis()]}, {"reviews": [critic_review("hyp_09")]}])
    assert res.critic_status == "incomplete"
    assert res.accepted[0]["stress_test"]["critic_status"] == "incomplete"
    assert "NOT stress-tested" in res.accepted[0]["stress_test"]["note"]


def test_critic_failure_means_not_stress_tested():
    res, _ = run_stage([{"hypotheses": [hypothesis()]}, "not json", "still not json"])
    assert res.critic_status == "failed" and res.accepted[0]["stress_test"]["critic_status"] == "failed"
    assert "NOT stress-tested" in res.accepted[0]["stress_test"]["note"]
    assert "critic_findings" not in res.accepted[0]["stress_test"]


def test_budget_exhausted_before_critic():
    """Clean candidates stay (marked NOT stress-tested); flagged ones are rejected, never presented as critiqued."""
    res, client = run_stage([{"hypotheses": [hypothesis(), hypothesis(hypothesis_id="h2", hypothesis=CAUSAL)]}],
                            max_attempts=1)
    assert len(client.requests) == 1 and res.stop == "budget_exhausted"
    assert res.critic_status == "not_run" and res.revision_status == "not_run"
    assert [h["hypothesis_id"] for h in res.accepted] == ["hyp_01"]
    assert res.accepted[0]["stress_test"]["critic_status"] == "not_run"
    assert "NOT stress-tested" in res.accepted[0]["stress_test"]["note"]
    assert "revision_not_run" in res.rejected[0]["reason_codes"]


def test_budget_exhausted_before_revision():
    res, client = run_stage([{"hypotheses": [hypothesis(hypothesis=CAUSAL)]},
                             {"reviews": [critic_review("hyp_01")]}], max_attempts=2)
    assert len(client.requests) == 2 and res.revision_status == "not_run" and res.accepted == []
    assert set(res.rejected[0]["reason_codes"]) >= {"unhedged_causal_language", "revision_not_run"}


def test_critic_substantive_problem_without_revision_rejects():
    res, _ = run_stage([{"hypotheses": [hypothesis()]},
                        {"reviews": [critic_review("hyp_01", fail=("prediction_discriminates",))]}], max_attempts=2)
    assert res.accepted == [] and res.rejected[0]["reason_codes"][0] == "critic_substantive_problems"


def test_uncertain_verdicts_alone_do_not_trigger_revision():
    res, client = run_stage([{"hypotheses": [hypothesis()]},
                             {"reviews": [critic_review("hyp_01", uncertain=("evidence_supports_mechanism",))]}])
    assert len(client.requests) == 2 and res.accepted[0]["stress_test"]["revision_status"] == "no revision required"


def test_preprint_backed_hypothesis_lowers_confidence_and_is_labelled():
    raw = hypothesis(evidence_ids=["ev_0001", "ev_0003"], mechanistic_claim_level="association",
                     hypothesis="GPIb engagement may be associated with the shear-induced P-selectin increase.",
                     confidence="high")
    res, _ = run_stage([{"hypotheses": [raw]}, {"reviews": [critic_review("hyp_01")]}])
    h = res.accepted[0]
    assert h["confidence"] == "low" and "Preprint — not peer-reviewed" in h["source_quality_summary"]


def test_no_hidden_reasoning_fields_stored():
    res, _ = run_stage([{"hypotheses": [hypothesis()]}, {"reviews": [critic_review("hyp_01")]}])
    blob = json.dumps(hypotheses_json(res))
    for key in ('"reasoning"', '"chain_of_thought"', '"thinking"', '"reasoning_trace"', '"encrypted_content"'):
        assert key not in blob
