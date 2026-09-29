"""v0.3 M2 evidence extraction + deterministic validation."""

from __future__ import annotations

import pytest

from m2_support import STRUCTURED_ABSTRACT_TEXT, TS, item, record
from sciforge.llm.budget import BudgetLimits, BudgetTracker
from sciforge.llm.fake import FakeModelClient
from sciforge.sourcetext import SourceText, sha256_text
from sciforge.stages.common import CallContext
from sciforge.stages.extraction import run_extraction_stage
from sciforge.stages.schemas import CONFIDENCE_LEVELS, EVIDENCE_CATEGORIES, EVIDENCE_SCHEMA, ITEM_FIELDS
from sciforge.stages.validation import BIBLIOGRAPHIC_KEYS, identifier_warnings, validate_item

NONUM = {"finding": "Aggregation increased.", "methods": "Whole blood assay."}
QUOTE = "Shear exposure increased P-selectin expression by 40% compared with static controls."
REC = record()
OTHER = record(pmid="27182818", doi=None)
TEXTS = {REC.record_id: STRUCTURED_ABSTRACT_TEXT}
SUPPLIED = frozenset({REC.record_id, OTHER.record_id})


def check(raw):
    return validate_item(raw, request_texts=TEXTS, supplied_ids=SUPPLIED)


def codes(raw):
    return [r["code"] for r in check(raw)[1]]


def st(rec, text=STRUCTURED_ABSTRACT_TEXT, **kw):
    base = dict(record_id=rec.record_id, verification_status="verified", eligibility="verified", status="ok",
                access_level="pubmed_abstract", origin="pubmed", source_text=text, sha256=sha256_text(text),
                original_chars=len(text), text_chars=len(text), max_source_chars=4000, retrieved_at=TS)
    base.update(kw)
    return SourceText(**base)


def run(script, sources, **limits):
    client = FakeModelClient(script)
    ctx = CallContext(client=client, tracker=BudgetTracker(BudgetLimits(max_spend_usd=None, **limits)),
                      sleep=lambda s: None)
    return run_extraction_stage(ctx, {"research_question": "Q?"}, sources, question_definition_used=False), client


# ------------------------------------------------------------------ schema / enums

def test_schema_has_only_the_nine_fields_and_enums():
    props = EVIDENCE_SCHEMA["$defs"]["ModelEvidenceItem"]["properties"]
    assert tuple(props) == ITEM_FIELDS
    assert not set(props) & BIBLIOGRAPHIC_KEYS
    assert EVIDENCE_CATEGORIES == ("established", "conflicting", "inference", "hypothesis")
    assert CONFIDENCE_LEVELS == ("high", "moderate", "low")
    assert EVIDENCE_SCHEMA["properties"]["items"]["maxItems"] == 5
    assert EVIDENCE_SCHEMA["additionalProperties"] is False


# ------------------------------------------------------------------ per-item validation

def test_valid_item_accepted():
    validated, reasons, warnings = check(item(REC.record_id, QUOTE))
    assert reasons == [] and warnings == [] and validated.quote == QUOTE


def test_unknown_record_id_rejected():
    assert codes(item("rec_deadbeefdeadbeef", QUOTE)) == ["unknown_record_id"]
    assert codes(item(None, QUOTE)) == ["unknown_record_id", "schema_violation"]


def test_supplied_but_other_request_record_id_rejected():
    assert codes(item(OTHER.record_id, QUOTE)) == ["record_id_not_in_request"]


@pytest.mark.parametrize("key,value", [("title", "Invented title"), ("doi", "10.9999/fake"), ("authors", ["X Y"]),
                                       ("journal", "J"), ("year", 2020), ("pmid", "123"), ("url", "https://x.org"),
                                       ("Source URL", "https://x.org")])
def test_invented_bibliographic_fields_rejected(key, value):
    raw = item(REC.record_id, QUOTE, **{key: value})
    validated, reasons, _ = check(raw)
    assert validated is None
    assert reasons[0]["code"] == "bibliographic_field"
    assert reasons[0]["keys"] == [key.lower().replace(" ", "_")]  # normalised key name only, never the value
    assert "unexpected_field" not in [r["code"] for r in reasons]


def test_nested_bibliographic_field_detected():
    raw = item(REC.record_id, QUOTE, finding={"text": "x", "citation": "Smith 2020"})
    assert "bibliographic_field" in codes(raw)


def test_unexpected_non_bibliographic_field_rejected():
    assert codes(item(REC.record_id, QUOTE, notes="x")) == ["unexpected_field"]


@pytest.mark.parametrize("quote", [
    "Shear exposure raised P-selectin expression by 40% compared with static controls.",   # paraphrase
    "shear exposure increased P-selectin expression by 40% compared with static controls.",  # case change
    "Shear exposure increased P-selectin ... compared with static controls.",                # ellipsis
    "Shear exposure increased P-selectin expression by 40%  compared with static controls.",  # whitespace
    "Shear exposure increased P-selectin expression by 45% compared with static controls.",  # number change
])
def test_non_exact_quotes_rejected(quote):
    assert codes(item(REC.record_id, quote)) == ["quote_not_in_source"]


@pytest.mark.parametrize("quote", [None, "", "   "])
def test_missing_or_empty_quote_rejected(quote):
    raw = item(REC.record_id, quote)
    if quote is None:
        del raw["quote"]
    assert codes(raw) == ["missing_quote"]


def test_short_quote_rejected():
    assert codes(item(REC.record_id, "P-selectin")) == ["quote_too_short"]


def test_unsupported_category_and_confidence():
    assert codes(item(REC.record_id, QUOTE, evidence_category="proven")) == ["unsupported_evidence_category"]
    assert codes(item(REC.record_id, QUOTE, confidence="very high")) == ["unsupported_confidence"]


def test_multiple_reasons_collected_and_schema_violation():
    raw = item("rec_0000000000000000", "made up quote that is long enough", evidence_category="x", claim=" ",
               doi="10.1/x")
    assert codes(raw) == ["bibliographic_field", "unknown_record_id", "unsupported_evidence_category", "empty_claim"]
    assert codes(item(REC.record_id, QUOTE, claim=42)) == ["schema_violation"]
    raw = item(REC.record_id, QUOTE)
    del raw["methods"]
    assert codes(raw) == ["schema_violation"]


def test_malformed_item():
    assert codes("just a string") == ["malformed_item"]


def test_identifier_strings_flagged_not_rejected():
    raw = item(REC.record_id, QUOTE, claim="Shear activates platelets (doi:10.1234/xyz, PMID: 998877).",
               relevance="see https://example.org and arXiv:2101.00001, PMC123456")
    validated, reasons, warnings = check(raw)
    assert validated is not None and reasons == []
    assert {(w["field"], w["type"]) for w in warnings} == {
        ("claim", "doi"), ("claim", "pmid"), ("relevance", "url"), ("relevance", "arxiv"), ("relevance", "pmcid")}


def test_identifier_patterns_no_false_positive_on_doses():
    assert identifier_warnings({"finding": "10.5 mg/kg reduced IL-6 by 10.2%"}) == []


# ------------------------------------------------------------------ stage

def test_stage_keeps_valid_items_and_records_invalid_ones():
    good = item(REC.record_id, QUOTE)
    bad_quote = item(REC.record_id, "Shear exposure raised P-selectin expression a lot.")
    bib = item(REC.record_id, QUOTE, title="Invented")
    res, client = run([{"items": [good, bad_quote, bib]}], [st(REC)])
    assert len(client.requests) == 1
    assert [a["evidence_id"] for a in res.accepted] == ["ev_0001"]
    acc = res.accepted[0]
    assert acc["access_level"] == "pubmed_abstract" and acc["abstract_only"] is True
    assert acc["source_identity"] == "verified" and acc["source_text_sha256"] == sha256_text(STRUCTURED_ABSTRACT_TEXT)
    assert [r["reasons"][0]["code"] for r in res.rejected] == ["quote_not_in_source", "bibliographic_field"]
    assert "model_output" not in res.rejected[1] and "Invented" not in str(res.rejected)
    assert res.rejected[1]["diagnostics"]["bibliographic_keys"] == ["title"]
    assert res.rejected_raw[1]["model_output"]["title"] == "Invented"  # in memory only (debug opt-in)
    counts = res.counts()
    assert counts["accepted"] == 1 and counts["rejected"] == 2 and counts["rejected_bibliographic_field"] == 1


def test_one_call_per_source_and_ids_sequential():
    text2 = "Elevated shear also increased platelet aggregation in whole blood samples."
    s2 = st(OTHER, text=text2)
    res, client = run([{"items": [item(REC.record_id, QUOTE)]},
                       {"items": [item(OTHER.record_id, text2, **NONUM), item(REC.record_id, QUOTE)]}], [st(REC), s2])
    assert len(client.requests) == 2
    assert [a["evidence_id"] for a in res.accepted] == ["ev_0001", "ev_0002"]
    # a quote from source 1 returned in the call for source 2 is rejected (id not in that request)
    assert res.rejected[0]["reasons"][0]["code"] == "record_id_not_in_request"


def test_more_than_five_items_rejected_beyond_limit():
    res, _ = run([{"items": [item(REC.record_id, QUOTE)] * 6}], [st(REC)])
    assert len(res.accepted) == 5 and res.rejected[0]["reasons"][0]["code"] == "exceeds_item_limit"


def test_malformed_envelope_triggers_one_repair_then_failure_recorded():
    res, client = run(["[1, 2]", {"evidence": []}], [st(REC)])
    assert len(client.requests) == 2 and client.requests[1].stage == "extraction:repair"
    assert res.calls[0]["status"] == "failed" and res.calls[0]["repair_attempted"] is True
    assert res.accepted == [] and [e["error_type"] for e in res.errors] == ["schema_violation", "schema_violation"]


def test_invalid_json_repaired():
    res, client = run(["not json at all", {"items": [item(REC.record_id, QUOTE)]}], [st(REC)])
    assert len(res.accepted) == 1 and res.calls[0]["repair_attempted"] is True and res.calls[0]["attempts"] == 2


def test_failure_on_one_source_continues_with_next():
    text2 = "Elevated shear also increased platelet aggregation in whole blood samples."
    res, _ = run(["bad", "still bad", {"items": [item(OTHER.record_id, text2, **NONUM)]}], [st(REC), st(OTHER, text=text2)])
    assert [c["status"] for c in res.calls] == ["failed", "ok"] and len(res.accepted) == 1


def test_budget_stop_skips_remaining_sources():
    text2 = "Elevated shear also increased platelet aggregation in whole blood samples."
    third = record(pmid="16180339", doi=None)
    res, client = run([{"items": []}], [st(REC), st(OTHER, text=text2), st(third, text=text2)], max_attempts=1)
    assert [c["status"] for c in res.calls] == ["ok", "failed", "skipped"]
    assert res.calls[1]["error"]["error_type"] == "budget_exhausted"
    assert res.stop == "budget_exhausted" and len(client.requests) == 1


def test_partially_verified_label_propagates():
    res, _ = run([{"items": [item(REC.record_id, QUOTE)]}],
                 [st(REC, verification_status="partially_verified", eligibility="partially_verified_opt_in")])
    assert res.accepted[0]["source_identity"] == "partially_verified"


def test_crossref_access_level_propagates():
    res, _ = run([{"items": [item(REC.record_id, QUOTE)]}], [st(REC, access_level="crossref_abstract", origin="crossref")])
    assert res.accepted[0]["access_level"] == "crossref_abstract"
