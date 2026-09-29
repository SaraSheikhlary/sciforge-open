"""v0.3 M3: report narrative validation, deterministic template and citation resolution (offline)."""

from __future__ import annotations

import pytest

from m2_support import BIB, record, verified
from m3_support import paragraph
from sciforge.stages.report import (
    NARRATIVE_SECTIONS, SECTION_TITLES, SourceIndex, build_report, render_citation, unresolved_marker,
    validate_paragraph,
)
from sciforge.stages.synthesis_common import SynthesisResult
from test_gaps_stage import BY_ID, EVIDENCE, R1, R2


def check(raw):
    return validate_paragraph(raw, evidence_by_id=BY_ID, known_sources={R1, R2}, next_id="par_01",
                              known_gaps={"gap_01"}, known_hypotheses={"hyp_01"})


def codes(raw):
    rec, reasons = check(raw)
    assert rec is None
    return [r["code"] for r in reasons]


def test_valid_paragraph_and_inline_refs():
    rec, reasons = check(paragraph(evidence_ids=[], text="P-selectin rose by 40% (ev_0001)."))
    assert reasons == [] and rec["evidence_ids"] == ["ev_0001"] and len(rec["text_sha256"]) == 64
    assert check(paragraph(section="next_steps", label="hypothesis", evidence_ids=[],
                           text="Test hyp_01 against gap_01."))[1] == []


def test_narrative_rejections():
    assert codes(paragraph(evidence_ids=["ev_0077"])) == ["unknown_evidence_id"]
    assert codes(paragraph(source_ids=["rec_1234567890abcdef"])) == ["unknown_source_id"]
    assert codes(paragraph(text="Shear matters [rec_deadbeefdeadbeef].", evidence_ids=["ev_0001"])) == ["unknown_source_id"]
    assert codes(paragraph(evidence_ids=[], text="Shear activates platelets.")) == ["missing_evidence_reference"]
    assert codes(paragraph(section="conflicting_evidence", label="conflicting", evidence_ids=[],
                           text="Studies disagree.")) == ["missing_evidence_reference"]
    assert codes(paragraph(journal="Invented Journal")) == ["fabricated_bibliographic_field"]
    assert codes(paragraph(text="Rose by 40% [ev_0001] (doi:10.5555/zz.1).")) == ["identifier_in_text",
                                                                                "bibliographic_text"]
    assert codes(paragraph(text="Rose by 40% [ev_0001], see 10.5555/zz.1 too.")) == ["identifier_in_text"]
    assert codes(paragraph(text="Rose by 40% [ev_0001], as Albemarle (1987) found.")) == ["bibliographic_text",
                                                                                          "unsupported_claim"]
    assert codes(paragraph(text="Rose by 60% [ev_0001].")) == ["unsupported_claim"]
    assert codes(paragraph(section="discussion")) == ["invalid_section"]
    assert codes(paragraph(label="proven")) == ["invalid_label"]
    assert codes(paragraph(section="next_steps", evidence_ids=[], text="Retest hyp_09.")) == ["unknown_hypothesis_id"]


def test_citation_rendered_from_v02_record_only():
    rec = record()
    line = render_citation("S1", rec, verified(rec), "pubmed_abstract")
    assert line == (
        f"- **[S1]** **[Source type unknown]** Quixote Albemarle; Vandersloot Perpetua. *{BIB['title']}*. {BIB['journal']}. 1987. "
        f"DOI: [10.5555/zqx.1987.424242](https://doi.org/10.5555/zqx.1987.424242). "
        f"PMID: [31415926](https://pubmed.ncbi.nlm.nih.gov/31415926/). URL: <https://pubmed.ncbi.nlm.nih.gov/31415926/>. "
        f"v0.2 verification: verified. Access: abstract only (PubMed abstract). "
        f"Source type: unknown (no conclusive bibliographic type metadata). Record: `{rec.record_id}`.")
    preprint = render_citation("S1", rec, verified(rec), "pubmed_abstract", "preprint")
    assert preprint.startswith("- **[S1]** **[Preprint — not peer-reviewed]** ")
    assert "Source type: **Preprint — not peer-reviewed** (from bibliographic metadata)." in preprint
    assert "Peer-reviewed journal article" not in preprint


def test_citation_missing_fields_never_guessed():
    rec = record(pmid=None, doi=None, authors=None, source="crossref")
    rec = rec.model_copy(update={"authors": [], "journal": None, "year": None, "source_url": None})
    line = render_citation("S2", rec, None, None)
    assert "authors not available" in line and "DOI: not available." in line and "PMID: not available." in line
    assert "URL: not available." in line and "v0.2 verification: not available." in line and "not accessed" in line


def test_citation_markdown_escaped():
    rec = record(title="A *bold* [link](http://x) | pipe")
    assert "\\*bold\\* \\[link\\](http://x) \\| pipe" in render_citation("S1", rec, None, None)


def test_citation_resolution_and_unresolved_fail_closed():
    r1, r2 = record(), record(pmid="27182818", doi=None, title="Second")
    index = SourceIndex([r1, r2], [verified(r1), verified(r2)], {})
    assert index.ref(r2.record_id) == "S1"
    assert index.ref("rec_ffffffffffffffff") == "[UNRESOLVED CITATION: rec_ffffffffffffffff]"
    assert index.ref(r1.record_id) == "S2" and index.ref(r2.record_id) == "S1"
    lines, citations, issues = index.render()
    assert lines[0].startswith("- **[S1]**") and "Second" in lines[0]
    assert lines[1] == ("- [UNRESOLVED CITATION: rec_ffffffffffffffff] — no v0.2 record with this id; "
                        "no citation rendered.")
    assert lines[2].startswith("- **[S2]**") and BIB["title"] in lines[2]
    assert [c["status"] for c in citations] == ["resolved", "unresolved", "resolved"]
    assert issues == [{"code": "unresolved_citation", "record_id": "rec_ffffffffffffffff",
                       "detail": "id not found among the v0.2 records; citation not rendered (fail closed)"}]
    assert "weird" not in unresolved_marker("weird <id>") and "redacted" in unresolved_marker("weird <id>")


def _base(accepted, records, **kw):
    source_texts = {"sources": [{"record_id": e["source_record_id"], "access_level": "pubmed_abstract"}
                                for e in accepted], "counts": {"ok": len(accepted)}, "eligibility_policy": "verified",
                    "max_sources": 10, "max_source_chars": 4000}
    evidence = {"accepted": accepted, "counts": {"accepted": len(accepted), "rejected": 0}}
    return build_report(question="Q?", question_definition=None, search_summary=None, source_texts=source_texts,
                        evidence=evidence, gaps=None, hypotheses=None, records=records,
                        verification=[verified(r) for r in records], stage_notes=[], **kw)


def test_unresolved_citation_in_report_when_record_missing():
    r1 = record()
    accepted = [{**EVIDENCE[0], "source_record_id": r1.record_id}, EVIDENCE[1]]   # R2 has no v0.2 record
    report, validation = _base(accepted, [r1], narrative=None)
    assert f"[UNRESOLVED CITATION: {R2}]" in report
    assert validation["status"] == "issues" and validation["issues"][0]["code"] == "unresolved_citation"
    assert validation["counts"]["citations_unresolved"] == 1 and validation["counts"]["citations_resolved"] == 1
    assert BIB["title"] in report


def test_report_sections_in_spec_order_and_code_fallbacks():
    r1 = record()
    accepted = [{**EVIDENCE[0], "source_record_id": r1.record_id}]
    report, validation = _base(accepted, [r1], narrative=None)
    positions = [report.index(f"## {SECTION_TITLES[k]}") for k in "ABCDEFGHIJ"]
    assert positions == sorted(positions) and len(validation["sections_present"]) == 10
    assert "Code-generated fallback" in report and "No validated research gaps." in report
    assert "| ev_0001 |" in report and "abstract only (PubMed abstract)" in report
    assert "v0.2 search metadata was not provided" in report


def test_rejected_narrative_excluded_and_accepted_rendered_with_label():
    r1 = record()
    accepted = [{**EVIDENCE[0], "source_record_id": r1.record_id}]
    ok, _ = validate_paragraph(paragraph(), evidence_by_id={"ev_0001": accepted[0]},
                               known_sources={r1.record_id}, next_id="par_01")
    narrative = SynthesisResult(stage="report", status="ok", accepted=[ok],
                                rejected=[{"reason_codes": ["unsupported_claim"], "item_index": 1}])
    report, validation = _base(accepted, [r1], narrative=narrative)
    assert "- **[established]** High shear increased P-selectin expression by 40% \\[ev\\_0001\\]. — evidence: ev_0001 [S1]" in report
    assert validation["status"] == "passed_with_rejections" and validation["counts"]["paragraphs_rejected"] == 1
    assert "text" not in validation["narrative"]["accepted"][0]


@pytest.mark.parametrize("section", NARRATIVE_SECTIONS)
def test_all_narrative_sections_accepted(section):
    rec, reasons = check(paragraph(section=section))
    assert reasons == [] and rec["section"] == section
