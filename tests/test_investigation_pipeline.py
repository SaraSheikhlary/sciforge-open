"""v0.3 M3: end-to-end mocked investigation (question → sources → extraction → gaps → hypotheses → report)."""

from __future__ import annotations

import copy
import json

import pytest

from m2_support import BIB, question_output, tracker
from m3_support import BIB2, full_script, gap, hypothesis, load, narrative, paragraph, records, run, evidence_items


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SCIFORGE_DEBUG_KEEP_REJECTED_RAW", raising=False)
    monkeypatch.delenv("SCIFORGE_MAX_SOURCE_CHARS", raising=False)


def test_end_to_end_mocked_investigation(tmp_path, settings):
    recs, ver = records()
    before = copy.deepcopy([r.model_dump() for r in recs])
    result, client = run(tmp_path, settings, full_script(recs), recs=recs, ver=ver)
    assert client.remaining == 0 and len(client.requests) == 6
    assert sorted(p.name for p in result.run_dir.iterdir()) == [
        "evidence.json", "gaps.json", "hypotheses.json", "model_calls.json", "question.json", "report.md",
        "report_validation.json", "source_texts.json"]
    gaps = load(result, "gaps")
    assert gaps["status"] == "ok" and gaps["counts"]["accepted"] == 1 and gaps["model_layer"] == "v0.3-m3"
    hyps = load(result, "hypotheses")
    assert hyps["accepted"][0]["research_gap_ids"] == ["gap_01"] and hyps["accepted"][0]["label"] == "hypothesis"
    validation = load(result, "report_validation")
    assert validation["status"] == "passed" and validation["issues"] == []
    assert [c["record_id"] for c in validation["citations"]] == [recs[0].record_id, recs[1].record_id]
    report = result.files["report"].read_text(encoding="utf-8")
    for key in "ABCDEFGHIJ":
        assert f"## {key}." in report
    assert "Hypothesis: Blocking GPIb may reduce" in report
    # J: every bibliographic value comes from the v0.2 records
    for bib in (BIB, BIB2):
        for key in ("title", "journal", "doi", "pmid"):
            assert str(bib[key]) in report
    # model_calls: one audit entry per attempt, all six stages; nothing needed redaction
    calls = load(result, "model_calls")
    entries = calls["entries"] if isinstance(calls, dict) else calls
    assert [e["stage"] for e in entries] == ["question", "extraction", "extraction", "gaps", "hypotheses", "report"]
    assert not any(e.get("model_output_redacted") for e in entries)
    assert result.budget["used"]["attempts"] == 6
    assert [r.model_dump() for r in recs] == before


def test_model_never_receives_bibliographic_data_in_synthesis_stages(tmp_path, settings):
    recs, ver = records()
    _, client = run(tmp_path, settings, full_script(recs), recs=recs, ver=ver)
    for request in client.requests[3:]:
        blob = json.dumps({"instructions": request.instructions,
                           "messages": [m.content for m in request.messages],
                           "schema": request.json_schema}, default=str)
        for bib in (BIB, BIB2):
            for key in ("title", "journal", "doi", "pmid"):
                assert str(bib[key]) not in blob
            for author in bib["authors"]:
                assert author not in blob
        for key in ('"title"', '"authors"', '"journal"', '"doi"', '"pmid"', '"url"', '"quote"', '"methods"'):
            assert key not in blob


def test_deterministic_validation_precedence_end_to_end(tmp_path, settings):
    """High model confidence / 'established' labels never rescue a deterministic failure."""
    recs, ver = records()
    script = full_script(recs,
                         gaps={"gaps": [gap(), gap(supporting_evidence_ids=["ev_0404"], confidence="high")]},
                         hyps={"hypotheses": [hypothesis(),
                                              hypothesis(statement="It is proven that GPIb drives this.",
                                                         confidence="high")]},
                         narr=narrative(paragraph(label="established", text="Shear raised it by 90% [ev_0001].")))
    result, _ = run(tmp_path, settings, script, recs=recs, ver=ver)
    for name, code in (("gaps", "unknown_evidence_id"), ("hypotheses", "hypothesis_asserted_as_fact")):
        rej = load(result, name)["rejected"][0]
        assert rej["reason_codes"] == [code]
        assert rej["support"]["final_status"] == "rejected" and rej["validation_status"] == "rejected_deterministic"
    v = load(result, "report_validation")
    assert v["status"] == "passed_with_rejections" and v["narrative"]["rejected"][0]["reason_codes"] == ["unsupported_claim"]
    assert "90%" not in result.files["report"].read_text(encoding="utf-8")


def test_hypotheses_skipped_without_gaps_and_report_still_built(tmp_path, settings):
    recs, ver = records()
    script = [question_output(), *evidence_items(recs), {"gaps": [gap(supporting_evidence_ids=[])]}, narrative()]
    result, client = run(tmp_path, settings, script, recs=recs, ver=ver)
    assert client.remaining == 0
    assert load(result, "hypotheses")["status"] == "skipped"
    assert load(result, "hypotheses")["skip_reason"] == "no_accepted_gaps"
    report = result.files["report"].read_text(encoding="utf-8")
    assert "No validated research gaps." in report and "No validated candidate hypotheses." in report


def test_no_accepted_evidence_skips_all_synthesis(tmp_path, settings):
    recs, ver = records()
    script = [question_output(), {"items": []}, {"items": []}]
    result, client = run(tmp_path, settings, script, recs=recs, ver=ver)
    assert client.remaining == 0 and len(client.requests) == 3
    for name in ("gaps", "hypotheses"):
        assert load(result, name)["skip_reason"] in {"no_accepted_evidence", "no_accepted_gaps"}
    v = load(result, "report_validation")
    assert v["narrative"]["status"] == "skipped" and v["citations"] == []
    assert "No accepted evidence records." in result.files["report"].read_text(encoding="utf-8")


def test_budget_exhaustion_stops_synthesis_and_report_is_code_only(tmp_path, settings):
    recs, ver = records()
    result, client = run(tmp_path, settings, full_script(recs), recs=recs, ver=ver,
                         tracker=tracker(max_attempts=4))
    assert len(client.requests) == 4                 # question, 2 extraction, gaps
    hyps = load(result, "hypotheses")
    assert hyps["status"] == "failed" and hyps["errors"][0]["error_type"] == "budget_exhausted"
    v = load(result, "report_validation")
    assert v["narrative"]["status"] == "skipped" and v["narrative"]["skip_reason"] == "budget_exhausted"
    report = result.files["report"].read_text(encoding="utf-8")
    assert "Model stages stopped early: budget\\_exhausted" in report and "gap_01" in report


def test_failed_gaps_call_is_redacted_and_recorded(tmp_path, settings):
    recs, ver = records()
    script = [question_output(), *evidence_items(recs), "not json {", {"gaps": "nope"}, narrative()]
    result, _ = run(tmp_path, settings, script, recs=recs, ver=ver)
    gaps = load(result, "gaps")
    assert gaps["status"] == "failed" and gaps["repair_attempted"] and gaps["attempts"] == 2
    calls = load(result, "model_calls")
    entries = calls["entries"] if isinstance(calls, dict) else calls
    gap_entries = [e for e in entries if e["stage"] in ("gaps", "gaps:repair")]
    assert len(gap_entries) == 2 and all(e["redaction_mode"] == "safe_diagnostics" for e in gap_entries)
    summary = json.loads(gap_entries[1]["response_text"])
    assert summary["synthesis_stage"] == "gaps" and "call_record_id" not in summary
    assert summary["note"].startswith("response failed validation")
    assert "nope" not in json.dumps(gap_entries)
