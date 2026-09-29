"""v0.3 M3: model_calls.json naming — M3 entries never misuse call_record_id; M2 entries are unchanged."""

from __future__ import annotations

import json
import re

import pytest

from conftest import Sleeper, mock_client
from m2_support import item, no_throttle, question_output, tracker
from m3_support import FIXED, QUESTION, QUOTE_1, api, evidence_items, full_script, gap, hypothesis, load, narrative, \
    paragraph, records, run
from sciforge.llm.fake import FakeModelClient
from sciforge.model_pipeline import run_model_pipeline

REC_RE = re.compile(r"^rec_[0-9a-f]{16}$")
M3_STAGES = {"gaps", "hypotheses", "report"}
# the M2 extraction safe-summary key set (validated envelope), fixed since M2
M2_SUMMARY_KEYS = {"redaction_mode", "call_record_id", "response_sha256", "response_length", "parse_status", "items"}


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SCIFORGE_DEBUG_KEEP_REJECTED_RAW", raising=False)
    monkeypatch.delenv("SCIFORGE_MAX_SOURCE_CHARS", raising=False)


def entries(result):
    calls = load(result, "model_calls")
    return calls["entries"] if isinstance(calls, dict) else calls


def base_stage(entry):
    return entry["stage"].split(":")[0]


def m2_script_with_rejections(recs):
    r1, _ = recs
    ev = evidence_items(recs)
    ev[0] = {"items": [item(r1.record_id, QUOTE_1), item(r1.record_id, QUOTE_1, claim="Rose by 45%.")]}
    return [question_output(), *ev]


def m3_rejecting_script(recs):
    return [*m2_script_with_rejections(recs),
            "not json {",                                                         # gaps: repair
            {"gaps": [gap(), gap(supporting_evidence_ids=["ev_0999"])]},
            {"hypotheses": [hypothesis(), hypothesis(research_gap_ids=[])]},
            narrative(paragraph(text="Rose by 70% [ev_0001]."))]


def check_no_misused_call_record_id(node):
    if isinstance(node, dict):
        if "call_record_id" in node:
            assert node["call_record_id"] is None or REC_RE.match(node["call_record_id"]), node["call_record_id"]
        for v in node.values():
            check_no_misused_call_record_id(v)
    elif isinstance(node, list):
        for v in node:
            check_no_misused_call_record_id(v)


@pytest.mark.parametrize("debug", [False, True])
def test_m3_entries_never_have_non_rec_call_record_id(tmp_path, settings, debug):
    recs, ver = records()
    result, client = run(tmp_path, settings, m3_rejecting_script(recs), recs=recs, ver=ver,
                         debug_keep_rejected_raw=debug)
    assert client.remaining == 0
    m3 = [e for e in entries(result) if base_stage(e) in M3_STAGES]
    assert {base_stage(e) for e in m3} == M3_STAGES and any(e["stage"] == "gaps:repair" for e in m3)
    for e in m3:
        assert e.get("redaction_mode") == "safe_diagnostics"
        check_no_misused_call_record_id(e)
        summary = json.loads(e["response_text"])
        assert "call_record_id" not in summary
        # the stage is recorded explicitly, matching the entry-level stage
        assert summary["synthesis_stage"] == base_stage(e)
        assert summary["input_evidence_ids"] == ["ev_0001", "ev_0002"]
        assert summary["input_source_record_ids"] == sorted(r.record_id for r in recs)
        assert all(REC_RE.match(r) for r in summary["input_source_record_ids"])
    check_no_misused_call_record_id(entries(result))
    if debug:
        data = json.loads((result.run_dir / "debug" / "rejected_raw.json").read_text(encoding="utf-8"))
        outs = data["synthesis_unredacted_model_outputs"]
        assert outs and all("call_record_id" not in o and o["synthesis_stage"] in M3_STAGES for o in outs)


def test_clean_m3_calls_are_not_redacted_and_have_no_call_record_id(tmp_path, settings):
    recs, ver = records()
    result, _ = run(tmp_path, settings, full_script(recs), recs=recs, ver=ver)
    for e in entries(result):
        if base_stage(e) in M3_STAGES:
            assert "redaction_mode" not in e and "call_record_id" not in e


def test_m2_extraction_entry_format_unchanged(tmp_path, settings):
    """Extraction entries written by the M3 wrapper are identical to those of the M2-only pipeline."""
    recs, ver = records()
    m3_result, _ = run(tmp_path / "m3", settings, m3_rejecting_script(recs), recs=recs, ver=ver)
    m2_result = run_model_pipeline(QUESTION, recs, ver, model_client=FakeModelClient(m2_script_with_rejections(recs)),
                                   settings=settings, tracker=tracker(), output_dir=tmp_path / "m2",
                                   http_client=mock_client(api()), pubmed_limiter=no_throttle(),
                                   crossref_limiter=no_throttle(), sleep=Sleeper(), now=lambda: FIXED)
    m2_calls = json.loads(m2_result.files["model_calls"].read_text(encoding="utf-8"))
    m2_entries = m2_calls["entries"] if isinstance(m2_calls, dict) else m2_calls
    m3_m2_part = [e for e in entries(m3_result) if base_stage(e) not in M3_STAGES]
    assert m3_m2_part == m2_entries
    redacted = [e for e in m2_entries if e.get("redaction_mode") == "safe_diagnostics"]
    assert len(redacted) == 1 and redacted[0]["stage"] == "extraction"
    summary = json.loads(redacted[0]["response_text"])
    assert set(summary) == M2_SUMMARY_KEYS
    assert summary["call_record_id"] == recs[0].record_id and "synthesis_stage" not in summary
    # the M2 run files (source_texts, question, evidence) are byte-identical
    for name in ("source_texts", "question", "evidence"):
        assert m3_result.files[name].read_bytes() == m2_result.files[name].read_bytes()
