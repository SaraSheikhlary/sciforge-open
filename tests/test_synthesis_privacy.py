"""v0.3 M3: rejected gap / hypothesis / report-narrative text never reaches normal outputs (bytes search)."""

from __future__ import annotations

import hashlib
import json

import pytest

from m2_support import question_output
from m3_support import evidence_items, full_script, gap, hypothesis, load, narrative, paragraph, records, run
from sciforge.output_guard import find_leaks, normal_output_files


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SCIFORGE_DEBUG_KEEP_REJECTED_RAW", raising=False)
    monkeypatch.delenv("SCIFORGE_MAX_SOURCE_CHARS", raising=False)


def s(tag: str) -> str:
    # includes Markdown-significant characters on purpose (report.md escapes them)
    return f"Sentinel {tag} QZXV_{hashlib.sha256(tag.encode()).hexdigest()[:10]}*"


def rejected_scenario():
    """Script with rejected gap, hypothesis and narrative items carrying unique sentinels."""
    recs, ver = records()
    needles: list[str] = []

    def mark(tag):
        needles.append(s(tag))
        return s(tag)

    gaps = {"gaps": [
        gap(),
        # unsupported number
        gap(gap_statement=mark("gap-number") + " with 77% effect", why_unresolved=mark("gap-why-1")),
        # unknown evidence id + invented bibliographic fields
        gap(gap_statement=mark("gap-unknown"), why_unresolved=mark("gap-why-2"), supporting_evidence_ids=["ev_0999"],
            title=mark("gap-title"), doi="10.99999/qzxv.gap.doi"),
        # identifier in text
        gap(gap_statement=mark("gap-ident") + " https://qzxv.example/gap", why_unresolved=mark("gap-why-3")),
    ]}
    needles += ["10.99999/qzxv.gap.doi", "https://qzxv.example/gap"]
    hyps = {"hypotheses": [
        hypothesis(),
        hypothesis(statement=mark("hyp-nogap"), rationale=mark("hyp-rat-1"), research_gap_ids=[],
                   predicted_observable_outcome=mark("hyp-out-1"), assumptions=[mark("hyp-ass-1")]),
        hypothesis(statement="It is established that " + mark("hyp-fact"), rationale=mark("hyp-rat-2"),
                   predicted_observable_outcome=mark("hyp-out-2"), assumptions=[]),
        hypothesis(statement=mark("hyp-noev"), supporting_evidence_ids=[], rationale=mark("hyp-rat-3"),
                   journal=mark("hyp-journal"), predicted_observable_outcome=mark("hyp-out-3")),
    ]}
    narr = narrative(
        paragraph(text=mark("par-number") + " rose by 88% [ev_0001]."),
        paragraph(text=mark("par-unknown") + " [ev_0555]."),
        paragraph(text=mark("par-noev"), evidence_ids=[]),
        paragraph(section="conflicting_evidence", label="conflicting", text=mark("par-src") + " [ev_0001]",
                  source_ids=["rec_ffffffffffffffff"]),
        paragraph(text=mark("par-bib") + " [ev_0001]", authors=[mark("par-author")], pmid="99999999"),
    )
    return recs, ver, full_script(recs, gaps=gaps, hyps=hyps, narr=narr), needles


def test_rejected_synthesis_text_absent_from_all_normal_outputs(tmp_path, settings):
    recs, ver, script, needles = rejected_scenario()
    result, client = run(tmp_path, settings, script, recs=recs, ver=ver)
    assert client.remaining == 0
    names = {p.name for p in normal_output_files(result.run_dir)}
    assert {"report.md", "gaps.json", "hypotheses.json", "report_validation.json", "model_calls.json"} <= names
    assert not (result.run_dir / "debug").exists()
    assert find_leaks(result.run_dir, needles) == []
    # accepted items still present; rejected counted with safe diagnostics
    assert load(result, "gaps")["counts"]["accepted"] == 1 and load(result, "gaps")["counts"]["rejected"] == 3
    assert load(result, "hypotheses")["counts"]["rejected"] == 3
    v = load(result, "report_validation")
    assert v["counts"]["paragraphs_rejected"] == 5 and v["counts"]["paragraphs_accepted"] == 4
    codes = {c for r in v["narrative"]["rejected"] for c in r["reason_codes"]}
    assert codes == {"unsupported_claim", "unknown_evidence_id", "missing_evidence_reference", "unknown_source_id",
                     "fabricated_bibliographic_field"}
    # the three M3 calls with rejections are stored as safe summaries in model_calls.json
    calls = load(result, "model_calls")
    entries = calls["entries"] if isinstance(calls, dict) else calls
    redacted = {e["stage"] for e in entries if e.get("redaction_mode") == "safe_diagnostics"}
    assert {"gaps", "hypotheses", "report"} <= redacted
    summary = json.loads(next(e for e in entries if e["stage"] == "gaps")["response_text"])
    assert summary["parse_status"] == "validated_envelope"
    assert [i["status"] for i in summary["items"]] == ["accepted", "rejected", "rejected", "rejected"]
    assert summary["items"][0]["id"] == "gap_01"


def test_debug_mode_keeps_raw_only_under_debug(tmp_path, settings):
    recs, ver, script, needles = rejected_scenario()
    result, _ = run(tmp_path, settings, script, recs=recs, ver=ver, debug_keep_rejected_raw=True)
    assert find_leaks(result.run_dir, needles) == []
    debug = (result.run_dir / "debug" / "rejected_raw.json").read_text(encoding="utf-8")
    data = json.loads(debug)
    assert len(data["synthesis_rejected_items"]) == 11
    assert {o["stage"] for o in data["synthesis_unredacted_model_outputs"]} == {"gaps", "hypotheses", "report"}
    for needle in needles:
        assert needle in debug or json.dumps(needle)[1:-1] in debug
    assert load(result, "report_validation")["debug_keep_rejected_raw"] is True


def test_debug_env_var_enables_debug_file(tmp_path, settings, monkeypatch):
    monkeypatch.setenv("SCIFORGE_DEBUG_KEEP_REJECTED_RAW", "true")
    recs, ver, script, needles = rejected_scenario()
    result, _ = run(tmp_path, settings, script, recs=recs, ver=ver)
    assert (result.run_dir / "debug" / "rejected_raw.json").exists()
    assert find_leaks(result.run_dir, needles) == []


def test_malformed_synthesis_responses_with_repair_do_not_leak(tmp_path, settings):
    recs, ver = records()
    bad1 = json.dumps({"gaps": {"text": s("malformed-gap-1")}})
    bad2 = "{" + s("malformed-gap-2")
    bad3 = json.dumps({"paragraphs": [paragraph()], "extra": s("malformed-report")})
    script = [question_output(), *evidence_items(recs), bad1, bad2, bad3, narrative()]
    result, client = run(tmp_path, settings, script, recs=recs, ver=ver)
    assert client.remaining == 0
    assert load(result, "gaps")["status"] == "failed"
    v = load(result, "report_validation")
    assert v["narrative"]["repair_attempted"] is True and v["counts"]["paragraphs_accepted"] == 3
    # no hypotheses were accepted (gaps failed), so the next-steps paragraph citing hyp_01 is rejected
    assert v["narrative"]["rejected"][0]["reason_codes"] == ["unknown_hypothesis_id"]
    needles = [s("malformed-gap-1"), s("malformed-gap-2"), s("malformed-report")]
    assert find_leaks(result.run_dir, needles) == []
    calls = load(result, "model_calls")
    entries = calls["entries"] if isinstance(calls, dict) else calls
    repair = next(e for e in entries if e["stage"] == "gaps:repair")
    echoed = [m for m in repair["request"]["messages"] if m["role"] == "assistant"]
    assert echoed and all(json.loads(m["content"])["echoed_model_output"] == "[redacted]" for m in echoed)


def test_markdown_escaped_leak_is_detected(tmp_path):
    # guard self-test: a needle rendered through report.md's escaping is still found
    from sciforge.stages.report import _md

    (tmp_path / "report.md").write_text(f"- {_md(s('guard'))}\n", encoding="utf-8")
    assert find_leaks(tmp_path, [s("guard")]) == [("report.md", s("guard"))]
