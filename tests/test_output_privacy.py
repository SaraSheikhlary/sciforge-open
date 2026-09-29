"""v0.3 M2: rejected natural-language model output never reaches normal run outputs (bytes search)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pytest

from conftest import Sleeper, mock_client
from m2_support import BIB, STRUCTURED_ABSTRACT_XML, Api, item, no_throttle, question_output, record, tracker, verified
from sciforge.llm.fake import FakeModelClient
from sciforge.model_pipeline import run_model_pipeline
from sciforge.output_guard import find_leaks, normal_output_files, rejected_text_values

QUOTE = "Shear exposure increased P-selectin expression by 40% compared with static controls."
FIXED = datetime(2026, 9, 28, 12, 0, 0, tzinfo=timezone.utc)
TEXT_FIELDS = ("claim", "quote", "finding", "methods", "limitations", "relevance")


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SCIFORGE_DEBUG_KEEP_REJECTED_RAW", raising=False)
    monkeypatch.delenv("SCIFORGE_MAX_SOURCE_CHARS", raising=False)


def sentinels(tag: str, fields=TEXT_FIELDS) -> dict[str, str]:
    return {f: f"Sentinel {tag} {f} text QZXV{hashlib.sha256(f"{tag}|{f}".encode()).hexdigest()[:8]}" for f in fields}


def scenario(name: str, rid: str) -> tuple[dict, list[str]]:
    """(raw rejected item, sentinel strings that must never leak)."""
    s = sentinels(name)
    if name == "bibliographic":
        raw = item(rid, s["quote"], **{k: v for k, v in s.items() if k != "quote"},
                   title=f"Sentinel {name} invented title QZXVTITLE01", doi="10.99999/qzxv.sentinel.doi01")
        return raw, [*s.values(), raw["title"], raw["doi"]]
    if name == "non_exact_quote":
        return item(rid, s["quote"], **{k: v for k, v in s.items() if k != "quote"}), list(s.values())
    if name == "numeric":
        # the quote must be exact for the numeric check to run; it is legitimately in source_texts.json
        s = {k: v for k, v in s.items() if k != "quote"}
        s["claim"] = f"{s['claim']} rose by 45%"
        return item(rid, QUOTE, **s), list(s.values())
    if name == "unknown_id":
        bad_id = "Sentinel unknown id QZXVID0001"
        return item(bad_id, s["quote"], **{k: v for k, v in s.items() if k != "quote"}), [*s.values(), bad_id]
    if name == "unsupported_category":
        cat = "Sentinel category QZXVCAT0001"
        return (item(rid, QUOTE, evidence_category=cat, **{k: v for k, v in s.items() if k != "quote"}),
                [*[v for k, v in s.items() if k != "quote"], cat])
    raise AssertionError(name)


def run(tmp_path, script, **kw):
    rec = record()
    client = FakeModelClient(script)
    result = run_model_pipeline("Does shear activate platelets?", [rec], [verified(rec)], model_client=client,
                                settings=kw.pop("settings"), tracker=tracker(), output_dir=tmp_path,
                                http_client=mock_client(Api(pubmed={BIB["pmid"]: STRUCTURED_ABSTRACT_XML})),
                                pubmed_limiter=no_throttle(), crossref_limiter=no_throttle(), sleep=Sleeper(),
                                now=lambda: FIXED, **kw)
    return result, client


SCENARIOS = ["bibliographic", "non_exact_quote", "numeric", "unknown_id", "unsupported_category"]


@pytest.mark.parametrize("name", SCENARIOS)
def test_rejected_text_never_in_normal_outputs(tmp_path, settings, name):
    rid = record().record_id
    raw, needles = scenario(name, rid)
    result, client = run(tmp_path, [question_output(), {"items": [item(rid, QUOTE), raw]}], settings=settings)
    assert client.remaining == 0
    ev = json.loads(result.files["evidence"].read_text())
    assert ev["counts"]["accepted"] == 1 and ev["counts"]["rejected"] == 1
    assert {f.name for f in normal_output_files(result.run_dir)} == {
        "source_texts.json", "question.json", "evidence.json", "model_calls.json"}
    assert find_leaks(result.run_dir, needles) == []
    assert not (result.run_dir / "debug").exists()
    calls = json.loads(result.files["model_calls"].read_text())
    ext = [c for c in calls if c["stage"] == "extraction"][0]
    assert ext["model_output_redacted"] is True and ext["redaction_mode"] == "safe_diagnostics"
    summary = json.loads(ext["response_text"])
    assert summary["items"][1]["status"] == "rejected" and summary["items"][1]["reason_codes"]


@pytest.mark.parametrize("name", SCENARIOS)
def test_with_debug_sentinels_appear_only_under_debug(tmp_path, settings, name):
    rid = record().record_id
    raw, needles = scenario(name, rid)
    result, _ = run(tmp_path, [question_output(), {"items": [raw]}], settings=settings, debug_keep_rejected_raw=True)
    assert find_leaks(result.run_dir, needles) == []
    debug = (result.run_dir / "debug" / "rejected_raw.json").read_bytes()
    for n in needles:
        assert json.dumps(n)[1:-1].encode() in debug, n


def malformed(tag: str) -> tuple[str, list[str]]:
    s = sentinels(tag)
    text = '{"items": [{"claim": "%s", "quote": "%s", "finding": "%s", "methods": "%s", "limitations": "%s", ' \
           '"relevance": "%s", "title": "Sentinel %s title QZXVT", "doi": "10.99999/qzxv.%s"' % (
               *s.values(), tag, tag)
    return text, [*s.values(), f"Sentinel {tag} title QZXVT", f"10.99999/qzxv.{tag}"]


def test_malformed_response_repair_fails_nothing_leaks(tmp_path, settings):
    bad1, n1 = malformed("first")
    bad2, n2 = malformed("second")
    result, client = run(tmp_path, [question_output(), bad1, bad2], settings=settings)
    assert len(client.requests) == 3 and client.requests[2].stage == "extraction:repair"
    assert find_leaks(result.run_dir, n1 + n2) == []
    calls = json.loads(result.files["model_calls"].read_text())
    ext = [c for c in calls if c["stage"].startswith("extraction")]
    assert len(ext) == 2 and all(c["redaction_mode"] == "safe_diagnostics" for c in ext)
    assert json.loads(ext[0]["response_text"])["parse_status"] == "invalid_json"
    echo = [m for m in ext[1]["request"]["messages"] if m["role"] == "assistant"][0]
    assert json.loads(echo["content"])["echoed_model_output"] == "[redacted]"


def test_malformed_response_repair_succeeds_nothing_leaks(tmp_path, settings):
    bad, needles = malformed("repaired")
    rid = record().record_id
    result, _ = run(tmp_path, [question_output(), bad, {"items": [item(rid, QUOTE)]}], settings=settings)
    assert json.loads(result.files["evidence"].read_text())["counts"]["accepted"] == 1
    assert find_leaks(result.run_dir, needles) == []
    calls = json.loads(result.files["model_calls"].read_text())
    repair = [c for c in calls if c["stage"] == "extraction:repair"][0]
    assert json.loads(repair["response_text"])["items"][0]["evidence_id"] == "ev_0001"


def test_malformed_with_debug_only_in_debug(tmp_path, settings):
    bad1, n1 = malformed("dbg1")
    bad2, n2 = malformed("dbg2")
    result, _ = run(tmp_path, [question_output(), bad1, bad2], settings=settings, debug_keep_rejected_raw=True)
    assert find_leaks(result.run_dir, n1 + n2) == []
    debug = (result.run_dir / "debug" / "rejected_raw.json").read_bytes()
    for n in n1 + n2:
        assert json.dumps(n)[1:-1].encode() in debug


def test_debug_hashes_correlate_with_model_calls(tmp_path, settings):
    raw, _ = scenario("bibliographic", record().record_id)
    result, _ = run(tmp_path, [question_output(), {"items": [raw]}], settings=settings, debug_keep_rejected_raw=True)
    debug = json.loads((result.run_dir / "debug" / "rejected_raw.json").read_text())
    calls = json.loads(result.files["model_calls"].read_text())
    ext = [c for c in calls if c["stage"] == "extraction"][0]
    assert debug["unredacted_model_outputs"][0]["response_sha256"] == ext["response_sha256"]
    assert json.loads(ext["response_text"])["response_sha256"] == ext["response_sha256"]
    assert debug["rejected_items"][0]["raw_item_sha256"] == \
        json.loads(ext["response_text"])["items"][0]["diagnostics"]["raw_item_sha256"]


def test_scanner_covers_every_file_outside_debug(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "debug").mkdir(parents=True)
    (run_dir / "debug" / "rejected_raw.json").write_text('{"x": "needle-one"}')
    (run_dir / "report.md").write_text("# Report\nneedle-two appears here\n")
    (run_dir / "sub").mkdir()
    (run_dir / "sub" / "extra.txt").write_text('escaped \\"needle \\"three\\"')
    assert find_leaks(run_dir, ["needle-one", "needle-two", 'needle "three"', ""]) == [
        ("report.md", "needle-two"), ("sub/extra.txt", 'needle "three"')]


def test_rejected_text_values_helper():
    raw = [{"model_output": {"claim": "long enough claim text", "source_record_id": "rec_0123456789abcdef",
                             "n": 5, "x": ["another long string"], "short": "tiny"}}]
    assert rejected_text_values(raw) == ["long enough claim text", "another long string"]
