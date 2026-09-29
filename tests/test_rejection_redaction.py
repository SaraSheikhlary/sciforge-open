"""v0.3 M2 safe rejection records, model_calls.json redaction, and the opt-in raw debug file."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from conftest import Sleeper, mock_client
from m2_support import BIB, STRUCTURED_ABSTRACT_TEXT, STRUCTURED_ABSTRACT_XML, Api, item, no_throttle, question_output, record, tracker, verified
from sciforge.config import ConfigError
from sciforge.llm.fake import FakeModelClient
from sciforge.model_pipeline import debug_keep_rejected_raw_from_env, run_model_pipeline
from sciforge.stages.validation import raw_item_sha256, redact_model_output_text, rejection_record, validate_item

REC = record()
QUOTE = "Shear exposure increased P-selectin expression by 40% compared with static controls."
INVENTED_DOI = "10.9999/made.up.2020"
INVENTED_TITLE = "An Entirely Invented Paper About Platelets"
FIXED = datetime(2026, 9, 28, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SCIFORGE_DEBUG_KEEP_REJECTED_RAW", raising=False)
    monkeypatch.delenv("SCIFORGE_MAX_SOURCE_CHARS", raising=False)


def reject(raw):
    _, reasons, _ = validate_item(raw, request_texts={REC.record_id: STRUCTURED_ABSTRACT_TEXT},
                                  supplied_ids={REC.record_id})
    return rejection_record(raw, reasons, call_record_id=REC.record_id, item_index=3, supplied_ids={REC.record_id})


# ------------------------------------------------------------------ rejection records

def test_bibliographic_rejection_record_has_no_values():
    raw = item(REC.record_id, QUOTE, title=INVENTED_TITLE, doi=INVENTED_DOI,
               claim="Invented claim text citing https://example.org/paper")
    rec = reject(raw)
    blob = json.dumps(rec)
    for secret in (INVENTED_TITLE, INVENTED_DOI, "Invented claim text", "example.org", QUOTE):
        assert secret not in blob
    assert rec["reasons"][0] == {"code": "bibliographic_field",
                                 "detail": "model attempted to return bibliographic fields (values not recorded)",
                                 "keys": ["doi", "title"]}
    d = rec["diagnostics"]
    assert d["raw_item_sha256"] == raw_item_sha256(raw) and d["bibliographic_keys"] == ["doi", "title"]
    assert d["other_field_count"] == 2 and d["field_lengths"]["quote"] == len(QUOTE) and d["quote_length"] == len(QUOTE)
    assert rec["validation_status"] == "rejected_deterministic" and rec["support"]["final_status"] == "rejected"
    assert rec["raw_output_stored"] is False and rec["source_record_id"] == REC.record_id


def test_unknown_record_id_hashed_unless_opaque_format():
    rec = reject(item("Smith et al. 2020, doi:10.1/x", QUOTE))
    r = rec["reasons"][0]
    assert r["code"] == "unknown_record_id" and r["value"] == "[redacted]" and len(r["value_sha256"]) == 64
    assert "Smith" not in json.dumps(rec) and rec["source_record_id"] is None
    assert reject(item("rec_deadbeefdeadbeef", QUOTE))["reasons"][0]["value"] == "rec_deadbeefdeadbeef"


def test_unsupported_enum_value_echoed_only_from_safe_allowlist():
    assert reject(item(REC.record_id, QUOTE, evidence_category="proven"))["reasons"][0]["value"] == "proven"
    rec = reject(item(REC.record_id, QUOTE, evidence_category="per Smith (Nature, 2020)"))
    assert rec["reasons"][0]["value"] == "[redacted]" and "Smith" not in json.dumps(rec)
    assert reject(item(REC.record_id, QUOTE, confidence="very high"))["reasons"][0]["value"] == "very high"


def test_unexpected_keys_sanitised():
    rec = reject(item(REC.record_id, QUOTE, notes="x", **{"https://evil.example/x": 1}))
    r = rec["reasons"][0]
    assert r["code"] == "unexpected_field" and r["keys"] == ["[redacted]", "notes"] and r["count"] == 2
    assert "evil" not in json.dumps(rec)


def test_numeric_rejection_detail_is_value_and_unit_only():
    rec = reject(item(REC.record_id, QUOTE, claim="P-selectin rose by 45% in stressed platelets."))
    assert rec["reasons"] == [{"code": "number_not_in_quote", "detail": "claim value not found in the quote",
                               "field": "claim", "value": "45", "unit": "%"}]
    assert "stressed platelets" not in json.dumps(rec)


def test_malformed_item_record():
    rec = reject("free text with doi 10.1234/abc")
    assert rec["diagnostics"] == {"raw_item_sha256": raw_item_sha256("free text with doi 10.1234/abc"),
                                  "item_type": "str"}
    assert "10.1234" not in json.dumps(rec)


# ------------------------------------------------------------------ text redaction

def test_redact_model_output_text_json_and_non_json():
    text = json.dumps({"items": [{"claim": "see doi:10.1234/abc and https://x.org", "title": "T", "authors": ["A"],
                                  "quote": "PMID: 12345 PMC999999"}]})
    out = redact_model_output_text(text)
    assert "10.1234" not in out and "x.org" not in out and '"T"' not in out and '"A"' not in out
    assert "12345" not in out and "PMC999999" not in out
    assert out.count("[redacted:bibliographic]") == 2 and "[redacted:identifier]" in out
    broken = '{"items": [{"title": "Invented", "doi": "10.5/x", "claim": "arXiv:2101.00001"'
    out2 = redact_model_output_text(broken)
    assert "Invented" not in out2 and "10.5/x" not in out2 and "2101.00001" not in out2
    assert redact_model_output_text(None) is None


def test_debug_env_parsing():
    assert debug_keep_rejected_raw_from_env({}) is False
    assert debug_keep_rejected_raw_from_env({"SCIFORGE_DEBUG_KEEP_REJECTED_RAW": "true"}) is True
    with pytest.raises(ConfigError):
        debug_keep_rejected_raw_from_env({"SCIFORGE_DEBUG_KEEP_REJECTED_RAW": "maybe"})


# ------------------------------------------------------------------ pipeline: outputs + debug file

def pipeline(tmp_path, settings, script, **kw):
    rec = record()
    client = FakeModelClient(script)
    result = run_model_pipeline("Does shear activate platelets?", [rec], [verified(rec)], model_client=client,
                                settings=settings, tracker=tracker(), output_dir=tmp_path,
                                http_client=mock_client(Api(pubmed={BIB["pmid"]: STRUCTURED_ABSTRACT_XML})),
                                pubmed_limiter=no_throttle(), crossref_limiter=no_throttle(), sleep=Sleeper(),
                                now=lambda: FIXED, **kw)
    return rec, result


def bad_item(rec):
    return item(rec.record_id, QUOTE, title=INVENTED_TITLE, doi=INVENTED_DOI)


def all_text(result, exclude_debug=True):
    return {p.name: p.read_text(encoding="utf-8") for p in result.run_dir.rglob("*.json")
            if not (exclude_debug and "debug" in p.parts)}


def test_default_no_raw_output_anywhere_and_model_calls_redacted(tmp_path, settings):
    rec, result = pipeline(tmp_path, settings, [question_output(), {"items": [item(record().record_id, QUOTE),
                                                                             bad_item(record())]}])
    assert not (result.run_dir / "debug").exists() and "debug_rejected_raw" not in result.files
    for name, text in all_text(result).items():
        assert INVENTED_TITLE not in text and INVENTED_DOI not in text, name
    ev = json.loads(result.files["evidence"].read_text())
    assert ev["debug_keep_rejected_raw"] is False and ev["model_calls_redacted_entries"] == 1
    assert ev["counts"]["accepted"] == 1 and ev["rejected"][0]["raw_output_stored"] is False
    calls = json.loads(result.files["model_calls"].read_text())
    extraction = [c for c in calls if c["stage"] == "extraction"][0]
    assert extraction["model_output_redacted"] is True
    assert extraction["redaction_mode"] == "safe_diagnostics"
    summary = json.loads(extraction["response_text"])
    assert summary["parse_status"] == "validated_envelope"
    assert [i["status"] for i in summary["items"]] == ["accepted", "rejected"]
    assert summary["items"][0]["evidence_id"] == "ev_0001"
    assert summary["items"][1]["reason_codes"] == ["bibliographic_field"]
    assert QUOTE not in extraction["response_text"]  # no model free text at all
    question = [c for c in calls if c["stage"] == "question"][0]
    assert "model_output_redacted" not in question


def test_repair_echo_in_request_history_is_redacted(tmp_path, settings):
    broken = '{"items": [{"title": "' + INVENTED_TITLE + '", "doi": "' + INVENTED_DOI + '"'
    rec, result = pipeline(tmp_path, settings, [question_output(), broken, {"items": [item(record().record_id, QUOTE)]}])
    calls = json.loads(result.files["model_calls"].read_text())
    repair = [c for c in calls if c["stage"] == "extraction:repair"][0]
    assistant = [m for m in repair["request"]["messages"] if m["role"] == "assistant"]
    assert assistant and INVENTED_TITLE not in assistant[0]["content"]
    for name, text in all_text(result).items():
        assert INVENTED_TITLE not in text and INVENTED_DOI not in text, name


def test_clean_call_not_redacted(tmp_path, settings):
    _, result = pipeline(tmp_path, settings, [question_output(), {"items": [item(record().record_id, QUOTE)]}])
    calls = json.loads(result.files["model_calls"].read_text())
    assert not any(c.get("model_output_redacted") for c in calls)


def test_debug_opt_in_writes_raw_only_to_debug_file(tmp_path, settings, monkeypatch):
    monkeypatch.setenv("SCIFORGE_DEBUG_KEEP_REJECTED_RAW", "true")
    rec, result = pipeline(tmp_path, settings, [question_output(), {"items": [bad_item(record())]}])
    debug = result.run_dir / "debug" / "rejected_raw.json"
    assert result.files["debug_rejected_raw"] == debug and debug.exists()
    data = json.loads(debug.read_text())
    assert data["warning"].startswith("LOCAL DEBUG FILE")
    raw = data["rejected_items"][0]
    assert raw["model_output"]["title"] == INVENTED_TITLE and raw["model_output"]["doi"] == INVENTED_DOI
    ev = json.loads(result.files["evidence"].read_text())
    assert raw["raw_item_sha256"] == ev["rejected"][0]["diagnostics"]["raw_item_sha256"]
    assert INVENTED_DOI in data["unredacted_model_outputs"][0]["response_text"]
    for name, text in all_text(result).items():  # normal outputs stay clean even with debug on
        assert INVENTED_TITLE not in text and INVENTED_DOI not in text, name
    assert ev["debug_keep_rejected_raw"] is True


def test_debug_parameter_overrides_env_and_no_file_without_rejections(tmp_path, settings):
    _, result = pipeline(tmp_path, settings, [question_output(), {"items": [item(record().record_id, QUOTE)]}],
                         debug_keep_rejected_raw=True)
    assert not (result.run_dir / "debug").exists()
