"""v0.3 M2 mocked end-to-end model pipeline (FakeModelClient + httpx.MockTransport; offline)."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone

import pytest

from conftest import FAKE_API_KEY, FAKE_EMAIL, FAKE_XAI_KEY, Sleeper, mock_client, model_env
from m2_support import (
    BIB, CROSSREF_JATS, CROSSREF_TEXT, STRUCTURED_ABSTRACT_TEXT, STRUCTURED_ABSTRACT_XML, Api, item, no_throttle,
    question_output, record, tracker, verified,
)
from sciforge.config import ModelSettings
from sciforge.llm.client import ModelAuthError
from sciforge.llm.fake import FakeModelClient
from sciforge.model_pipeline import run_model_pipeline

QUOTE = "Shear exposure increased P-selectin expression by 40% compared with static controls."
FIXED = datetime(2026, 9, 28, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _no_cap_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SCIFORGE_MAX_SOURCE_CHARS", raising=False)
    monkeypatch.delenv("SCIFORGE_DEBUG_KEEP_REJECTED_RAW", raising=False)


def run(tmp_path, settings, api, records, ver, script, **kw):
    client = FakeModelClient(script)
    kw.setdefault("tracker", tracker())
    result = run_model_pipeline("Does high shear stress directly activate human platelets?", records, ver,
                                model_client=client, settings=settings, output_dir=tmp_path,
                                http_client=mock_client(api), pubmed_limiter=no_throttle(),
                                crossref_limiter=no_throttle(), sleep=Sleeper(), now=lambda: FIXED, **kw)
    return result, client


def load(result, name):
    return json.loads(result.files[name].read_text(encoding="utf-8"))


def test_end_to_end_writes_files_with_sane_content(tmp_path, settings):
    pm = record()
    cr = record(pmid=None, doi="10.7777/red", source="crossref", title="Stored red cells")
    nope = record(pmid="8", doi=None)
    partial = record(pmid="9", doi=None)
    ver = [verified(pm), verified(cr), verified(nope), verified(partial, "partially_verified")]
    api = Api(pubmed={BIB["pmid"]: STRUCTURED_ABSTRACT_XML, "8": None, "9": "<AbstractText>X.</AbstractText>"},
              crossref={"10.7777/red": CROSSREF_JATS})
    cr_quote = "ex vivo rheometry showed a 25% loss after 35 days."
    script = [question_output(),
              {"items": [item(pm.record_id, QUOTE), item(pm.record_id, "P-selectin rose sharply in all donors.")]},
              {"items": [item(cr.record_id, cr_quote, claim="Stored red cells lose deformability.",
                              finding="25% loss after 35 days.", methods=None, doi="10.9999/invented")]}]
    before = copy.deepcopy([r.model_dump() for r in (pm, cr, nope, partial)])
    result, client = run(tmp_path, settings, api, [pm, cr, nope, partial], ver, script)

    assert result.run_dir.parent == tmp_path and result.run_dir.name == "20260928T120000Z"
    assert sorted(p.name for p in result.run_dir.iterdir()) == [
        "evidence.json", "model_calls.json", "question.json", "source_texts.json"]
    assert client.remaining == 0 and len(client.requests) == 3

    src = load(result, "source_texts")
    assert src["sent_to_model"] == [pm.record_id, cr.record_id]
    assert src["counts"] == {"no_abstract": 1, "not_eligible": 1, "ok": 2}
    by_id = {s["record_id"]: s for s in src["sources"]}
    assert by_id[pm.record_id]["access_level"] == "pubmed_abstract"
    assert by_id[pm.record_id]["source_text"] == STRUCTURED_ABSTRACT_TEXT
    assert by_id[cr.record_id]["access_level"] == "crossref_abstract"
    assert by_id[cr.record_id]["source_text"] == CROSSREF_TEXT
    assert by_id[nope.record_id]["status"] == "no_abstract"
    assert by_id[partial.record_id]["status"] == "not_eligible"
    assert src["eligibility_policy"] == "verified" and src["max_source_chars"] == 4000

    q = load(result, "question")
    assert q["status"] == "ok" and q["definition"]["research_question"].startswith("Does high shear")

    ev = load(result, "evidence")
    assert ev["counts"]["accepted"] == 1 and ev["counts"]["rejected"] == 2
    assert ev["accepted"][0]["evidence_id"] == "ev_0001" and ev["accepted"][0]["quote"] == QUOTE
    assert ev["accepted"][0]["abstract_only"] is True
    reasons = sorted(r["reasons"][0]["code"] for r in ev["rejected"])
    assert reasons == ["bibliographic_field", "quote_not_in_source"]
    assert ev["question_definition_used"] is True and ev["budget"]["used"]["attempts"] == 3

    calls = load(result, "model_calls")
    assert [c["stage"] for c in calls] == ["question", "extraction", "extraction"]
    assert all(c["entry_type"] == "attempt" and c["outcome"] == "success" for c in calls)

    # v0.2 records not modified; no bibliographic data sent to the model; no secrets on disk
    assert [r.model_dump() for r in (pm, cr, nope, partial)] == before
    sent = "\n".join(m.content for r in client.requests for m in r.messages)
    for value in (BIB["title"], BIB["journal"], BIB["doi"], BIB["pmid"], "10.7777/red", "Stored red cells",
                  *BIB["authors"]):
        assert value not in sent
    for path in result.files.values():
        text = path.read_text(encoding="utf-8")
        assert FAKE_API_KEY not in text and FAKE_EMAIL not in text


def test_efetch_failure_recorded_and_pipeline_continues(tmp_path, settings):
    pm = record(doi=None)
    cr = record(pmid=None, doi="10.7777/red", source="crossref")
    api = Api(efetch_status=503, crossref={"10.7777/red": CROSSREF_JATS})
    script = [question_output(), {"items": [item(cr.record_id, "ex vivo rheometry showed a 25% loss after 35 days.",
                                                        finding="A 25% loss after 35 days.", methods=None)]}]
    result, client = run(tmp_path, settings, api, [pm, cr], [verified(pm), verified(cr)], script)
    src = load(result, "source_texts")
    assert {s["record_id"]: s["status"] for s in src["sources"]} == {pm.record_id: "fetch_failed", cr.record_id: "ok"}
    assert src["errors"] and load(result, "evidence")["counts"]["accepted"] == 1


def test_no_usable_sources_skips_extraction(tmp_path, settings):
    pm = record(doi=None)
    result, client = run(tmp_path, settings, Api(pubmed={BIB["pmid"]: None}), [pm], [verified(pm)],
                         [question_output()])
    ev = load(result, "evidence")
    assert len(client.requests) == 1 and ev["counts"]["sources_sent"] == 0
    assert ev["note"].startswith("No source text available")


def test_question_failure_falls_back_to_raw_question(tmp_path, settings):
    pm = record()
    script = ["bad", "bad again", {"items": [item(pm.record_id, QUOTE)]}]
    result, client = run(tmp_path, settings, Api(pubmed={BIB["pmid"]: STRUCTURED_ABSTRACT_XML}), [pm],
                         [verified(pm)], script)
    assert load(result, "question")["status"] == "failed"
    ev = load(result, "evidence")
    assert ev["question_definition_used"] is False and ev["counts"]["accepted"] == 1
    payload = json.loads(client.requests[2].messages[0].content)
    assert payload["question_definition"] == {"research_question": "Does high shear stress directly activate human platelets?"}


def test_auth_error_stops_model_calls(tmp_path, settings):
    pm = record()
    result, client = run(tmp_path, settings, Api(pubmed={BIB["pmid"]: STRUCTURED_ABSTRACT_XML}), [pm],
                         [verified(pm)], [ModelAuthError("HTTP 401", http_status=401)])
    ev = load(result, "evidence")
    assert len(client.requests) == 1 and ev["stop_reason"] == "auth_error"
    assert ev["calls"][0]["status"] == "skipped"


def test_budget_max_sources_honoured(tmp_path, settings):
    recs = [record(pmid=str(700 + i), doi=None) for i in range(3)]
    api = Api(pubmed={r.pmid: STRUCTURED_ABSTRACT_XML for r in recs})
    script = [question_output()] + [{"items": []}] * 2
    result, client = run(tmp_path, settings, api, recs, [verified(r) for r in recs], script,
                         tracker=tracker(max_sources=2))
    src = load(result, "source_texts")
    assert src["sent_to_model"] == [recs[0].record_id, recs[1].record_id]
    assert src["counts"] == {"ok": 2, "source_limit": 1} and len(client.requests) == 3
    assert load(result, "evidence")["budget"]["sources_limited"] is True


def test_partially_verified_opt_in_via_model_settings_is_labelled(tmp_path, settings):
    pm = record(doi=None)
    ms = ModelSettings.from_env(model_env(SCIFORGE_MODEL_ELIGIBILITY="verified_or_partial"))
    script = [question_output(), {"items": [item(pm.record_id, QUOTE)]}]
    result, _ = run(tmp_path, settings, Api(pubmed={BIB["pmid"]: STRUCTURED_ABSTRACT_XML}), [pm],
                    [verified(pm, "partially_verified")], script, tracker=None, model_settings=ms)
    src = load(result, "source_texts")
    assert src["eligibility_policy"] == "verified_or_partial"
    assert src["sources"][0]["eligibility"] == "partially_verified_opt_in"
    assert load(result, "evidence")["accepted"][0]["source_identity"] == "partially_verified"
    for path in result.files.values():
        assert FAKE_XAI_KEY not in path.read_text(encoding="utf-8")


def test_max_source_chars_from_env_used(tmp_path, settings, monkeypatch):
    monkeypatch.setenv("SCIFORGE_MAX_SOURCE_CHARS", "200")
    pm = record(doi=None)
    result, client = run(tmp_path, settings, Api(pubmed={BIB["pmid"]: STRUCTURED_ABSTRACT_XML}), [pm],
                         [verified(pm)], [question_output(), {"items": [item(pm.record_id, QUOTE)]}])
    s = load(result, "source_texts")["sources"][0]
    assert s["truncated"] is True and s["text_chars"] <= 200
    # the RESULTS sentence was cut off by the cap, so its quote is no longer in the capped text
    assert load(result, "evidence")["rejected"][0]["reasons"][0]["code"] == "quote_not_in_source"


def test_requires_budget(tmp_path, settings):
    with pytest.raises(ValueError):
        run_model_pipeline("q", [], [], model_client=FakeModelClient(), settings=settings, output_dir=tmp_path)
    with pytest.raises(ValueError):
        run_model_pipeline("  ", [], [], model_client=FakeModelClient(), settings=settings, tracker=tracker(),
                           output_dir=tmp_path)


def test_existing_run_dir_used(tmp_path, settings):
    target = tmp_path / "existing"
    result, _ = run(tmp_path, settings, Api(), [], [], [question_output()], run_dir=target)
    assert result.run_dir == target and (target / "evidence.json").exists()
