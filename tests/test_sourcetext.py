"""v0.3 M2 source-text layer: efetch/JATS parsing, cap, hashing, eligibility, budget, failures."""

from __future__ import annotations

import copy
import hashlib

import pytest

from conftest import FAKE_API_KEY, FAKE_EMAIL
from m2_support import (
    BIB, CROSSREF_JATS, CROSSREF_TEXT, PLAIN_ABSTRACT_TEXT, STRUCTURED_ABSTRACT_TEXT, STRUCTURED_ABSTRACT_XML,
    Api, efetch_xml, make_fetcher, no_throttle, record, tracker, verified,
)
from sciforge.config import ConfigError
from sciforge.http_utils import MalformedResponseError
from sciforge.sourcetext import (
    DEFAULT_MAX_SOURCE_CHARS, cap_text, jats_to_text, max_source_chars_from_env, parse_efetch_abstracts,
    prepare_source_texts, sha256_text,
)


@pytest.fixture(autouse=True)
def _no_cap_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SCIFORGE_MAX_SOURCE_CHARS", raising=False)


def prepare(api, settings, records, verification, **kw):
    fetcher = make_fetcher(api, settings)
    tr = kw.pop("tr", None) or tracker()
    batch = prepare_source_texts(records, verification, fetcher=fetcher, tracker=tr, pubmed_limiter=no_throttle(),
                                 crossref_limiter=no_throttle(), **kw)
    return batch, tr, fetcher


# ------------------------------------------------------------------ efetch XML parsing

def test_structured_abstract_labels_and_markup():
    out = parse_efetch_abstracts(efetch_xml([("123", STRUCTURED_ABSTRACT_XML)]))
    assert out == {"123": STRUCTURED_ABSTRACT_TEXT}


def test_unstructured_abstract():
    out = parse_efetch_abstracts(efetch_xml([("5", f"<AbstractText>{PLAIN_ABSTRACT_TEXT}</AbstractText>")]))
    assert out["5"] == PLAIN_ABSTRACT_TEXT


def test_nested_markup_and_whitespace_collapsed():
    xml = efetch_xml([("7", "<AbstractText>CO<sub>2</sub> levels  rose\n  in <b>all</b> <i>E. coli</i> "
                            "strains.</AbstractText>")])
    assert parse_efetch_abstracts(xml)["7"] == "CO2 levels rose in all E. coli strains."


def test_multiple_unlabelled_abstracttext_joined():
    xml = efetch_xml([("8", "<AbstractText>First part.</AbstractText><AbstractText>Second part.</AbstractText>")])
    assert parse_efetch_abstracts(xml)["8"] == "First part. Second part."


def test_missing_abstract_and_multiple_articles():
    xml = efetch_xml([("1", None), ("2", "<AbstractText>Text two.</AbstractText>"), ("3", "<AbstractText/>")])
    assert parse_efetch_abstracts(xml) == {"1": None, "2": "Text two.", "3": None}


def test_article_title_never_part_of_abstract():
    out = parse_efetch_abstracts(efetch_xml([("9", "<AbstractText>Body.</AbstractText>")]))
    assert "title" not in out["9"].lower()


@pytest.mark.parametrize("body", [
    "<PubmedArticleSet><PubmedArticle>",                      # malformed
    "<Other/>",                                                 # wrong root
    "",                                                         # empty
    '<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">]><PubmedArticleSet/>',  # entity declaration
])
def test_unsafe_or_malformed_xml_rejected(body):
    with pytest.raises(MalformedResponseError):
        parse_efetch_abstracts(body)


def test_public_doctype_line_is_allowed():
    assert parse_efetch_abstracts(efetch_xml([("4", "<AbstractText>A.</AbstractText>")], doctype=True)) == {"4": "A."}


# ------------------------------------------------------------------ Crossref JATS

def test_jats_stripped_to_plain_text():
    assert jats_to_text(CROSSREF_JATS) == CROSSREF_TEXT


def test_jats_section_titles_become_labels_and_entities_unescaped():
    jats = ("<jats:sec><jats:title>Background</jats:title><jats:p>A &amp; B &lt;5 mm.</jats:p></jats:sec>"
            "<jats:sec><jats:title>Results</jats:title><jats:p>H<jats:sub>2</jats:sub>O rose.</jats:p></jats:sec>")
    assert jats_to_text(jats) == "Background: A & B <5 mm. Results: H2O rose."


@pytest.mark.parametrize("value", [None, "", "   ", "<jats:p> </jats:p>", 42])
def test_jats_empty_is_none(value):
    assert jats_to_text(value) is None


# ------------------------------------------------------------------ cap + hash + config

def test_cap_truncates_on_word_boundary():
    text = "alpha beta gamma delta"
    capped, truncated = cap_text(text, 13)
    assert (capped, truncated) == ("alpha beta", True)
    assert cap_text(text, 100) == (text, False)
    assert cap_text("abcdefghij", 5) == ("abcde", True)  # no whitespace → hard cut


def test_sha256_exact_and_stable():
    assert sha256_text("abc") == hashlib.sha256(b"abc").hexdigest() == sha256_text("abc")
    assert sha256_text("abc ") != sha256_text("abc")


def test_max_source_chars_env():
    assert max_source_chars_from_env({}) == DEFAULT_MAX_SOURCE_CHARS == 4000
    assert max_source_chars_from_env({"SCIFORGE_MAX_SOURCE_CHARS": "1200"}) == 1200
    for bad in ("abc", "10", "999999"):
        with pytest.raises(ConfigError):
            max_source_chars_from_env({"SCIFORGE_MAX_SOURCE_CHARS": bad})


def test_cap_applied_in_source_text_with_hash_of_capped_text(settings):
    rec = record()
    api = Api(pubmed={BIB["pmid"]: STRUCTURED_ABSTRACT_XML})
    batch, _, _ = prepare(api, settings, [rec], [verified(rec)], max_source_chars=200)
    st = batch.sources[0]
    assert st.status == "ok" and st.truncated is True
    assert len(st.source_text) <= 200 and STRUCTURED_ABSTRACT_TEXT.startswith(st.source_text)
    assert st.original_chars == len(STRUCTURED_ABSTRACT_TEXT) and st.text_chars == len(st.source_text)
    assert st.sha256 == hashlib.sha256(st.source_text.encode()).hexdigest()
    assert not st.source_text.endswith(" ")


# ------------------------------------------------------------------ retrieval

def test_pubmed_abstract_retrieved_with_etiquette_and_redaction(settings):
    rec = record()
    api = Api(pubmed={BIB["pmid"]: STRUCTURED_ABSTRACT_XML})
    batch, _, fetcher = prepare(api, settings, [rec], [verified(rec)])
    st = batch.sources[0]
    assert st.status == "ok" and st.access_level == "pubmed_abstract" and st.origin == "pubmed"
    assert st.abstract_only is True and st.source_text == STRUCTURED_ABSTRACT_TEXT and not st.truncated
    assert st.eligibility == "verified" and st.verification_status == "verified"
    assert st.retrieved_at.endswith("Z")
    (req,) = api.efetch_requests()
    params = req.url.params
    assert params["db"] == "pubmed" and params["retmode"] == "xml" and params["id"] == BIB["pmid"]
    assert params["tool"] == "sciforge" and params["email"] == FAKE_EMAIL and params["api_key"] == FAKE_API_KEY
    logged = batch.to_json()["requests"][0]
    assert logged["stage"] == "source_text" and logged["params"]["api_key"] == "[REDACTED]"
    assert FAKE_API_KEY not in str(batch.to_json()) and FAKE_EMAIL not in str(batch.to_json())
    assert not api.crossref_requests()


def test_source_text_carries_no_bibliographic_fields(settings):
    rec = record()
    batch, _, _ = prepare(Api(pubmed={BIB["pmid"]: STRUCTURED_ABSTRACT_XML}), settings, [rec], [verified(rec)])
    dumped = batch.sources[0].model_dump()
    assert not {"title", "authors", "journal", "year", "doi", "pmid", "source_url"} & set(dumped)
    blob = str(batch.to_json())
    for value in (BIB["title"], BIB["journal"], BIB["doi"], *BIB["authors"]):
        assert value not in blob


def test_efetch_batches_ids(settings):
    recs = [record(pmid=str(1000 + i), doi=None) for i in range(4)]
    api = Api(pubmed={r.pmid: "<AbstractText>Text.</AbstractText>" for r in recs})
    batch, _, _ = prepare(api, settings, recs, [verified(r) for r in recs])
    assert len(api.efetch_requests()) == 1
    assert api.efetch_requests()[0].url.params["id"] == "1000,1001,1002,1003"
    assert [s.status for s in batch.sources] == ["ok"] * 4


def test_missing_abstract_is_no_abstract_and_not_usable(settings):
    rec = record(doi=None)
    batch, tr, _ = prepare(Api(pubmed={BIB["pmid"]: None}), settings, [rec], [verified(rec)])
    st = batch.sources[0]
    assert st.status == "no_abstract" and st.source_text is None and st.sha256 is None
    assert st.access_level == "not_accessed" and batch.usable == [] and tr.sources == 0


def test_crossref_abstract_for_doi_only_record(settings):
    rec = record(pmid=None, source="crossref")
    api = Api(crossref={BIB["doi"]: CROSSREF_JATS})
    batch, _, _ = prepare(api, settings, [rec], [verified(rec)])
    st = batch.sources[0]
    assert st.status == "ok" and st.access_level == "crossref_abstract" and st.origin == "crossref"
    assert st.source_text == CROSSREF_TEXT
    (req,) = api.crossref_requests()
    assert req.url.params["mailto"] == FAKE_EMAIL
    assert not api.efetch_requests()


def test_crossref_fallback_when_pubmed_has_no_abstract(settings):
    rec = record()
    api = Api(pubmed={BIB["pmid"]: None}, crossref={BIB["doi"]: CROSSREF_JATS})
    batch, _, _ = prepare(api, settings, [rec], [verified(rec)])
    assert batch.sources[0].access_level == "crossref_abstract"


def test_crossref_without_abstract_is_no_abstract(settings):
    rec = record(pmid=None, source="crossref")
    batch, _, _ = prepare(Api(crossref={BIB["doi"]: None}), settings, [rec], [verified(rec)])
    assert batch.sources[0].status == "no_abstract"


def test_efetch_failure_recorded_and_no_crossref_fallback(settings):
    rec = record()
    other = record(pmid=None, doi="10.7777/other", source="crossref")
    api = Api(efetch_status=500, crossref={"10.7777/other": CROSSREF_JATS, BIB["doi"]: CROSSREF_JATS})
    batch, _, _ = prepare(api, settings, [rec, other], [verified(rec), verified(other)])
    by_id = {s.record_id: s for s in batch.sources}
    assert by_id[rec.record_id].status == "fetch_failed"
    assert by_id[rec.record_id].error["http_status"] == 500
    assert by_id[other.record_id].status == "ok"  # continues with other sources
    assert len(api.efetch_requests()) == 1 + settings.max_retries
    assert [r.url.path for r in api.crossref_requests()] == ["/works/10.7777/other"]
    assert batch.to_json()["errors"]


def test_efetch_parse_error_recorded(settings):
    rec = record(doi=None)
    batch, _, _ = prepare(Api(efetch_body="<PubmedArticleSet><broken"), settings, [rec], [verified(rec)])
    assert batch.sources[0].status == "parse_error"
    assert batch.to_json()["requests"][0]["status"] == "error"


# ------------------------------------------------------------------ eligibility + budget

def test_partially_verified_excluded_by_default_and_labelled_when_opted_in(settings):
    ok, partial, bad = record(pmid="1"), record(pmid="2", doi=None), record(pmid="3", doi=None)
    ver = [verified(ok), verified(partial, "partially_verified"), verified(bad, "not_verified")]
    api = Api(pubmed={p: "<AbstractText>Text.</AbstractText>" for p in ("1", "2", "3")})
    batch, _, _ = prepare(api, settings, [ok, partial, bad], ver)
    status = {s.record_id: s for s in batch.sources}
    assert status[partial.record_id].status == "not_eligible"
    assert status[bad.record_id].status == "not_eligible"
    assert api.efetch_requests()[0].url.params["id"] == "1"
    assert batch.to_json()["eligibility_policy"] == "verified"

    api2 = Api(pubmed={p: "<AbstractText>Text.</AbstractText>" for p in ("1", "2", "3")})
    batch2, _, _ = prepare(api2, settings, [ok, partial, bad], ver, include_partially_verified=True)
    s2 = {s.record_id: s for s in batch2.sources}
    assert s2[partial.record_id].status == "ok"
    assert s2[partial.record_id].eligibility == "partially_verified_opt_in"
    assert s2[partial.record_id].verification_status == "partially_verified"
    assert s2[bad.record_id].status == "not_eligible"
    assert batch2.to_json()["eligibility_policy"] == "verified_or_partial"
    assert [s.record_id for s in batch2.usable] == [ok.record_id, partial.record_id]  # verified first


def test_missing_verification_and_non_opaque_ids_not_eligible(settings):
    rec = record()
    odd = record(pmid="77", doi=None).model_copy(update={"record_id": "custom-id"})
    batch, _, _ = prepare(Api(), settings, [rec, odd], [verified(odd)])
    assert [s.status for s in batch.sources] == ["not_eligible", "not_eligible"]
    assert batch.sources[1].error["message"].startswith("record_id is not an opaque")


def test_max_sources_honoured_with_deterministic_order(settings):
    doi_only = record(pmid=None, doi="10.7777/d", source="crossref")
    pm = [record(pmid=str(500 + i), doi=None) for i in range(4)]
    recs = [doi_only, *pm]
    api = Api(pubmed={r.pmid: "<AbstractText>Text.</AbstractText>" for r in pm}, crossref={"10.7777/d": CROSSREF_JATS})
    batch, tr, _ = prepare(api, settings, recs, [verified(r) for r in recs], tr=tracker(max_sources=3))
    assert [s.record_id for s in batch.usable] == [r.record_id for r in pm[:3]]  # PMID first, v0.2 order
    limited = [s for s in batch.sources if s.status == "source_limit"]
    assert {s.record_id for s in limited} == {pm[3].record_id, doi_only.record_id}
    assert tr.sources == 3 and tr.sources_limited is True
    assert api.efetch_requests()[0].url.params["id"] == "500,501,502"  # later records not fetched
    assert not api.crossref_requests()


def test_slot_freed_by_missing_abstract_goes_to_next_candidate(settings):
    recs = [record(pmid=str(600 + i), doi=None) for i in range(3)]
    api = Api(pubmed={"600": None, "601": "<AbstractText>B.</AbstractText>", "602": "<AbstractText>C.</AbstractText>"})
    batch, tr, _ = prepare(api, settings, recs, [verified(r) for r in recs], tr=tracker(max_sources=2))
    assert [s.record_id for s in batch.usable] == [recs[1].record_id, recs[2].record_id]
    assert tr.sources == 2


def test_v02_records_and_verification_not_mutated(settings):
    recs = [record(), record(pmid=None, doi="10.7777/x", source="crossref")]
    ver = [verified(recs[0]), verified(recs[1], "partially_verified")]
    before = (copy.deepcopy([r.model_dump() for r in recs]), copy.deepcopy([v.model_dump() for v in ver]))
    api = Api(pubmed={BIB["pmid"]: STRUCTURED_ABSTRACT_XML}, crossref={"10.7777/x": CROSSREF_JATS})
    prepare(api, settings, recs, ver, include_partially_verified=True, max_source_chars=200)
    assert ([r.model_dump() for r in recs], [v.model_dump() for v in ver]) == before
