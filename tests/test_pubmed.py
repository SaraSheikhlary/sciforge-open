"""PubMed client: parameters, parsing, malformed responses."""

import httpx
import pytest

from conftest import FAKE_API_KEY, FAKE_EMAIL, esearch_payload, esummary_doc, esummary_payload, json_response
from sciforge.config import Settings
from sciforge.http_utils import MalformedResponseError
from sciforge.pubmed import parse_esearch, parse_esummary, parse_pubmed_year, record_from_summary


def pubmed_handler(esearch, esummary):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("esearch.fcgi"):
            return esearch(request) if callable(esearch) else esearch
        if request.url.path.endswith("esummary.fcgi"):
            return esummary(request) if callable(esummary) else esummary
        return httpx.Response(500)
    return handler


def test_search_params_and_records(make_harness):
    h = make_harness(pubmed_handler(json_response(esearch_payload(["111", "222"], count=57)),
                                    json_response(esummary_payload([esummary_doc("111"),
                                                                    esummary_doc("222", doi=None, pubdate="Spring")]))))
    out = h.pubmed.search("platelet shear", 5, from_year=2015, to_year=2020)
    assert out.status == "ok" and out.total_hits == 57 and len(out.records) == 2
    es = h.requests[0].url.params
    assert es["db"] == "pubmed" and es["retmode"] == "json" and es["term"] == "platelet shear" and es["retmax"] == "5"
    assert es["datetype"] == "pdat" and es["mindate"] == "2015" and es["maxdate"] == "2020"
    assert es["tool"] == "sciforge" and es["email"] == FAKE_EMAIL and es["api_key"] == FAKE_API_KEY
    assert h.requests[1].url.params["id"] == "111,222"
    assert h.requests[0].headers["User-Agent"].startswith("SciForge/0.4")
    r1, r2 = out.records
    assert r1.pmid == "111" and r1.doi == "10.1000/abc123" and r1.year == 2021
    assert r1.authors == ["Smith JA", "Lee K"] and r1.journal == "Journal of Thrombosis"
    assert r1.source_url == "https://pubmed.ncbi.nlm.nih.gov/111/"
    assert r2.doi is None
    assert r2.year == 2021  # from sortpubdate when pubdate lacks a year
    # logged params are redacted
    logged = h.run_log.entries[0].params
    assert logged["api_key"] == "[REDACTED]" and logged["email"] == "[REDACTED]"
    assert h.run_log.entries[0].result_count == 2


def test_no_key_or_email_not_sent(make_harness):
    s = Settings(max_retries=0, backoff_seconds=0)
    h = make_harness(pubmed_handler(json_response(esearch_payload([])), json_response({})), s)
    out = h.pubmed.search("q", 3)
    assert out.status == "ok" and out.records == []
    params = h.requests[0].url.params
    assert "api_key" not in params and "email" not in params and "mindate" not in params
    assert len(h.requests) == 1  # no esummary for empty result


def test_open_ended_year_range(make_harness):
    h = make_harness(pubmed_handler(json_response(esearch_payload([])), json_response({})))
    h.pubmed.search("q", 3, from_year=2020)
    assert h.requests[0].url.params["mindate"] == "2020" and h.requests[0].url.params["maxdate"] == "3000"


@pytest.mark.parametrize("pubdate, sortpubdate, expected", [
    ("2021 Mar 15", None, 2021), ("2019 Dec-2020 Jan", None, 2019), ("Spring", "2020/04/01 00:00", 2020),
    ("", "", None), (None, None, None), ("unknown", "n/a", None), ("0999", None, None), (2021, None, None),
])
def test_parse_year(pubdate, sortpubdate, expected):
    assert parse_pubmed_year(pubdate, sortpubdate) == expected


def test_record_missing_fields_are_none():
    r = record_from_summary("5", {"uid": "5"})
    assert r.title is None and r.year is None and r.doi is None and r.journal is None and r.authors == []


def test_record_ignores_collective_and_malformed_authors():
    doc = esummary_doc("5")
    doc["authors"] = [{"name": "Group X", "authtype": "CollectiveName"}, "bad", {"name": ""}, {"name": "Doe J"}]
    doc["articleids"] = [{"idtype": "doi", "value": "not a doi"}, "junk"]
    r = record_from_summary("5", doc)
    assert r.authors == ["Doe J"] and r.doi is None


@pytest.mark.parametrize("payload", [
    [], "text", {"nope": 1}, {"esearchresult": []}, {"esearchresult": {"idlist": "123"}},
    {"esearchresult": {"ERROR": "Invalid query"}}, {"esearchresult": {"idlist": ["12a"]}},
])
def test_parse_esearch_malformed(payload):
    with pytest.raises(MalformedResponseError):
        parse_esearch(payload)


@pytest.mark.parametrize("payload", [
    None, [], {"result": []}, {"result": {"uids": "1"}}, {"error": "API rate limit exceeded"},
])
def test_parse_esummary_malformed(payload):
    with pytest.raises(MalformedResponseError):
        parse_esummary(payload)


def test_parse_esummary_skips_error_docs():
    docs = parse_esummary(esummary_payload([esummary_doc("1")], missing=["2"]))
    assert list(docs) == ["1"]


def test_search_malformed_esearch_is_recorded(make_harness):
    h = make_harness(pubmed_handler(json_response({"unexpected": True}), json_response({})))
    out = h.pubmed.search("q", 3)
    assert out.status == "failed" and out.records == []
    assert h.run_log.errors[0].error_type == "unexpected_structure"
    assert h.run_log.entries[0].status == "error"


def test_search_invalid_json(make_harness):
    h = make_harness(pubmed_handler(httpx.Response(200, content=b"<html>oops"), json_response({})))
    out = h.pubmed.search("q", 3)
    assert out.status == "failed"
    err = h.run_log.errors[0]
    assert err.error_type == "invalid_json" and err.database == "pubmed" and err.query == "q"
    assert err.http_status == 200


def test_search_esummary_failure_is_failed(make_harness):
    h = make_harness(pubmed_handler(json_response(esearch_payload(["1"])), httpx.Response(400)))
    out = h.pubmed.search("q", 3)
    assert out.status == "failed" and h.run_log.errors[0].http_status == 400


def test_search_missing_summary_is_partial(make_harness):
    h = make_harness(pubmed_handler(json_response(esearch_payload(["1", "2"])),
                                    json_response(esummary_payload([esummary_doc("1")], missing=["2"]))))
    out = h.pubmed.search("q", 3)
    assert out.status == "partial" and len(out.records) == 1
    assert h.run_log.errors[0].error_type == "missing_summary"


def test_lookup_many(make_harness):
    h = make_harness(pubmed_handler(None, json_response(esummary_payload([esummary_doc("1")], missing=["2"]))))
    res = h.pubmed.lookup_many(["1", "PMID: 2", "1", "bad"])
    assert set(res) == {"1", "2"}
    assert res["1"].outcome == "resolved" and res["1"].record.pmid == "1"
    assert res["2"].outcome == "not_found" and res["2"].http_status == 200
    assert len(h.requests) == 1


def test_lookup_many_failure_is_lookup_failed(make_harness):
    h = make_harness(pubmed_handler(None, json_response({"result": "garbage"})))
    res = h.pubmed.lookup_many(["1"])
    assert res["1"].outcome == "lookup_failed" and res["1"].error_type == "unexpected_structure"
