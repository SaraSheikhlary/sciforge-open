"""Crossref client: parameters, parsing, malformed responses, DOI lookup."""

import httpx
import pytest

from conftest import FAKE_EMAIL, crossref_list_payload, crossref_work, crossref_work_payload, json_response
from sciforge.config import Settings
from sciforge.crossref import parse_crossref_year, parse_works_list, record_from_work
from sciforge.http_utils import MalformedResponseError


def test_search_params_and_records(make_harness):
    items = [crossref_work(), crossref_work("10.2000/NoYear", year=None)]
    h = make_harness(lambda req: json_response(crossref_list_payload(items, total=999)))
    out = h.crossref.search("platelet shear", 7, from_year=2015, to_year=2020)
    assert out.status == "ok" and out.total_hits == 999 and len(out.records) == 2
    req = h.requests[0]
    assert req.url.path == "/works"
    p = req.url.params
    assert p["query.bibliographic"] == "platelet shear" and p["rows"] == "7"
    assert p["filter"] == "from-pub-date:2015,until-pub-date:2020"
    assert p["mailto"] == FAKE_EMAIL
    assert f"mailto:{FAKE_EMAIL}" in req.headers["User-Agent"]
    r = out.records[0]
    assert r.doi == "10.1000/abc123" and r.year == 2021 and r.authors == ["Smith, John A.", "Lee, Kim"]
    assert r.journal == "Journal of Thrombosis" and r.source_url == "https://doi.org/10.1000/abc123" and r.pmid is None
    assert out.records[1].year is None
    assert h.run_log.entries[0].params["mailto"] == "[REDACTED]"


def test_no_mailto_without_email(make_harness):
    h = make_harness(lambda req: json_response(crossref_list_payload([])), Settings(max_retries=0))
    h.crossref.search("q", 3)
    req = h.requests[0]
    assert "mailto" not in req.url.params and "mailto" not in req.headers["User-Agent"]
    assert "filter" not in req.url.params


def test_parse_year_variants():
    assert parse_crossref_year({"issued": {"date-parts": [[2019]]}}) == 2019
    assert parse_crossref_year({"issued": {"date-parts": [[None]]}, "published": {"date-parts": [[2018, 1]]}}) == 2018
    assert parse_crossref_year({"issued": {"date-parts": []}}) is None
    assert parse_crossref_year({"issued": {"date-parts": [["2019"]]}}) is None
    assert parse_crossref_year({"issued": "2019"}) is None
    assert parse_crossref_year({}) is None


def test_record_from_sparse_work():
    r = record_from_work({"title": [], "author": [{"name": "Consortium"}, {"given": "NoFamily"}, "x"]})
    assert r.title is None and r.authors == [] and r.doi is None and r.year is None and r.journal is None


@pytest.mark.parametrize("payload", [
    [], "x", {"status": "failed", "message": {}}, {"status": "ok"}, {"status": "ok", "message": []},
    {"status": "ok", "message": {"items": {"a": 1}}},
])
def test_parse_works_list_malformed(payload):
    with pytest.raises(MalformedResponseError):
        parse_works_list(payload)


def test_search_skips_malformed_items(make_harness):
    h = make_harness(lambda req: json_response(crossref_list_payload([crossref_work(), "junk", 5])))
    out = h.crossref.search("q", 3)
    assert out.status == "partial" and len(out.records) == 1
    assert [e.error_type for e in h.run_log.errors] == ["unexpected_structure"] * 2


def test_search_missing_keys(make_harness):
    h = make_harness(lambda req: json_response({"status": "ok", "message": {}}))
    out = h.crossref.search("q", 3)
    assert out.status == "failed" and h.run_log.errors[0].error_type == "unexpected_structure"


def test_lookup_resolved(make_harness):
    h = make_harness(lambda req: json_response(crossref_work_payload(crossref_work("10.1000/ABC123"))))
    res = h.crossref.lookup("10.1000/abc123")
    assert res.outcome == "resolved" and res.http_status == 200 and res.record.doi == "10.1000/abc123"
    assert h.requests[0].url.path == "/works/10.1000/abc123"


def test_lookup_404_is_not_found_and_not_an_error(make_harness):
    h = make_harness(lambda req: httpx.Response(404, text="Resource not found."))
    res = h.crossref.lookup("10.1000/missing")
    assert res.outcome == "not_found" and res.http_status == 404
    assert h.run_log.errors == [] and h.run_log.entries[0].status == "not_found"
    assert len(h.requests) == 1  # 404 is not retried


def test_lookup_malformed_is_lookup_failed(make_harness):
    h = make_harness(lambda req: json_response({"status": "ok", "message": "nope"}))
    res = h.crossref.lookup("10.1000/x")
    assert res.outcome == "lookup_failed" and res.error_type == "unexpected_structure"
