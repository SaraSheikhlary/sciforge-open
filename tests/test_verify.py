"""Metadata comparison and verification status rules."""

from sciforge.models import LookupResult, Record
from sciforge.verify import (
    TITLE_SIMILARITY_THRESHOLD,
    build_verification,
    compare_journal,
    compare_title,
)

TS = "2026-09-28T23:41:00Z"


def rec(source="pubmed", **kw):
    kw.setdefault("title", "Shear stress activates human platelets in vitro")
    kw.setdefault("authors", ["Smith JA", "Lee K"])
    kw.setdefault("year", 2021)
    kw.setdefault("journal", "Journal of Thrombosis")
    return Record(source_database=source, retrieval_timestamp=TS, **kw)


def resolved(kind, ident, reference):
    db = "crossref" if kind == "doi" else "pubmed"
    return LookupResult(db, kind, ident, "resolved", http_status=200, record=reference)


def test_title_similarity_threshold():
    assert TITLE_SIMILARITY_THRESHOLD == 0.9
    assert compare_title("Shear stress activates platelets.", "shear stress activates platelets").status == "match"
    near = compare_title("Shear stress activates human platelets in vitro", "Shear stress activates human platelet in vitro")
    assert near.status == "match" and near.similarity >= 0.9
    far = compare_title("Shear stress activates platelets", "Cold storage of red cells")
    assert far.status == "mismatch" and far.similarity < 0.9
    assert compare_title(None, "x").status == "not_compared"


def test_journal_is_soft():
    j = compare_journal("Blood", "Nature")
    assert j.status == "mismatch" and j.affects_status is False
    assert compare_journal("J Thromb", "J Thromb Haemost").status == "match"


def test_verified_doi():
    r = rec(doi="10.1/x")
    ref = rec("crossref", doi="10.1/x", authors=["Smith, John A."], journal="Nature")  # journal mismatch is soft
    v = build_verification(r, resolved("doi", "10.1/x", ref), None)
    assert v.status == "verified"
    check = v.checks[0]
    assert check.identifier_type == "doi" and check.http_status == 200 and check.outcome == "resolved"
    assert {c.field for c in check.comparisons} == {"title", "year", "first_author", "journal"}


def test_verified_when_year_unknown_on_one_side():
    r = rec(doi="10.1/x", year=None)
    v = build_verification(r, resolved("doi", "10.1/x", rec("crossref", doi="10.1/x")), None)
    assert v.status == "verified"
    assert [c for c in v.checks[0].comparisons if c.field == "year"][0].status == "not_compared"


def test_partially_verified_year_mismatch():
    r = rec(doi="10.1/x", year=2020)
    v = build_verification(r, resolved("doi", "10.1/x", rec("crossref", doi="10.1/x", year=2021)), None)
    assert v.status == "partially_verified"
    assert "doi:year_mismatch" in v.reasons


def test_partially_verified_title_mismatch_and_author_mismatch():
    r = rec(doi="10.1/x")
    ref = rec("crossref", doi="10.1/x", title="An unrelated paper about corals", authors=["Jones, B"])
    v = build_verification(r, resolved("doi", "10.1/x", ref), None)
    assert v.status == "partially_verified"
    assert "doi:title_mismatch" in v.reasons and "doi:first_author_mismatch" in v.reasons


def test_partially_verified_title_not_comparable():
    r = rec(doi="10.1/x", title=None)
    v = build_verification(r, resolved("doi", "10.1/x", rec("crossref", doi="10.1/x")), None)
    assert v.status == "partially_verified" and "doi:title_not_compared" in v.reasons


def test_not_verified_no_identifier():
    v = build_verification(rec(doi=None, pmid=None), None, None)
    assert v.status == "not_verified" and v.reasons == ["no_identifier"] and v.checks == []


def test_not_verified_not_found_vs_lookup_failed():
    r = rec(doi="10.1/x")
    nf = LookupResult("crossref", "doi", "10.1/x", "not_found", http_status=404, error_type="not_found")
    v = build_verification(r, nf, None)
    assert v.status == "not_verified" and v.reasons[0] == "not_found" and v.checks[0].http_status == 404
    failed = LookupResult("crossref", "doi", "10.1/x", "lookup_failed", http_status=503, error_type="http_error",
                          error_message="HTTP 503")
    v = build_verification(r, failed, None)
    assert v.status == "not_verified" and v.reasons[0] == "lookup_failed"
    assert v.checks[0].error_message == "HTTP 503"


def test_both_identifiers_verified():
    r = rec(doi="10.1/x", pmid="42")
    v = build_verification(r, resolved("doi", "10.1/x", rec("crossref", doi="10.1/x")),
                           resolved("pmid", "42", rec(pmid="42")))
    assert v.status == "verified" and len(v.checks) == 2


def test_both_identifiers_one_not_found_is_partial():
    r = rec(doi="10.1/x", pmid="42")
    v = build_verification(r, resolved("doi", "10.1/x", rec("crossref", doi="10.1/x")),
                           LookupResult("pubmed", "pmid", "42", "not_found", http_status=200))
    assert v.status == "partially_verified" and "pmid:not_found" in v.reasons


def test_both_identifiers_one_lookup_failed_still_verified():
    r = rec(doi="10.1/x", pmid="42")
    v = build_verification(r, resolved("doi", "10.1/x", rec("crossref", doi="10.1/x")),
                           LookupResult("pubmed", "pmid", "42", "lookup_failed", error_type="timeout"))
    assert v.status == "verified" and "pmid:lookup_failed" in v.reasons


def test_both_identifiers_one_partial():
    r = rec(doi="10.1/x", pmid="42")
    v = build_verification(r, resolved("doi", "10.1/x", rec("crossref", doi="10.1/x")),
                           resolved("pmid", "42", rec(pmid="42", year=1999)))
    assert v.status == "partially_verified" and "pmid:year_mismatch" in v.reasons


def test_missing_lookup_is_lookup_failed():
    v = build_verification(rec(doi="10.1/x"), None, None)
    assert v.status == "not_verified" and v.checks[0].error_type == "not_attempted"
