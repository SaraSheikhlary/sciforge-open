"""Record model invariants."""

import pytest
from pydantic import ValidationError

from sciforge.models import Record, make_record_id


def test_record_normalizes_identifiers_and_never_fills_fields():
    r = Record(title=" T ", doi="https://doi.org/10.1000/ABC", pmid="PMID: 42", source_database="pubmed",
               retrieval_timestamp="2026-01-01T00:00:00Z")
    assert r.doi == "10.1000/abc"
    assert r.pmid == "42"
    assert r.title == "T"
    assert r.year is None and r.journal is None and r.authors == []
    assert r.provenance[0].source_database == "pubmed"
    assert r.record_id.startswith("rec_")


def test_invalid_identifiers_become_none():
    r = Record(doi="not-a-doi", pmid="abc", source_database="crossref", retrieval_timestamp="x")
    assert r.doi is None and r.pmid is None


def test_record_id_is_stable():
    a = make_record_id("pubmed", None, "42", "t", 2020)
    assert a == make_record_id("pubmed", None, "42", "other title", 1999)
    assert a != make_record_id("crossref", None, "42", "t", 2020)
    r1 = Record(title="T", year=2020, source_database="crossref", retrieval_timestamp="a")
    r2 = Record(title="T", year=2020, source_database="crossref", retrieval_timestamp="b")
    assert r1.record_id == r2.record_id


def test_year_must_be_int():
    with pytest.raises(ValidationError):
        Record(year="2020", source_database="pubmed", retrieval_timestamp="x")
    with pytest.raises(ValidationError):
        Record(year=True, source_database="pubmed", retrieval_timestamp="x")
