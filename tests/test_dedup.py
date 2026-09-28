"""Duplicate detection and merging."""

from sciforge.dedup import deduplicate, match_reason, merge_records, provenance_databases
from sciforge.models import Record

TS = "2026-09-28T23:41:00Z"


def rec(source="pubmed", **kw):
    kw.setdefault("title", "Shear stress activates platelets")
    kw.setdefault("authors", ["Smith JA"])
    kw.setdefault("year", 2021)
    return Record(source_database=source, retrieval_timestamp=TS, **kw)


def test_doi_match_across_sources():
    pm = rec("pubmed", doi="10.1000/ABC", pmid="111", journal=None, title="Shear stress activates platelets.")
    cr = rec("crossref", doi="https://doi.org/10.1000/abc", journal="J Thromb", authors=["Smith, John A."],
             title="Completely different formatting of title")
    assert match_reason(pm, cr) == "doi"
    unique, merged = deduplicate([pm, cr])
    assert merged == 1 and len(unique) == 1
    m = unique[0]
    assert m.record_id == pm.record_id
    assert m.pmid == "111" and m.journal == "J Thromb"
    assert provenance_databases(m) == ["pubmed", "crossref"]
    assert m.provenance[1].matched_on == "doi"
    assert any(c.startswith("title:") for c in m.conflicts)


def test_different_dois_never_merge_even_with_same_title():
    a = rec("pubmed", doi="10.1000/a")
    b = rec("crossref", doi="10.1000/b")
    assert match_reason(a, b) is None
    assert len(deduplicate([a, b])[0]) == 2


def test_fallback_title_year_first_author():
    a = rec("pubmed", doi=None, pmid="5", title="Shear Stress Activates Platelets.", authors=["Smith JA"])
    b = rec("crossref", doi="10.1000/x", title="shear stress  activates platelets", authors=["Smith, John"])
    assert match_reason(a, b) == "title_year_first_author"
    unique, merged = deduplicate([a, b])
    assert merged == 1 and unique[0].doi == "10.1000/x" and unique[0].pmid == "5"


def test_fallback_requires_all_fields():
    base = dict(doi=None)
    assert match_reason(rec(**base, year=None), rec("crossref", **base)) is None
    assert match_reason(rec(**base, authors=[]), rec("crossref", **base)) is None
    assert match_reason(rec(**base, title=None), rec("crossref", **base)) is None


def test_near_but_different_records_not_merged():
    a = rec("pubmed", doi=None, title="Shear stress activates platelets")
    # different year
    assert match_reason(a, rec("crossref", doi=None, year=2022)) is None
    # different first author
    assert match_reason(a, rec("crossref", doi=None, authors=["Jones B"])) is None
    # one-word title difference (high similarity but not exact)
    assert match_reason(a, rec("crossref", doi=None, title="Shear stress activates platelet")) is None
    # different PMIDs
    assert match_reason(rec(doi=None, pmid="1"), rec(doi=None, pmid="2")) is None
    records = [a, rec("crossref", doi=None, year=2022), rec("crossref", doi=None, authors=["Jones B"])]
    assert deduplicate(records) == (records, 0)


def test_merge_records_conflicts_keep_first():
    a = rec("pubmed", doi="10.1/x", year=2020, journal="Blood")
    b = rec("crossref", doi="10.1/x", year=2021, journal="BLOOD", authors=["Jones, B", "Other, C"])
    m = merge_records(a, b, "doi")
    assert m.year == 2020 and m.journal == "Blood"
    assert any(c.startswith("year:") for c in m.conflicts)
    assert not any(c.startswith("journal:") for c in m.conflicts)  # only case differs
    assert any(c.startswith("first_author:") for c in m.conflicts)
    assert any(c.startswith("author_count:") for c in m.conflicts)
    assert a.conflicts == []  # original unchanged


def test_merge_does_not_fabricate():
    a = rec("pubmed", doi="10.1/x", year=None, journal=None)
    b = rec("crossref", doi="10.1/x", year=None, journal=None)
    m = merge_records(a, b, "doi")
    assert m.year is None and m.journal is None


def test_pmid_match():
    a = rec("pubmed", doi=None, pmid="77", title="A")
    b = rec("pubmed", doi=None, pmid="PMID: 77", title="B")
    assert match_reason(a, b) == "pmid"
