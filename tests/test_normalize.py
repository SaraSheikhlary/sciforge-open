"""DOI / PMID / title / surname normalization."""

import pytest

from sciforge.normalize import (
    extract_surname,
    first_author_surname,
    normalize_doi,
    normalize_pmid,
    normalize_title,
)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("10.1000/ABC123", "10.1000/abc123"),
        ("  10.1000/abc123  ", "10.1000/abc123"),
        ("https://doi.org/10.1000/abc123", "10.1000/abc123"),
        ("http://doi.org/10.1000/abc123", "10.1000/abc123"),
        ("http://dx.doi.org/10.1000/abc123", "10.1000/abc123"),
        ("https://www.doi.org/10.1000/abc123", "10.1000/abc123"),
        ("doi:10.1000/abc123", "10.1000/abc123"),
        ("DOI: 10.1000/abc123", "10.1000/abc123"),
        ("DOI 10.1000/abc123", "10.1000/abc123"),
        ("doi: https://doi.org/10.1000/abc123", "10.1000/abc123"),
        ("10.1000/abc123.", "10.1000/abc123"),
        ("10.1000/abc123;", "10.1000/abc123"),
        ("https://doi.org/10.1000%2Fabc%28x%29", "10.1000/abc(x)"),
        ("10.1002/(SICI)1097-4636(199706)35:4<405::AID-JBM1>3.0.CO;2-9",
         "10.1002/(sici)1097-4636(199706)35:4<405::aid-jbm1>3.0.co;2-9"),
        ("10.1000.10/xyz", "10.1000.10/xyz"),
    ],
)
def test_normalize_doi_valid(raw, expected):
    assert normalize_doi(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [None, "", "   ", "abc", "11.1000/abc", "10.1000", "10./abc", "10.abc/def", "10.1000/", "10.1000/ab c",
     "https://example.org/10.1000/abc", 1234, ["10.1000/abc"]],
)
def test_normalize_doi_invalid(raw):
    assert normalize_doi(raw) is None


@pytest.mark.parametrize(
    "raw, expected",
    [("123", "123"), (" 123 ", "123"), ("PMID: 123", "123"), ("pmid:123", "123"), ("PMID 34567890", "34567890"),
     ("\t987654\n", "987654"), ("00123", "123"), (123, "123")],
)
def test_normalize_pmid_valid(raw, expected):
    assert normalize_pmid(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [None, "", "  ", "abc", "12a3", "123.0", "-123", "0", "000", 0, -5, True, False, 12.0, "1234567890",
     "PMC123456", "10.1000/abc", "12 34"],
)
def test_normalize_pmid_rejects(raw):
    assert normalize_pmid(raw) is None


def test_normalize_title():
    assert normalize_title("  Shear-Stress <i>Activates</i> Platelets.  ") == "shear stress activates platelets"
    assert normalize_title("Café résumé") == "cafe resume"
    assert normalize_title("[Platelet activation]. ") == "platelet activation"
    assert normalize_title("A &amp; B") == "a b"
    assert normalize_title("") is None
    assert normalize_title("...") is None
    assert normalize_title(None) is None


@pytest.mark.parametrize(
    "name, expected",
    [("Smith JA", "smith"), ("Smith, John A.", "smith"), ("John Smith", "smith"), ("van der Berg J", "vanderberg"),
     ("van der Berg, Johan", "vanderberg"), ("Müller K", "muller"), ("O'Neil P", "oneil"), ("Smith", "smith"),
     ("", None), ("   ", None)],
)
def test_extract_surname(name, expected):
    assert extract_surname(name) == expected


def test_first_author_surname():
    assert first_author_surname(["Smith JA", "Lee K"]) == "smith"
    assert first_author_surname([]) is None
    assert first_author_surname(None) is None
