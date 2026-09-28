"""Deterministic citation verification.

Each deduplicated record's DOI is resolved against Crossref and its PMID
against PubMed; the returned metadata is compared field by field.

Per-identifier status
---------------------
* ``not_verified``: the identifier did not resolve (``not_found``) or the
  lookup failed (``lookup_failed``).
* ``verified``: it resolved AND the title matches AND the year does not
  mismatch AND the first-author surname does not mismatch. Year and first
  author only count when known on both sides ("when both known"); if either
  side lacks them they are ``not_compared`` and do not block ``verified``.
  The title must be present on both sides and match.
* ``partially_verified``: it resolved but the title mismatches or cannot be
  compared, or the year or first author mismatches.

Journal is a soft field: it is compared and reported but never changes the
status (abbreviations and naming variants are common).

Combining DOI and PMID checks
-----------------------------
* No identifier -> ``not_verified`` (reason ``no_identifier``).
* No identifier resolved -> ``not_verified`` with reason ``lookup_failed`` if
  any lookup failed, else ``not_found``.
* Otherwise ``verified`` if every resolved check is ``verified`` and no
  identifier was ``not_found``; a lookup failure of the other identifier is
  noted in ``reasons`` but does not downgrade. Anything else is
  ``partially_verified``.
"""

from __future__ import annotations

from difflib import SequenceMatcher

from sciforge.crossref import CrossrefClient
from sciforge.logging_utils import iso_utc
from sciforge.models import FieldComparison, IdentifierCheck, LookupResult, Record, VerificationResult
from sciforge.normalize import first_author_surname, normalize_text_field, normalize_title
from sciforge.pubmed import PubMedClient

TITLE_SIMILARITY_THRESHOLD = 0.9
JOURNAL_SIMILARITY_THRESHOLD = 0.8


def _ratio(a: str, b: str) -> float:
    return round(SequenceMatcher(None, a, b).ratio(), 4)


def compare_title(record_title: str | None, reference_title: str | None) -> FieldComparison:
    """Exact normalized match, or difflib ratio >= ``TITLE_SIMILARITY_THRESHOLD``."""
    a, b = normalize_title(record_title), normalize_title(reference_title)
    if a is None or b is None:
        return FieldComparison(field="title", status="not_compared", record_value=record_title,
                               reference_value=reference_title, note="title missing on at least one side")
    if a == b:
        return FieldComparison(field="title", status="match", record_value=record_title,
                               reference_value=reference_title, similarity=1.0, note="exact normalized match")
    ratio = _ratio(a, b)
    status = "match" if ratio >= TITLE_SIMILARITY_THRESHOLD else "mismatch"
    return FieldComparison(field="title", status=status, record_value=record_title, reference_value=reference_title,
                           similarity=ratio, note=f"similarity threshold {TITLE_SIMILARITY_THRESHOLD}")


def compare_year(record_year: int | None, reference_year: int | None) -> FieldComparison:
    """Exact integer match; ``not_compared`` if either is unknown."""
    if record_year is None or reference_year is None:
        return FieldComparison(field="year", status="not_compared", record_value=record_year,
                               reference_value=reference_year, note="year missing on at least one side")
    status = "match" if record_year == reference_year else "mismatch"
    return FieldComparison(field="year", status=status, record_value=record_year, reference_value=reference_year)


def compare_first_author(record_authors: list[str], reference_authors: list[str]) -> FieldComparison:
    """Exact match of normalized first-author surnames."""
    a, b = first_author_surname(record_authors), first_author_surname(reference_authors)
    rv = record_authors[0] if record_authors else None
    fv = reference_authors[0] if reference_authors else None
    if a is None or b is None:
        return FieldComparison(field="first_author", status="not_compared", record_value=rv, reference_value=fv,
                               note="first author missing on at least one side")
    status = "match" if a == b else "mismatch"
    return FieldComparison(field="first_author", status=status, record_value=rv, reference_value=fv,
                           note="compared normalized surnames")


def compare_journal(record_journal: str | None, reference_journal: str | None) -> FieldComparison:
    """Soft comparison: equal, containment, or ratio >= ``JOURNAL_SIMILARITY_THRESHOLD``.
    Never affects the verification status."""
    a, b = normalize_text_field(record_journal), normalize_text_field(reference_journal)
    if a is None or b is None:
        return FieldComparison(field="journal", status="not_compared", record_value=record_journal,
                               reference_value=reference_journal, affects_status=False,
                               note="journal missing on at least one side")
    if a == b or a in b or b in a:
        ratio = 1.0 if a == b else _ratio(a, b)
        return FieldComparison(field="journal", status="match", record_value=record_journal,
                               reference_value=reference_journal, similarity=ratio, affects_status=False,
                               note="soft field (equal or contained)")
    ratio = _ratio(a, b)
    status = "match" if ratio >= JOURNAL_SIMILARITY_THRESHOLD else "mismatch"
    return FieldComparison(field="journal", status=status, record_value=record_journal,
                           reference_value=reference_journal, similarity=ratio, affects_status=False,
                           note=f"soft field; threshold {JOURNAL_SIMILARITY_THRESHOLD}; does not affect status")


def compare_records(record: Record, reference: Record) -> list[FieldComparison]:
    """Compare title, year, first author, and journal."""
    return [
        compare_title(record.title, reference.title),
        compare_year(record.year, reference.year),
        compare_first_author(record.authors, reference.authors),
        compare_journal(record.journal, reference.journal),
    ]


def status_from_comparisons(comparisons: list[FieldComparison]) -> tuple[str, list[str]]:
    """Per-identifier status for a resolved identifier, plus reasons."""
    by_field = {c.field: c for c in comparisons}
    reasons: list[str] = []
    title = by_field.get("title")
    if title is None or title.status != "match":
        reasons.append("title_not_compared" if title is None or title.status == "not_compared" else "title_mismatch")
    for field in ("year", "first_author"):
        comp = by_field.get(field)
        if comp is not None and comp.status == "mismatch":
            reasons.append(f"{field}_mismatch")
    return ("verified" if not reasons else "partially_verified"), reasons


def check_identifier(record: Record, lookup: LookupResult) -> IdentifierCheck:
    """Turn a lookup into an :class:`IdentifierCheck` for ``record``."""
    base = dict(
        identifier_type=lookup.identifier_type,
        identifier=lookup.identifier,
        database=lookup.database,
        outcome=lookup.outcome,
        http_status=lookup.http_status,
        error_type=lookup.error_type,
        error_message=lookup.error_message,
    )
    if lookup.outcome != "resolved" or lookup.record is None:
        outcome = lookup.outcome if lookup.outcome != "resolved" else "lookup_failed"
        base["outcome"] = outcome
        return IdentifierCheck(**base, status="not_verified", reasons=[outcome])
    comparisons = compare_records(record, lookup.record)
    status, reasons = status_from_comparisons(comparisons)
    return IdentifierCheck(**base, status=status, reasons=reasons, comparisons=comparisons)


def combine_checks(checks: list[IdentifierCheck]) -> tuple[str, list[str]]:
    """Overall status from per-identifier checks (see module docstring)."""
    if not checks:
        return "not_verified", ["no_identifier"]
    resolved = [c for c in checks if c.outcome == "resolved"]
    if not resolved:
        reason = "lookup_failed" if any(c.outcome == "lookup_failed" for c in checks) else "not_found"
        return "not_verified", [reason] + [f"{c.identifier_type}:{c.outcome}" for c in checks]
    reasons = [f"{c.identifier_type}:{r}" for c in checks for r in c.reasons]
    not_found = any(c.outcome == "not_found" for c in checks)
    if all(c.status == "verified" for c in resolved) and not not_found:
        return "verified", reasons
    return "partially_verified", reasons


def build_verification(
    record: Record, doi_lookup: LookupResult | None, pmid_lookup: LookupResult | None
) -> VerificationResult:
    """Verification result for ``record`` from already-performed lookups."""
    checks: list[IdentifierCheck] = []
    if record.doi:
        if doi_lookup is None:
            doi_lookup = LookupResult("crossref", "doi", record.doi, "lookup_failed",
                                      error_type="not_attempted", error_message="DOI lookup was not performed")
        checks.append(check_identifier(record, doi_lookup))
    if record.pmid:
        if pmid_lookup is None:
            pmid_lookup = LookupResult("pubmed", "pmid", record.pmid, "lookup_failed",
                                       error_type="not_attempted", error_message="PMID lookup was not performed")
        checks.append(check_identifier(record, pmid_lookup))
    status, reasons = combine_checks(checks)
    return VerificationResult(record_id=record.record_id, status=status, reasons=reasons, checks=checks,
                              verified_at=iso_utc())


class Verifier:
    """Resolves identifiers for a list of records and builds results."""

    def __init__(self, pubmed: PubMedClient, crossref: CrossrefClient) -> None:
        self.pubmed = pubmed
        self.crossref = crossref

    def verify_all(self, records: list[Record]) -> list[VerificationResult]:
        """Verify every record. PMIDs are resolved in batched esummary calls;
        each distinct DOI is resolved once against Crossref."""
        pmids = [r.pmid for r in records if r.pmid]
        pmid_lookups = self.pubmed.lookup_many(pmids) if pmids else {}
        doi_lookups: dict[str, LookupResult] = {}
        for record in records:
            if record.doi and record.doi not in doi_lookups:
                doi_lookups[record.doi] = self.crossref.lookup(record.doi)
        return [
            build_verification(r, doi_lookups.get(r.doi) if r.doi else None,
                               pmid_lookups.get(r.pmid) if r.pmid else None)
            for r in records
        ]
