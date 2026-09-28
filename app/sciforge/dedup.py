"""Conservative duplicate detection and merging.

Rules (applied in order to a candidate pair):

1. Both have a DOI: duplicates iff the normalized DOIs are equal. Different
   DOIs are never merged, even if titles match.
2. Both have a PMID: duplicates iff the PMIDs are equal. Different PMIDs are
   never merged.
3. Otherwise (a DOI is missing on at least one side) fall back to requiring
   ALL of: identical normalized title, identical year, identical normalized
   first-author surname. If any of those is missing on either side, the
   records are NOT merged.
"""

from __future__ import annotations

from sciforge.models import Record
from sciforge.normalize import first_author_surname, normalize_doi, normalize_text_field, normalize_title

__all__ = ["deduplicate", "match_reason", "merge_records", "normalize_doi", "normalize_title"]

MATCH_DOI = "doi"
MATCH_PMID = "pmid"
MATCH_FALLBACK = "title_year_first_author"


def match_reason(a: Record, b: Record) -> str | None:
    """Return the rule that makes ``a`` and ``b`` duplicates, or None."""
    if a.doi and b.doi:
        return MATCH_DOI if a.doi == b.doi else None
    if a.pmid and b.pmid:
        return MATCH_PMID if a.pmid == b.pmid else None
    title_a, title_b = normalize_title(a.title), normalize_title(b.title)
    author_a, author_b = first_author_surname(a.authors), first_author_surname(b.authors)
    if None in (title_a, title_b, a.year, b.year, author_a, author_b):
        return None
    if title_a == title_b and a.year == b.year and author_a == author_b:
        return MATCH_FALLBACK
    return None


def _label(record: Record) -> str:
    return record.source_database


def _conflict(field: str, kept: object, other: object, primary: Record, secondary: Record) -> str:
    return f"{field}: kept {kept!r} ({_label(primary)}); other value {other!r} ({_label(secondary)})"


def merge_records(primary: Record, secondary: Record, matched_on: str) -> Record:
    """Merge ``secondary`` into ``primary``.

    Non-null values from ``primary`` win; null fields are filled from
    ``secondary``. Nothing is fabricated. Disagreements are recorded in
    ``conflicts``; provenance of both records is kept. ``record_id`` stays the
    primary's.
    """
    updates: dict[str, object] = {}
    conflicts = list(primary.conflicts)

    for field in ("title", "year", "doi", "pmid", "journal"):
        mine, theirs = getattr(primary, field), getattr(secondary, field)
        if mine is None and theirs is not None:
            updates[field] = theirs
        elif mine is not None and theirs is not None and mine != theirs:
            if field == "title" and normalize_title(mine) == normalize_title(theirs):
                continue  # only formatting differs
            if field == "journal" and normalize_text_field(mine) == normalize_text_field(theirs):
                continue
            conflicts.append(_conflict(field, mine, theirs, primary, secondary))

    if not primary.authors and secondary.authors:
        updates["authors"] = list(secondary.authors)
    elif primary.authors and secondary.authors:
        fa_mine, fa_theirs = first_author_surname(primary.authors), first_author_surname(secondary.authors)
        if fa_mine and fa_theirs and fa_mine != fa_theirs:
            conflicts.append(_conflict("first_author", primary.authors[0], secondary.authors[0], primary, secondary))
        if len(primary.authors) != len(secondary.authors):
            conflicts.append(
                _conflict("author_count", len(primary.authors), len(secondary.authors), primary, secondary)
            )

    provenance = list(primary.provenance)
    for prov in secondary.provenance:
        provenance.append(prov.model_copy(update={"matched_on": prov.matched_on or matched_on}))
    conflicts.extend(c for c in secondary.conflicts if c not in conflicts)

    updates["provenance"] = provenance
    updates["conflicts"] = conflicts
    return primary.model_copy(update=updates, deep=True)


def deduplicate(records: list[Record]) -> tuple[list[Record], int]:
    """Deduplicate ``records`` preserving input order.

    Each record is compared with the already-merged records; it is merged into
    the first one it matches. Returns (unique records, number merged away).
    """
    unique: list[Record] = []
    merged_count = 0
    for record in records:
        for index, existing in enumerate(unique):
            reason = match_reason(existing, record)
            if reason is not None:
                unique[index] = merge_records(existing, record, reason)
                merged_count += 1
                break
        else:
            unique.append(record)
    return unique, merged_count


def provenance_databases(record: Record) -> list[str]:
    """Distinct source databases contributing to ``record``."""
    seen: list[str] = []
    for prov in record.provenance:
        if prov.source_database not in seen:
            seen.append(prov.source_database)
    return seen
