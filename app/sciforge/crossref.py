"""Crossref retrieval (``/works`` search) and DOI lookup (``/works/{doi}``)."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

from sciforge.config import Settings
from sciforge.http_utils import FetchResult, HttpFetcher, MalformedResponseError, RateLimiter
from sciforge.logging_utils import iso_utc
from sciforge.models import ErrorEntry, LookupResult, Record, SearchOutcome
from sciforge.normalize import normalize_doi
from sciforge.source_classification import crossref_metadata

__all__ = ["CrossrefClient", "normalize_doi", "parse_crossref_year", "parse_works_list", "record_from_work"]

DATABASE = "crossref"
WORKS_URL = "https://api.crossref.org/works"
MAX_ROWS = 1000
MIN_PLAUSIBLE_YEAR = 1500
MAX_PLAUSIBLE_YEAR = 2100


def _first_str(value: Any) -> str | None:
    """First non-empty string of a Crossref list field (or a plain string)."""
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.strip():
                return item.strip()
    return None


def _year_from_date(value: Any) -> int | None:
    if not isinstance(value, dict):
        return None
    parts = value.get("date-parts")
    if not isinstance(parts, list) or not parts or not isinstance(parts[0], list) or not parts[0]:
        return None
    year = parts[0][0]
    if isinstance(year, bool) or not isinstance(year, int):
        return None
    return year if MIN_PLAUSIBLE_YEAR <= year <= MAX_PLAUSIBLE_YEAR else None


def parse_crossref_year(work: dict[str, Any]) -> int | None:
    """Year from ``issued``, then ``published``, ``published-print``,
    ``published-online`` date-parts. None if absent or malformed."""
    for key in ("issued", "published", "published-print", "published-online"):
        year = _year_from_date(work.get(key))
        if year is not None:
            return year
    return None


def _authors(work: dict[str, Any]) -> list[str]:
    """Person authors formatted ``"Family, Given"`` (or ``"Family"``).

    Organisational authors (``name`` only) are excluded, mirroring PubMed where
    only ``authtype == "Author"`` entries are kept.
    """
    authors = work.get("author")
    if not isinstance(authors, list):
        return []
    names: list[str] = []
    for author in authors:
        if not isinstance(author, dict):
            continue
        family = author.get("family")
        given = author.get("given")
        if not isinstance(family, str) or not family.strip():
            continue
        if isinstance(given, str) and given.strip():
            names.append(f"{family.strip()}, {given.strip()}")
        else:
            names.append(family.strip())
    return names


def record_from_work(work: dict[str, Any], retrieved_at: str | None = None) -> Record:
    """Build a :class:`Record` from a Crossref work object. Missing fields stay None."""
    doi = normalize_doi(work.get("DOI"))
    url = work.get("URL") if isinstance(work.get("URL"), str) else None
    if doi:
        url = f"https://doi.org/{doi}"
    return Record(
        title=_first_str(work.get("title")),
        authors=_authors(work),
        year=parse_crossref_year(work),
        doi=doi,
        pmid=None,
        journal=_first_str(work.get("container-title")),
        source_database=DATABASE,
        source_url=url,
        retrieval_timestamp=retrieved_at or iso_utc(),
    )


def parse_works_list(data: Any) -> tuple[list[Any], int | None]:
    """Return (items, total-results) from a ``/works`` search response.

    Items are returned as-is; callers must check each is a dict.

    Raises:
        MalformedResponseError: if the structure is not as documented.
    """
    if not isinstance(data, dict):
        raise MalformedResponseError("Crossref response is not a JSON object")
    if data.get("status") not in (None, "ok"):
        raise MalformedResponseError(f"Crossref reported status {str(data.get('status'))[:50]!r}")
    message = data.get("message")
    if not isinstance(message, dict):
        raise MalformedResponseError("Crossref response has no 'message' object")
    items = message.get("items")
    if not isinstance(items, list):
        raise MalformedResponseError("Crossref 'message.items' is missing or not a list")
    total = message.get("total-results")
    return items, total if isinstance(total, int) and not isinstance(total, bool) else None


def parse_work(data: Any) -> dict[str, Any]:
    """Return the work object from a ``/works/{doi}`` response."""
    if not isinstance(data, dict):
        raise MalformedResponseError("Crossref response is not a JSON object")
    message = data.get("message")
    if not isinstance(message, dict):
        raise MalformedResponseError("Crossref response has no 'message' object")
    return message


class CrossrefClient:
    """Searches Crossref and resolves DOIs."""

    database = DATABASE

    def __init__(self, fetcher: HttpFetcher, settings: Settings, limiter: RateLimiter | None = None) -> None:
        self.fetcher = fetcher
        self.settings = settings
        self.limiter = limiter or fetcher.make_limiter(settings.crossref_min_interval)
        # Side data captured from search results (Records are unchanged): record_id -> type metadata
        # (type/subtype/publisher/container/institution) and record_id -> raw JATS abstract (when deposited).
        self.source_metadata: dict[str, dict[str, Any]] = {}
        self.search_abstracts: dict[str, str] = {}

    def _base_params(self) -> dict[str, str]:
        return {"mailto": self.settings.contact_email} if self.settings.contact_email else {}

    def _get(self, stage: str, url: str, params: dict[str, Any], query: str | None) -> FetchResult:
        return self.fetcher.get_json(
            database=DATABASE, stage=stage, url=url, params={**params, **self._base_params()}, query=query, limiter=self.limiter
        )

    def _malformed(self, result: FetchResult, stage: str, query: str | None, message: str) -> ErrorEntry:
        error = ErrorEntry(
            database=DATABASE,
            stage=stage,
            query=query,
            timestamp=iso_utc(),
            error_type="unexpected_structure",
            http_status=result.http_status,
            message=message,
            url=result.entry.url if result.entry else None,
        )
        if result.entry is not None:
            self.fetcher.run_log.mark_failed(result.entry, error)
        else:
            self.fetcher.run_log.add_error(error)
        return error

    def search(
        self, query: str, max_results: int, from_year: int | None = None, to_year: int | None = None
    ) -> SearchOutcome:
        """Search ``/works`` with ``query.bibliographic``. Never raises for API problems."""
        params: dict[str, Any] = {"query.bibliographic": query, "rows": max(0, min(max_results, MAX_ROWS))}
        filters = []
        if from_year is not None:
            filters.append(f"from-pub-date:{from_year}")
        if to_year is not None:
            filters.append(f"until-pub-date:{to_year}")
        if filters:
            params["filter"] = ",".join(filters)
        result = self._get("search", WORKS_URL, params, query)
        if not result.ok:
            self.fetcher.run_log.add_error(result.to_error(DATABASE, "search", query))
            return SearchOutcome(database=DATABASE, status="failed")
        try:
            items, total = parse_works_list(result.data)
        except MalformedResponseError as exc:
            self._malformed(result, "search", query, str(exc))
            return SearchOutcome(database=DATABASE, status="failed")
        retrieved_at = iso_utc()
        records: list[Record] = []
        skipped = 0
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                skipped += 1
                self.fetcher.run_log.add_error(
                    ErrorEntry(
                        database=DATABASE,
                        stage="search",
                        query=query,
                        timestamp=retrieved_at,
                        error_type="unexpected_structure",
                        http_status=result.http_status,
                        message=f"skipped malformed item at index {index} (not an object)",
                    )
                )
                continue
            record = record_from_work(item, retrieved_at)
            records.append(record)
            self.source_metadata[record.record_id] = crossref_metadata(item)
            abstract = item.get("abstract")
            if isinstance(abstract, str) and abstract.strip():
                self.search_abstracts[record.record_id] = abstract
        if result.entry is not None:
            result.entry.result_count = len(records)
        status = "ok" if not skipped else ("partial" if records else "failed")
        return SearchOutcome(database=DATABASE, status=status, records=records, total_hits=total)

    def lookup(self, doi: str) -> LookupResult:
        """Resolve one DOI via ``/works/{doi}``.

        HTTP 404 -> ``not_found``; any other failure -> ``lookup_failed``.
        """
        normalized = normalize_doi(doi) or doi
        url = f"{WORKS_URL}/{quote(normalized, safe='/')}"
        query = f"verify DOI: {normalized}"
        result = self._get("verification", url, {}, query)
        base = {"database": DATABASE, "identifier_type": "doi", "identifier": normalized}
        if result.not_found:
            return LookupResult(**base, outcome="not_found", http_status=404, error_type="not_found",
                                error_message="Crossref has no record for this DOI (HTTP 404)")
        if not result.ok:
            self.fetcher.run_log.add_error(result.to_error(DATABASE, "verification", query))
            return LookupResult(**base, outcome="lookup_failed", http_status=result.http_status,
                                error_type=result.error_type, error_message=result.message)
        try:
            work = parse_work(result.data)
        except MalformedResponseError as exc:
            error = self._malformed(result, "verification", query, str(exc))
            return LookupResult(**base, outcome="lookup_failed", http_status=result.http_status,
                                error_type=error.error_type, error_message=error.message)
        if result.entry is not None:
            result.entry.result_count = 1
        return LookupResult(**base, outcome="resolved", http_status=result.http_status, record=record_from_work(work))
