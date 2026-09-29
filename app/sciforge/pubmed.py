"""PubMed retrieval through NCBI E-utilities (esearch + esummary)."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from sciforge.config import Settings
from sciforge.http_utils import FetchResult, HttpFetcher, MalformedResponseError, RateLimiter
from sciforge.logging_utils import iso_utc
from sciforge.models import ErrorEntry, LookupResult, Record, SearchOutcome
from sciforge.normalize import normalize_doi, normalize_pmid
from sciforge.source_classification import pubmed_metadata

__all__ = ["PubMedClient", "normalize_pmid", "parse_esearch", "parse_esummary", "parse_pubmed_year", "record_from_summary"]

DATABASE = "pubmed"
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ESEARCH_URL = f"{EUTILS_BASE}/esearch.fcgi"
ESUMMARY_URL = f"{EUTILS_BASE}/esummary.fcgi"
RECORD_URL = "https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
ESUMMARY_BATCH_SIZE = 200
MAX_RETMAX = 1000
# NCBI requires both mindate and maxdate; open-ended bounds use these years.
OPEN_MIN_YEAR = 1800
OPEN_MAX_YEAR = 3000
MIN_PLAUSIBLE_YEAR = 1500
MAX_PLAUSIBLE_YEAR = 2100

_YEAR_RE = re.compile(r"^\s*(\d{4})(?!\d)")


def parse_pubmed_year(pubdate: Any, sortpubdate: Any = None) -> int | None:
    """Year from esummary ``pubdate`` (e.g. ``"2021 Mar 15"``) or, failing that,
    ``sortpubdate`` (``"2021/03/15 00:00"``). None when neither is parseable."""
    for value in (pubdate, sortpubdate):
        if isinstance(value, str):
            match = _YEAR_RE.match(value)
            if match:
                year = int(match.group(1))
                if MIN_PLAUSIBLE_YEAR <= year <= MAX_PLAUSIBLE_YEAR:
                    return year
    return None


def parse_esearch(data: Any) -> tuple[list[str], int | None]:
    """Return (PMIDs, total hit count) from an esearch JSON response.

    Raises:
        MalformedResponseError: if the structure is not as documented, or
            NCBI reported an error in the body.
    """
    if not isinstance(data, dict):
        raise MalformedResponseError("esearch response is not a JSON object")
    result = data.get("esearchresult")
    if not isinstance(result, dict):
        raise MalformedResponseError("esearch response has no 'esearchresult' object")
    if "ERROR" in result:
        raise MalformedResponseError(f"esearch reported an error: {str(result['ERROR'])[:200]}")
    idlist = result.get("idlist")
    if not isinstance(idlist, list):
        raise MalformedResponseError("esearch 'idlist' is missing or not a list")
    pmids: list[str] = []
    for raw in idlist:
        pmid = normalize_pmid(raw)
        if pmid is None:
            raise MalformedResponseError(f"esearch idlist contains a non-PMID value: {str(raw)[:50]!r}")
        if pmid not in pmids:
            pmids.append(pmid)
    count = result.get("count")
    total: int | None = None
    if isinstance(count, (str, int)) and not isinstance(count, bool) and str(count).isdigit():
        total = int(count)
    return pmids, total


def parse_esummary(data: Any) -> dict[str, dict[str, Any]]:
    """Return ``{pmid: document}`` from an esummary JSON response.

    Documents carrying an ``error`` key (unknown PMID) are omitted.

    Raises:
        MalformedResponseError: if the structure is not as documented.
    """
    if not isinstance(data, dict):
        raise MalformedResponseError("esummary response is not a JSON object")
    result = data.get("result")
    if not isinstance(result, dict):
        if "error" in data:
            raise MalformedResponseError(f"esummary reported an error: {str(data['error'])[:200]}")
        raise MalformedResponseError("esummary response has no 'result' object")
    uids = result.get("uids")
    if not isinstance(uids, list):
        raise MalformedResponseError("esummary 'uids' is missing or not a list")
    docs: dict[str, dict[str, Any]] = {}
    for uid in uids:
        pmid = normalize_pmid(uid)
        doc = result.get(str(uid))
        if pmid is None or not isinstance(doc, dict) or "error" in doc:
            continue
        docs[pmid] = doc
    return docs


def _authors(doc: dict[str, Any]) -> list[str]:
    authors = doc.get("authors")
    if not isinstance(authors, list):
        return []
    names: list[str] = []
    for author in authors:
        if not isinstance(author, dict):
            continue
        authtype = author.get("authtype", "Author")
        name = author.get("name")
        if authtype == "Author" and isinstance(name, str) and name.strip():
            names.append(name.strip())
    return names


def _doi(doc: dict[str, Any]) -> str | None:
    articleids = doc.get("articleids")
    if not isinstance(articleids, list):
        return None
    for item in articleids:
        if isinstance(item, dict) and str(item.get("idtype", "")).lower() == "doi":
            doi = normalize_doi(item.get("value"))
            if doi:
                return doi
    return None


def _str_or_none(value: Any) -> str | None:
    return value.strip() or None if isinstance(value, str) else None


def record_from_summary(pmid: str, doc: dict[str, Any], retrieved_at: str | None = None) -> Record:
    """Build a :class:`Record` from one esummary document. Missing fields stay None."""
    return Record(
        title=_str_or_none(doc.get("title")),
        authors=_authors(doc),
        year=parse_pubmed_year(doc.get("pubdate"), doc.get("sortpubdate")),
        doi=_doi(doc),
        pmid=pmid,
        journal=_str_or_none(doc.get("fulljournalname")) or _str_or_none(doc.get("source")),
        source_database=DATABASE,
        source_url=RECORD_URL.format(pmid=pmid),
        retrieval_timestamp=retrieved_at or iso_utc(),
    )


def _chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


class PubMedClient:
    """Searches PubMed and resolves PMIDs via NCBI E-utilities."""

    database = DATABASE

    def __init__(self, fetcher: HttpFetcher, settings: Settings, limiter: RateLimiter | None = None) -> None:
        self.fetcher = fetcher
        self.settings = settings
        self.limiter = limiter or fetcher.make_limiter(settings.pubmed_min_interval)
        # Side data captured from esummary search results (Records are unchanged):
        # record_id -> publication types + journal (used for source-type classification only).
        self.source_metadata: dict[str, dict[str, Any]] = {}

    def _base_params(self) -> dict[str, str]:
        params = {"tool": "sciforge"}
        if self.settings.contact_email:
            params["email"] = self.settings.contact_email
        if self.settings.ncbi_api_key:
            params["api_key"] = self.settings.ncbi_api_key
        return params

    def _get(self, stage: str, url: str, params: dict[str, Any], query: str | None) -> FetchResult:
        return self.fetcher.get_json(
            database=DATABASE, stage=stage, url=url, params={**params, **self._base_params()}, query=query, limiter=self.limiter
        )

    def _error(self, result: FetchResult, stage: str, query: str | None) -> None:
        self.fetcher.run_log.add_error(result.to_error(DATABASE, stage, query))

    def _malformed(self, result: FetchResult, stage: str, query: str | None, exc: Exception) -> None:
        error = ErrorEntry(
            database=DATABASE,
            stage=stage,
            query=query,
            timestamp=iso_utc(),
            error_type="unexpected_structure",
            http_status=result.http_status,
            message=str(exc),
            url=result.entry.url if result.entry else None,
        )
        if result.entry is not None:
            self.fetcher.run_log.mark_failed(result.entry, error)
        else:
            self.fetcher.run_log.add_error(error)

    def esearch(
        self, query: str, max_results: int, from_year: int | None = None, to_year: int | None = None
    ) -> tuple[list[str], int | None] | None:
        """Run esearch. Returns (pmids, total) or None on failure (error recorded)."""
        params: dict[str, Any] = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": max(0, min(max_results, MAX_RETMAX)),
        }
        if from_year is not None or to_year is not None:
            params.update(
                datetype="pdat",
                mindate=str(from_year if from_year is not None else OPEN_MIN_YEAR),
                maxdate=str(to_year if to_year is not None else OPEN_MAX_YEAR),
            )
        result = self._get("search", ESEARCH_URL, params, query)
        if not result.ok:
            self._error(result, "search", query)
            return None
        try:
            pmids, total = parse_esearch(result.data)
        except MalformedResponseError as exc:
            self._malformed(result, "search", query, exc)
            return None
        if result.entry is not None:
            result.entry.result_count = len(pmids)
        return pmids, total

    def esummary(self, pmids: list[str], stage: str, query: str | None) -> tuple[dict[str, dict[str, Any]], FetchResult | None]:
        """Fetch esummary documents for ``pmids`` (one batch).

        Returns (docs, failed_result). ``failed_result`` is None on success.
        """
        params = {"db": "pubmed", "retmode": "json", "id": ",".join(pmids)}
        result = self._get(stage, ESUMMARY_URL, params, query)
        if not result.ok:
            return {}, result
        try:
            docs = parse_esummary(result.data)
        except MalformedResponseError as exc:
            self._malformed(result, stage, query, exc)
            result.ok = False
            result.error_type = "unexpected_structure"
            result.message = str(exc)
            return {}, result
        if result.entry is not None:
            result.entry.result_count = len(docs)
        return docs, None

    def search(
        self, query: str, max_results: int, from_year: int | None = None, to_year: int | None = None
    ) -> SearchOutcome:
        """Search PubMed and return normalized records. Never raises for API problems."""
        found = self.esearch(query, max_results, from_year, to_year)
        if found is None:
            return SearchOutcome(database=DATABASE, status="failed")
        pmids, total = found
        records: list[Record] = []
        had_error = False
        for batch in _chunks(pmids, ESUMMARY_BATCH_SIZE):
            docs, failed = self.esummary(batch, "summary", query)
            if failed is not None:
                had_error = True
                if failed.error_type != "unexpected_structure":  # already recorded
                    self._error(failed, "summary", query)
                continue
            retrieved_at = iso_utc()
            for pmid in batch:
                doc = docs.get(pmid)
                if doc is None:
                    had_error = True
                    self.fetcher.run_log.add_error(
                        ErrorEntry(
                            database=DATABASE,
                            stage="summary",
                            query=query,
                            timestamp=retrieved_at,
                            error_type="missing_summary",
                            message=f"esummary returned no document for PMID {pmid}",
                        )
                    )
                    continue
                record = record_from_summary(pmid, doc, retrieved_at)
                records.append(record)
                self.source_metadata[record.record_id] = pubmed_metadata(doc, record.doi)
        if had_error:
            status = "partial" if records else "failed"
        else:
            status = "ok"
        return SearchOutcome(database=DATABASE, status=status, records=records, total_hits=total)

    def lookup_many(self, pmids: Iterable[str]) -> dict[str, LookupResult]:
        """Resolve PMIDs via esummary for verification.

        A PMID is ``not_found`` only when esummary succeeds but returns no
        document (or an error document) for it. Any request failure (including
        HTTP 404 from the endpoint itself) gives ``lookup_failed``.
        """
        unique: list[str] = []
        for raw in pmids:
            pmid = normalize_pmid(raw)
            if pmid and pmid not in unique:
                unique.append(pmid)
        results: dict[str, LookupResult] = {}
        for batch in _chunks(unique, ESUMMARY_BATCH_SIZE):
            query = f"verify PMIDs: {','.join(batch)}"
            docs, failed = self.esummary(batch, "verification", query)
            if failed is not None:
                if failed.error_type != "unexpected_structure":
                    self._error(failed, "verification", query)
                for pmid in batch:
                    results[pmid] = LookupResult(
                        database=DATABASE,
                        identifier_type="pmid",
                        identifier=pmid,
                        outcome="lookup_failed",
                        http_status=failed.http_status,
                        error_type=failed.error_type,
                        error_message=failed.message,
                    )
                continue
            retrieved_at = iso_utc()
            for pmid in batch:
                doc = docs.get(pmid)
                if doc is None:
                    results[pmid] = LookupResult(
                        database=DATABASE,
                        identifier_type="pmid",
                        identifier=pmid,
                        outcome="not_found",
                        http_status=200,
                        error_type="not_found",
                        error_message="esummary returned no document for this PMID",
                    )
                else:
                    results[pmid] = LookupResult(
                        database=DATABASE,
                        identifier_type="pmid",
                        identifier=pmid,
                        outcome="resolved",
                        http_status=200,
                        record=record_from_summary(pmid, doc, retrieved_at),
                    )
        return results
