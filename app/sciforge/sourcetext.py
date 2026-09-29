"""v0.3 source-text layer: code-fetched abstracts of already-verified v0.2 records.

Governing principle: "V0.2 remains the authority for source identity. V0.3 can
interpret verified sources, but it cannot create citations."

This module never creates, edits or enriches a v0.2 :class:`~sciforge.models.Record`.
It only reads a record's ``record_id`` / ``pmid`` / ``doi`` (to know *which*
abstract to fetch) and the v0.2 :class:`~sciforge.models.VerificationResult`
status (to decide *whether* the record is eligible). A :class:`SourceText`
carries the opaque ``record_id`` and the abstract text, never bibliographic
fields (title, authors, journal, publication date, identifiers, links).

Retrieval rules
---------------
* Eligibility (D2): v0.2 status ``verified`` by default; ``partially_verified``
  only with ``include_partially_verified=True`` (``SCIFORGE_MODEL_ELIGIBILITY=
  verified_or_partial``), and then every SourceText carries
  ``eligibility="partially_verified_opt_in"`` / ``verification_status``.
  Records whose ``record_id`` is not an opaque v0.2 id (``rec_`` + 16 hex) are
  not eligible (the id is the only thing the model may see).
* Order (deterministic): verified before partially verified; within that,
  records with a PMID before DOI-only records; then v0.2 order (the order of
  ``records`` as written to ``sources.json``). Sources are admitted through
  ``BudgetTracker.admit_sources`` in that order until ``max_sources`` sources
  **with text** have been admitted; later candidates are not fetched
  (status ``source_limit``).
* PubMed (``access_level="pubmed_abstract"``): ``efetch.fcgi?db=pubmed&
  retmode=xml&id=...`` in batches (≤ 200 ids), with v0.2 ``tool``/``email``/
  ``api_key`` etiquette, throttling, bounded retries and redacted request logs
  (reuses :class:`~sciforge.http_utils.HttpFetcher`). ``AbstractText`` elements
  are flattened (nested markup such as ``<i>``/``<sup>`` keeps its text) and
  labelled sections are joined as ``"LABEL: text"``, separated by one space.
* Crossref (``access_level="crossref_abstract"``; D10): only when the record has
  a DOI and PubMed gave no abstract (no PMID, or PubMed has no abstract). The
  source-text layer re-fetches ``GET /works/{doi}`` (v0.2 records do not retain
  the ``abstract`` field and ``models.Record`` is deliberately unchanged) and
  uses only the ``abstract`` that Crossref itself returns publicly (JATS, tags
  stripped). Publisher pages are never fetched. A PubMed fetch *failure* does
  not fall back to Crossref (status ``fetch_failed``; provenance stays clear).
* Cap: text longer than ``max_source_chars`` (``SCIFORGE_MAX_SOURCE_CHARS``,
  default 4000) is cut at the last word boundary at or before the cap and
  marked ``truncated``. ``sha256`` is computed over the exact (capped) text that
  is sent to the model; ``original_chars`` is the uncapped length.

XML safety: NCBI responses are parsed with the standard-library
``xml.etree.ElementTree`` (expat), which never fetches external entities or
DTDs. In addition, any document containing an ``<!ENTITY`` declaration or a
DOCTYPE internal subset is rejected before parsing (blocks entity-expansion
attacks), and bodies larger than :data:`MAX_XML_BYTES` are refused. The plain
``<!DOCTYPE PubmedArticleSet PUBLIC ...>`` line NCBI sends is allowed.
"""

from __future__ import annotations

import hashlib
import html
import os
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import quote

import httpx
from pydantic import BaseModel, Field

from sciforge.config import ConfigError, _clean
from sciforge.crossref import WORKS_URL, parse_work
from sciforge.http_utils import FetchResult, HttpFetcher, MalformedResponseError, RateLimiter
from sciforge.logging_utils import iso_utc
from sciforge.models import ErrorEntry, Record, VerificationResult
from sciforge.normalize import normalize_doi, normalize_pmid
from sciforge.pubmed import EUTILS_BASE, PubMedClient

__all__ = [
    "DEFAULT_MAX_SOURCE_CHARS", "EFETCH_URL", "SourceText", "SourceTextBatch", "cap_text", "jats_to_text",
    "max_source_chars_from_env", "parse_efetch_abstracts", "prepare_source_texts", "sha256_text",
]

EFETCH_URL = f"{EUTILS_BASE}/efetch.fcgi"
EFETCH_BATCH_SIZE = 200
MAX_XML_BYTES = 10 * 1024 * 1024
DEFAULT_MAX_SOURCE_CHARS = 4000
MIN_MAX_SOURCE_CHARS = 200
MAX_MAX_SOURCE_CHARS = 50_000
MAX_SOURCE_CHARS_ENV = "SCIFORGE_MAX_SOURCE_CHARS"
STAGE = "source_text"

OPAQUE_RECORD_ID = re.compile(r"^rec_[0-9a-f]{16}$")

AccessLevel = Literal["pubmed_abstract", "crossref_abstract", "not_accessed"]
RetrievalStatus = Literal["ok", "no_abstract", "not_eligible", "fetch_failed", "parse_error", "source_limit"]
Eligibility = Literal["verified", "partially_verified_opt_in", "not_eligible"]

ORDERING_RULE = ("verified before partially_verified (opt-in); PMID before DOI-only; then v0.2 record order; "
                 "admitted via BudgetTracker.admit_sources until max_sources sources with text")


def max_source_chars_from_env(environ: Mapping[str, str] | None = None) -> int:
    """``SCIFORGE_MAX_SOURCE_CHARS`` (default 4000, range 200–50000).

    Raises :class:`~sciforge.config.ConfigError` for invalid values.
    """
    env = os.environ if environ is None else environ
    raw = _clean(env.get(MAX_SOURCE_CHARS_ENV))
    if raw is None:
        return DEFAULT_MAX_SOURCE_CHARS
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{MAX_SOURCE_CHARS_ENV} must be an integer") from exc
    if not MIN_MAX_SOURCE_CHARS <= value <= MAX_MAX_SOURCE_CHARS:
        raise ConfigError(f"{MAX_SOURCE_CHARS_ENV} must be between {MIN_MAX_SOURCE_CHARS} and {MAX_MAX_SOURCE_CHARS}")
    return value


# ============================================================ model


class SourceText(BaseModel):
    """Code-built abstract text for one v0.2 record (``source_texts.json``).

    Contains the opaque v0.2 ``record_id`` and NO bibliographic fields.
    """

    record_id: str
    verification_status: str | None = Field(description="v0.2 bibliographic verification status (label only)")
    eligibility: Eligibility
    access_level: AccessLevel = "not_accessed"
    origin: Literal["pubmed", "crossref"] | None = None
    abstract_only: Literal[True] = True
    status: RetrievalStatus
    source_text: str | None = None
    sha256: str | None = Field(default=None, description="SHA-256 of the exact (capped) text sent to the model")
    original_chars: int | None = None
    text_chars: int | None = None
    truncated: bool = False
    max_source_chars: int
    retrieved_at: str
    error: dict[str, Any] | None = None

    @property
    def usable(self) -> bool:
        return self.status == "ok" and bool(self.source_text)


@dataclass
class SourceTextBatch:
    """Result of :func:`prepare_source_texts`."""

    sources: list[SourceText]
    requests: list[dict[str, Any]]
    errors: list[dict[str, Any]]
    include_partially_verified: bool
    max_source_chars: int
    max_sources: int

    @property
    def usable(self) -> list[SourceText]:
        """Sources whose text will be sent to the model (in admission order)."""
        return [s for s in self.sources if s.usable]

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in self.sources:
            out[s.status] = out.get(s.status, 0) + 1
        return dict(sorted(out.items()))

    def to_json(self) -> dict[str, Any]:
        return {
            "eligibility_policy": "verified_or_partial" if self.include_partially_verified else "verified",
            "abstract_only": True,
            "max_source_chars": self.max_source_chars,
            "max_sources": self.max_sources,
            "ordering": ORDERING_RULE,
            "note": ("Abstracts only. Source texts carry the opaque v0.2 record_id only; bibliographic data stays "
                     "in the v0.2 sources.json/verification.json records."),
            "counts": self.counts(),
            "sent_to_model": [s.record_id for s in self.usable],
            "sources": [s.model_dump() for s in self.sources],
            "requests": self.requests,
            "errors": self.errors,
        }


# ============================================================ text helpers

_WS_RE = re.compile(r"\s+")


def _collapse(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def sha256_text(text: str) -> str:
    """SHA-256 hex digest of ``text`` encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cap_text(text: str, max_chars: int) -> tuple[str, bool]:
    """Cut ``text`` to at most ``max_chars`` characters on a word boundary.

    Returns (capped_text, truncated). If there is no whitespace in the second
    half of the window, the cut is made exactly at ``max_chars``.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    if len(text) <= max_chars:
        return text, False
    window = text[:max_chars]
    if not text[max_chars].isspace():
        cut = max(window.rfind(" "), window.rfind("\n"), window.rfind("\t"))
        if cut >= max_chars // 2:
            window = window[:cut]
    return window.rstrip(), True


# JATS tags whose content is inline (joined without a space).
_JATS_INLINE = {"italic", "bold", "sub", "sup", "sc", "underline", "monospace", "roman", "strike", "inline-formula",
                "named-content", "styled-content", "ext-link", "xref", "email", "uri", "i", "b", "em", "strong", "span"}
_TAG_RE = re.compile(r"<(/?)([A-Za-z_][\w.\-]*:)?([A-Za-z_][\w.\-]*)[^>]*?(/?)>", re.DOTALL)
_JATS_TITLE_RE = re.compile(r"<(?:[\w.\-]+:)?title\b[^>]*>(.*?)</(?:[\w.\-]+:)?title\s*>", re.DOTALL | re.IGNORECASE)


def jats_to_text(jats: str | None) -> str | None:
    """Plain text from a Crossref JATS ``abstract`` string (None if empty).

    Section titles become ``"Title: "`` labels (a lone ``Abstract`` title is
    dropped); inline markup keeps its text; other tags become spaces; HTML
    entities are unescaped; whitespace runs are collapsed. Regex-based on
    purpose: Crossref fragments often use undeclared ``jats:`` prefixes that a
    strict XML parser rejects, and no entity is ever resolved beyond HTML's.
    """
    if not isinstance(jats, str) or not jats.strip():
        return None

    def title(match: re.Match[str]) -> str:
        inner = _collapse(html.unescape(_TAG_RE.sub("", match.group(1))))
        if not inner or inner.lower() == "abstract":
            return " "
        return f" {inner.rstrip(':')}: "

    text = _JATS_TITLE_RE.sub(title, jats)

    def tag(match: re.Match[str]) -> str:
        return "" if match.group(3).lower() in _JATS_INLINE else " "

    text = _TAG_RE.sub(tag, text)
    text = _collapse(html.unescape(text))
    return text or None


# ============================================================ PubMed efetch XML

_ENTITY_DECL_RE = re.compile(r"<!ENTITY", re.IGNORECASE)
_DOCTYPE_SUBSET_RE = re.compile(r"<!DOCTYPE[^>\[]*\[", re.IGNORECASE)


def _check_xml_safety(xml_text: str) -> None:
    if len(xml_text.encode("utf-8", errors="ignore")) > MAX_XML_BYTES:
        raise MalformedResponseError("efetch XML exceeds the size limit")
    if _ENTITY_DECL_RE.search(xml_text) or _DOCTYPE_SUBSET_RE.search(xml_text):
        raise MalformedResponseError("efetch XML contains an entity declaration or DOCTYPE internal subset (refused)")


def _abstract_from_article(article: ET.Element) -> str | None:
    abstract = article.find("./MedlineCitation/Article/Abstract")
    if abstract is None:
        return None
    parts: list[str] = []
    for node in abstract.findall("AbstractText"):
        body = _collapse("".join(node.itertext()))
        if not body:
            continue
        label = _collapse(node.get("Label") or "")
        parts.append(f"{label}: {body}" if label else body)
    return " ".join(parts) or None


def parse_efetch_abstracts(xml_text: str) -> dict[str, str | None]:
    """``{pmid: abstract or None}`` for every ``PubmedArticle`` in an efetch XML body.

    Raises :class:`~sciforge.http_utils.MalformedResponseError` for unsafe or
    malformed XML or an unexpected root element.
    """
    if not isinstance(xml_text, str) or not xml_text.strip():
        raise MalformedResponseError("efetch response is empty")
    _check_xml_safety(xml_text)
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise MalformedResponseError(f"efetch XML could not be parsed ({exc})") from None
    if root.tag != "PubmedArticleSet":
        raise MalformedResponseError(f"unexpected efetch root element {root.tag[:50]!r}")
    out: dict[str, str | None] = {}
    for article in root.findall("PubmedArticle"):
        pmid = normalize_pmid(article.findtext("./MedlineCitation/PMID"))
        if pmid is None or pmid in out:
            continue
        out[pmid] = _abstract_from_article(article)
    return out


class _TextFetcher(HttpFetcher):
    """:class:`HttpFetcher` variant whose successful result is the response *text*.

    Reuses the v0.2 retry loop, Retry-After handling, throttling and redacted
    request logging of ``HttpFetcher.get_json`` (only ``_attempt`` differs).
    """

    @classmethod
    def wrap(cls, fetcher: HttpFetcher) -> _TextFetcher:
        return cls(fetcher.client, fetcher.settings, fetcher.run_log, sleep=fetcher.sleep, now=fetcher.now,
                   clock=fetcher.clock)

    def _attempt(self, url: str, params: Mapping[str, Any]) -> tuple[FetchResult, bool, httpx.Response | None]:
        try:
            response = self.client.get(
                url, params=dict(params),
                headers={"User-Agent": self.settings.user_agent, "Accept": "application/xml, text/xml"},
                timeout=self.settings.timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            return FetchResult(False, None, error_type="timeout", message=f"request timed out ({type(exc).__name__})"), True, None
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            return FetchResult(False, None, error_type="connection_error", message=f"connection failed ({type(exc).__name__}: {exc})"), True, None
        except httpx.TransportError as exc:
            return FetchResult(False, None, error_type="transport_error", message=f"transport error ({type(exc).__name__}: {exc})"), True, None
        except Exception as exc:  # noqa: BLE001 - a run must never crash on one request
            return FetchResult(False, None, error_type="request_error", message=f"request error ({type(exc).__name__}: {exc})"), False, None
        status = response.status_code
        if status == 429:
            return FetchResult(False, status, error_type="rate_limited", message="HTTP 429 Too Many Requests"), True, response
        if status >= 500:
            return FetchResult(False, status, error_type="http_error", message=f"HTTP {status} server error"), True, response
        if status >= 300:
            return FetchResult(False, status, error_type="http_error", message=f"HTTP {status} response"), False, response
        if len(response.content) > MAX_XML_BYTES:
            return FetchResult(False, status, error_type="response_too_large", message="response body exceeds the size limit"), False, response
        return FetchResult(True, status, data=response.text), False, response

    def get_text(self, **kwargs: Any) -> FetchResult:
        return self.get_json(**kwargs)


# ============================================================ orchestration


@dataclass
class _Candidate:
    index: int
    record: Record
    status: str
    eligibility: str


def _eligibility(record: Record, verification: VerificationResult | None,
                 include_partially_verified: bool) -> tuple[str, str | None]:
    """(eligibility label, reason if not eligible)."""
    status = verification.status if verification is not None else None
    if not OPAQUE_RECORD_ID.match(record.record_id or ""):
        return "not_eligible", "record_id is not an opaque v0.2 record id"
    if not (record.pmid or record.doi):
        return "not_eligible", "record has no PMID or DOI to fetch an abstract with"
    if status == "verified":
        return "verified", None
    if status == "partially_verified":
        if include_partially_verified:
            return "partially_verified_opt_in", None
        return "not_eligible", "partially_verified records are excluded unless explicitly opted in"
    return "not_eligible", f"v0.2 verification status is {status or 'missing'}"


def _order_key(c: _Candidate) -> tuple[int, int, int]:
    return (0 if c.eligibility == "verified" else 1, 0 if c.record.pmid else 1, c.index)


def prepare_source_texts(
    records: Sequence[Record],
    verification: Iterable[VerificationResult],
    *,
    fetcher: HttpFetcher,
    tracker: Any,
    include_partially_verified: bool = False,
    max_source_chars: int = DEFAULT_MAX_SOURCE_CHARS,
    pubmed_limiter: RateLimiter | None = None,
    crossref_limiter: RateLimiter | None = None,
    now: Callable[[], str] = iso_utc,
) -> SourceTextBatch:
    """Fetch abstracts for eligible v0.2 records (read-only; records are not modified).

    ``tracker`` is a :class:`~sciforge.llm.budget.BudgetTracker`; its
    ``admit_sources`` enforces ``max_sources``. Never raises for network or
    parse problems; they become per-source statuses and ``errors`` entries.
    """
    settings = fetcher.settings
    text_fetcher = _TextFetcher.wrap(fetcher)
    pubmed_limiter = pubmed_limiter or fetcher.make_limiter(settings.pubmed_min_interval)
    crossref_limiter = crossref_limiter or fetcher.make_limiter(settings.crossref_min_interval)
    pubmed_params = PubMedClient(fetcher, settings, limiter=pubmed_limiter)._base_params()
    crossref_params = {"mailto": settings.contact_email} if settings.contact_email else {}
    entries_before = len(fetcher.run_log.entries)
    errors_before = len(fetcher.run_log.errors)

    vmap = {v.record_id: v for v in verification}
    results: dict[int, SourceText] = {}
    candidates: list[_Candidate] = []
    for index, record in enumerate(records):
        v = vmap.get(record.record_id)
        label, reason = _eligibility(record, v, include_partially_verified)
        if reason is not None:
            results[index] = SourceText(record_id=record.record_id, verification_status=v.status if v else None,
                                        eligibility="not_eligible", status="not_eligible",
                                        max_source_chars=max_source_chars, retrieved_at=now(),
                                        error={"error_type": "not_eligible", "message": reason})
            continue
        candidates.append(_Candidate(index, record, v.status, label))  # type: ignore[union-attr]
    candidates.sort(key=_order_key)

    pubmed_cache: dict[str, str | None] = {}
    pubmed_failed: dict[str, dict[str, Any]] = {}

    def efetch(pmids: list[str]) -> None:
        todo = [p for p in dict.fromkeys(pmids) if p not in pubmed_cache and p not in pubmed_failed]
        for start in range(0, len(todo), EFETCH_BATCH_SIZE):
            batch = todo[start:start + EFETCH_BATCH_SIZE]
            query = f"efetch abstracts: {','.join(batch)}"
            result = text_fetcher.get_text(database="pubmed", stage=STAGE, url=EFETCH_URL,
                                           params={"db": "pubmed", "retmode": "xml", "id": ",".join(batch),
                                                   **pubmed_params},
                                           query=query, limiter=pubmed_limiter)
            if not result.ok:
                fetcher.run_log.add_error(result.to_error("pubmed", STAGE, query))
                err = {"status": "fetch_failed", "error_type": result.error_type, "http_status": result.http_status,
                       "message": result.message}
                for p in batch:
                    pubmed_failed[p] = err
                continue
            try:
                parsed = parse_efetch_abstracts(result.data)
            except MalformedResponseError as exc:
                error = ErrorEntry(database="pubmed", stage=STAGE, query=query, timestamp=iso_utc(),
                                   error_type="parse_error", http_status=result.http_status,
                                   message=fetcher.run_log.scrub(str(exc)),
                                   url=result.entry.url if result.entry else None)
                if result.entry is not None:
                    fetcher.run_log.mark_failed(result.entry, error)
                else:
                    fetcher.run_log.add_error(error)
                err = {"status": "parse_error", "error_type": "parse_error", "http_status": result.http_status,
                       "message": error.message}
                for p in batch:
                    pubmed_failed[p] = err
                continue
            if result.entry is not None:
                result.entry.result_count = len(parsed)
            for p in batch:
                pubmed_cache[p] = parsed.get(p)

    def crossref_abstract(doi: str) -> tuple[str | None, dict[str, Any] | None]:
        normalized = normalize_doi(doi) or doi
        url = f"{WORKS_URL}/{quote(normalized, safe='/')}"
        query = f"crossref abstract: {normalized}"
        result = fetcher.get_json(database="crossref", stage=STAGE, url=url, params=dict(crossref_params),
                                  query=query, limiter=crossref_limiter)
        if not result.ok:
            fetcher.run_log.add_error(result.to_error("crossref", STAGE, query))
            return None, {"status": "fetch_failed", "error_type": result.error_type,
                          "http_status": result.http_status, "message": result.message}
        try:
            work = parse_work(result.data)
        except MalformedResponseError as exc:
            return None, {"status": "parse_error", "error_type": "parse_error", "http_status": result.http_status,
                          "message": str(exc)}
        return jats_to_text(work.get("abstract") if isinstance(work.get("abstract"), str) else None), None

    def build(c: _Candidate) -> SourceText:
        base = {"record_id": c.record.record_id, "verification_status": c.status, "eligibility": c.eligibility,
                "max_source_chars": max_source_chars}
        text: str | None = None
        origin: str | None = None
        detail: dict[str, Any] | None = None
        pmid = c.record.pmid
        if pmid:
            if pmid in pubmed_failed:
                err = pubmed_failed[pmid]
                return SourceText(**base, status=err["status"], retrieved_at=now(),
                                  error={k: v for k, v in err.items() if k != "status"} | {"origin": "pubmed"})
            text = pubmed_cache.get(pmid)
            if text:
                origin = "pubmed"
            else:
                detail = {"pubmed": "no abstract in PubMed efetch" if pmid in pubmed_cache
                          else "PMID not returned by efetch"}
        if not text and c.record.doi:
            text, err = crossref_abstract(c.record.doi)
            if err is not None:
                if origin is None and not pmid:
                    return SourceText(**base, status=err["status"], retrieved_at=now(),
                                      error={k: v for k, v in err.items() if k != "status"} | {"origin": "crossref"})
                detail = (detail or {}) | {"crossref": err.get("message")}
            elif text:
                origin = "crossref"
            else:
                detail = (detail or {}) | {"crossref": "Crossref work has no abstract"}
        if not text:
            return SourceText(**base, status="no_abstract", retrieved_at=now(),
                              error={"error_type": "no_abstract", "message": "no abstract available", **(detail or {})})
        capped, truncated = cap_text(text, max_source_chars)
        return SourceText(**base, status="ok", access_level="pubmed_abstract" if origin == "pubmed" else "crossref_abstract",
                          origin=origin, source_text=capped, sha256=sha256_text(capped), original_chars=len(text),
                          text_chars=len(capped), truncated=truncated, retrieved_at=now())

    position = 0
    limited = False
    while position < len(candidates):
        slots = tracker.limits.max_sources - tracker.sources
        if slots <= 0:
            limited = True
            break
        window = candidates[position:position + slots]
        efetch([c.record.pmid for c in window if c.record.pmid])
        for c in window:
            st = build(c)
            if st.usable and tracker.admit_sources(1) != 1:  # defensive; slots were computed above
                st = st.model_copy(update={"status": "source_limit", "source_text": None, "sha256": None,
                                           "access_level": "not_accessed", "origin": None})
            results[c.index] = st
        position += len(window)
    if limited:
        remaining = candidates[position:]
        if remaining:
            tracker.admit_sources(len(remaining))  # records the limit being hit (admits 0)
        for c in remaining:
            results[c.index] = SourceText(record_id=c.record.record_id, verification_status=c.status,
                                          eligibility=c.eligibility, status="source_limit",
                                          max_source_chars=max_source_chars, retrieved_at=now(),
                                          error={"error_type": "source_limit",
                                                 "message": f"max_sources ({tracker.limits.max_sources}) reached"})

    ordered_candidates = [results[c.index] for c in candidates]
    not_eligible = [results[i] for i in sorted(results) if results[i].status == "not_eligible"]
    requests = [e.model_dump() for e in fetcher.run_log.entries[entries_before:]]
    errors = [e.model_dump() for e in fetcher.run_log.errors[errors_before:]]
    return SourceTextBatch(sources=ordered_candidates + not_eligible, requests=requests, errors=errors,
                           include_partially_verified=include_partially_verified,
                           max_source_chars=max_source_chars, max_sources=tracker.limits.max_sources)
