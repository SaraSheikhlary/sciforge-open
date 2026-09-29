"""Bounded abstract enrichment for deterministic source selection (no model).

Search results carry titles only (PubMed esummary) or sometimes an abstract
(Crossref ``/works`` items include the publicly deposited ``abstract`` field
when a publisher supplied one). Before the final relevance ranking, SciForge
enriches AT MOST ``SCIFORGE_ABSTRACT_ENRICHMENT_LIMIT`` deduplicated candidates
(default 50, integer 0-500; 0 disables enrichment and restores title-only
scoring) with abstract text:

* Choice (deterministic): candidates sorted by their title/provenance
  pre-score (descending), then best search rank (ascending), then
  ``record_id`` (ascending); the first ``limit`` are "considered".
* PubMed abstracts: ONE batched ``efetch`` request per 200 PMIDs of the
  considered candidates (re-uses the v0.3 source-text efetch parser
  :func:`sciforge.sourcetext.parse_efetch_abstracts`, its XML safety checks
  and the v0.2 ``HttpFetcher`` retry/throttle/redaction).
* Crossref abstracts: only the ``abstract`` field already returned in the
  Crossref search response (no extra request; JATS converted with
  :func:`sciforge.sourcetext.jats_to_text`). A PubMed abstract wins when both
  exist.
* Candidates outside the limit, or without any retrieved abstract, stay
  title-only. Failures never stop the run (the candidate stays title-only and
  the error is logged).

Abstracts are used ONLY for ranking. They are not stored in the logs (only
origin and length), are not added to the v0.2 records, and are fetched again
by the v0.3 source-text layer (unchanged) for the sources sent to the model.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from sciforge.models import Record

__all__ = ["DEFAULT_ABSTRACT_ENRICHMENT_LIMIT", "ENRICHMENT_BATCH_SIZE", "MAX_ABSTRACT_ENRICHMENT_LIMIT",
           "EnrichmentOutcome", "choose_enrichment_candidates", "enrich_abstracts", "make_pubmed_abstract_fetcher"]

DEFAULT_ABSTRACT_ENRICHMENT_LIMIT = 50
MIN_ABSTRACT_ENRICHMENT_LIMIT = 0
MAX_ABSTRACT_ENRICHMENT_LIMIT = 500
ABSTRACT_ENRICHMENT_ENV = "SCIFORGE_ABSTRACT_ENRICHMENT_LIMIT"
ENRICHMENT_BATCH_SIZE = 200
STAGE = "abstract_enrichment"
CHOICE_RULE = ("top candidates by title/provenance pre-score (desc), then best search rank (asc), then record_id "
               "(asc); at most SCIFORGE_ABSTRACT_ENRICHMENT_LIMIT")

# pmids -> {pmid: abstract or None}; raises on failure.
PubMedAbstractFetcher = Callable[[list[str]], Mapping[str, str | None]]


@dataclass
class EnrichmentOutcome:
    limit: int
    considered: list[str] = field(default_factory=list)
    abstracts: dict[str, str] = field(default_factory=dict)
    origin: dict[str, str] = field(default_factory=dict)
    title_only: dict[str, str] = field(default_factory=dict)       # record_id -> reason
    pubmed_batches: list[list[str]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "limit": self.limit,
            "enabled": self.limit > 0,
            "choice_rule": CHOICE_RULE,
            "candidates_considered": list(self.considered),
            "with_abstract": [{"record_id": rid, "abstract_origin": self.origin[rid],
                               "abstract_chars": len(self.abstracts[rid])} for rid in self.considered
                              if rid in self.abstracts],
            "title_only": [{"record_id": rid, "reason": reason} for rid, reason in self.title_only.items()],
            "pubmed_efetch_batches": len(self.pubmed_batches),
            "pubmed_ids_requested": sum(len(b) for b in self.pubmed_batches),
            "errors": list(self.errors),
            "note": "abstract text is used for ranking only and is not stored in this log",
        }

    def counts(self) -> dict[str, int]:
        return {"limit": self.limit, "considered": len(self.considered), "with_abstract": len(self.abstracts),
                "title_only_considered": len(self.considered) - len(self.abstracts),
                "pubmed_efetch_batches": len(self.pubmed_batches)}


def choose_enrichment_candidates(prescores: Sequence[Any], limit: int) -> list[str]:
    """Deterministic choice of at most ``limit`` record ids (see module docstring)."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("abstract enrichment limit must be a non-negative integer")
    ordered = sorted(prescores, key=lambda s: (-s.base_score, s.best_rank if s.best_rank is not None else 10**6,
                                               s.record_id))
    return [s.record_id for s in ordered[:limit]]


def enrich_abstracts(records_by_id: Mapping[str, Record], considered: Sequence[str], *, limit: int,
                     inline_abstracts: Mapping[str, str] | None = None,
                     fetch_pubmed: PubMedAbstractFetcher | None = None,
                     batch_size: int = ENRICHMENT_BATCH_SIZE) -> EnrichmentOutcome:
    """Fetch/collect abstracts for ``considered`` (never more than ``limit`` ids; extra ids are ignored)."""
    chosen = list(dict.fromkeys(considered))[:max(0, limit)]
    out = EnrichmentOutcome(limit=limit, considered=chosen)
    if not chosen:
        return out
    inline = inline_abstracts or {}
    pmids = list(dict.fromkeys(records_by_id[rid].pmid for rid in chosen if records_by_id[rid].pmid))
    pubmed: dict[str, str | None] = {}
    failed: set[str] = set()
    if pmids and fetch_pubmed is not None:
        for start in range(0, len(pmids), batch_size):
            batch = pmids[start:start + batch_size]
            out.pubmed_batches.append(batch)
            try:
                result = fetch_pubmed(batch)
            except Exception as exc:  # noqa: BLE001 - enrichment must never stop the run
                out.errors.append(f"pubmed efetch batch {len(out.pubmed_batches)} failed ({type(exc).__name__})")
                failed.update(batch)
                continue
            for p in batch:
                value = result.get(p) if isinstance(result, Mapping) else None
                pubmed[p] = value if isinstance(value, str) and value.strip() else None
    for rid in chosen:
        rec = records_by_id[rid]
        text = pubmed.get(rec.pmid) if rec.pmid else None
        if text:
            out.abstracts[rid], out.origin[rid] = text, "pubmed_efetch"
            continue
        crossref = inline.get(rid)
        if crossref:
            out.abstracts[rid], out.origin[rid] = crossref, "crossref_search_result"
            continue
        if rec.pmid and rec.pmid in failed:
            reason = "pubmed efetch failed; no Crossref abstract in search result"
        elif rec.pmid and fetch_pubmed is not None:
            reason = "no abstract in PubMed efetch or Crossref search result"
        else:
            reason = "no abstract in Crossref search result (no PMID)"
        out.title_only[rid] = reason
    return out


def make_pubmed_abstract_fetcher(fetcher: Any, settings: Any, limiter: Any = None) -> PubMedAbstractFetcher:
    """Real batched PubMed efetch (re-uses the v0.3 source-text efetch code path and XML safety checks)."""
    from sciforge.http_utils import MalformedResponseError
    from sciforge.pubmed import PubMedClient
    from sciforge.sourcetext import EFETCH_URL, _TextFetcher, parse_efetch_abstracts

    text_fetcher = _TextFetcher.wrap(fetcher)
    limiter = limiter or fetcher.make_limiter(settings.pubmed_min_interval)
    base = PubMedClient(fetcher, settings, limiter=limiter)._base_params()

    def fetch(pmids: list[str]) -> Mapping[str, str | None]:
        query = f"efetch abstracts for ranking: {','.join(pmids)}"
        result = text_fetcher.get_text(database="pubmed", stage=STAGE, url=EFETCH_URL,
                                       params={"db": "pubmed", "retmode": "xml", "id": ",".join(pmids), **base},
                                       query=query, limiter=limiter)
        if not result.ok:
            fetcher.run_log.add_error(result.to_error("pubmed", STAGE, query))
            raise RuntimeError(f"efetch failed: {result.error_type}")
        try:
            parsed = parse_efetch_abstracts(result.data)
        except MalformedResponseError as exc:
            from sciforge.logging_utils import iso_utc
            from sciforge.models import ErrorEntry

            error = ErrorEntry(database="pubmed", stage=STAGE, query=query, timestamp=iso_utc(),
                               error_type="parse_error", http_status=result.http_status,
                               message=fetcher.run_log.scrub(str(exc)))
            if result.entry is not None:
                fetcher.run_log.mark_failed(result.entry, error)
            else:
                fetcher.run_log.add_error(error)
            raise
        if result.entry is not None:
            result.entry.result_count = len(parsed)
        return parsed

    return fetch
