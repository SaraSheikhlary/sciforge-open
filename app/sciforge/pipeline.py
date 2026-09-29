"""Orchestrates one v0.2 investigation.

Flow: expand queries -> search (larger candidate pool per query and database)
-> merge -> dedup -> title pre-score -> bounded abstract enrichment ->
deterministic relevance/diversity scoring -> source-type classification and
source policy -> verify (with backfill) -> write files.

Query plan (:mod:`sciforge.query_expansion`, deterministic, no model): the
research question verbatim (always the first query) plus, unless disabled
(``SCIFORGE_QUERY_EXPANSION=false`` / ``query_expansion=False``), focused
anchor×facet queries from a curated concept map. Every query is run on PubMed
and Crossref, each requesting ``max(candidate_pool_per_query, max_results)``
records (``SCIFORGE_CANDIDATE_POOL_PER_QUERY``, default 10). Per database the
hits are merged (identical hits kept once; no cap), then:

1. the unchanged v0.2 dedup merges cross-database / cross-query duplicates;
2. every unique candidate is pre-scored from its title and query provenance;
   the top ``SCIFORGE_ABSTRACT_ENRICHMENT_LIMIT`` (default 50; 0 = off) are
   enriched with abstracts (batched PubMed efetch; Crossref abstracts already
   present in the search results) — :mod:`sciforge.abstract_enrichment`;
3. :mod:`sciforge.source_selection` scores every candidate (title components +
   separate, lower-weighted abstract components + provenance) and ranks the pool
   greedily for concept diversity (deterministic, stable tie-breaking);
4. :mod:`sciforge.source_classification` labels every candidate from
   bibliographic metadata (journal article / preprint / conference paper /
   book/chapter / unknown) and applies ``SCIFORGE_SOURCE_POLICY``
   (``peer_reviewed_preferred`` by default);
5. the unchanged v0.2 verifier checks the top ``max_selected`` candidates in
   policy order (default ``2 * max_results``; the web app passes the model's
   ``max_sources``); failures are backfilled from the ranking (at most
   ``3 * max_selected`` candidates verified in total).

``sources.json`` / ``verification.json`` contain every candidate that was sent
to verification (in deduplication order; each record in ``sources.json`` also
carries its ``source_status``); the selected ids are listed in ``summary.json``
(``selection.selected_record_ids``). The plan, per-query candidate counts,
dedup results, abstract enrichment, per-candidate score components, source
classification and policy decisions are written to ``search_log.json`` (and
summarised in ``summary.json``).
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from sciforge import __version__
from sciforge.abstract_enrichment import (
    EnrichmentOutcome,
    PubMedAbstractFetcher,
    choose_enrichment_candidates,
    enrich_abstracts,
    make_pubmed_abstract_fetcher,
)
from sciforge.config import MAX_ABSTRACT_ENRICHMENT_LIMIT, SOURCE_POLICIES, Settings
from sciforge.crossref import CrossrefClient
from sciforge.dedup import deduplicate
from sciforge.http_utils import HttpFetcher, Sleep
from sciforge.logging_utils import RunLog, iso_utc, redact_text, run_stamp, utc_now
from sciforge.models import ErrorEntry, Record, SearchOutcome, VerificationResult
from sciforge.pubmed import PubMedClient
from sciforge.query_expansion import QueryPlan, expand_queries, merge_query_outcomes, single_query_plan
from sciforge.source_classification import STATUS_NOTE, PolicyOutcome, SourceClassification, apply_source_policy
from sciforge.source_classification import POLICY_ONLY, classify_record
from sciforge.source_selection import METHOD as SELECTION_METHOD
from sciforge.source_selection import (
    VERIFY_BACKFILL_FACTOR,
    RankedCandidate,
    SelectionOutcome,
    rank_candidates,
    score_candidates,
    select_and_verify,
)
from sciforge.verify import JOURNAL_SIMILARITY_THRESHOLD, TITLE_SIMILARITY_THRESHOLD, Verifier

DATABASES = ("pubmed", "crossref")


@dataclass
class InvestigationResult:
    """What a completed run produced."""

    run_dir: Path
    summary: dict[str, Any]
    records: list[Record]
    verification: list[VerificationResult]


def make_run_dir(output_dir: Path, started: datetime) -> Path:
    """Create ``output_dir/<UTC stamp>`` (with ``-N`` suffix if it exists)."""
    base = output_dir / run_stamp(started)
    candidate, n = base, 1
    while candidate.exists():
        candidate = base.with_name(f"{base.name}-{n}")
        n += 1
    candidate.mkdir(parents=True)
    return candidate


def _write_json(path: Path, payload: Any, secrets: list[str]) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    path.write_text(redact_text(text, secrets) + "\n", encoding="utf-8")


def _safe_search(search: Callable[[], SearchOutcome], database: str, query: str, run_log: RunLog) -> SearchOutcome:
    """Run a database search; convert any unexpected exception into an error entry."""
    try:
        return search()
    except Exception as exc:  # noqa: BLE001 - one database must not crash the run
        run_log.add_error(ErrorEntry(database=database, stage="search", query=query, timestamp=iso_utc(),
                                     error_type="unexpected_error", message=f"{type(exc).__name__}: {exc}"))
        return SearchOutcome(database=database, status="failed")  # type: ignore[arg-type]


def run_investigation(
    question: str,
    *,
    max_results: int = 20,
    from_year: int | None = None,
    to_year: int | None = None,
    output_dir: str | Path = "runs",
    settings: Settings | None = None,
    client: httpx.Client | None = None,
    sleep: Sleep | None = None,
    now: Callable[[], datetime] = utc_now,
    query_expansion: bool | None = None,
    candidate_pool_per_query: int | None = None,
    max_selected: int | None = None,
    accept_partially_verified: bool = False,
    abstract_enrichment_limit: int | None = None,
    source_policy: str | None = None,
    abstract_fetcher: PubMedAbstractFetcher | None = None,
) -> InvestigationResult:
    """Run a full retrieve-select-verify investigation and write output files.

    Never raises for network or API problems; they are recorded in the logs.
    ``sleep`` defaults to :func:`time.sleep`, looked up at call time.
    ``query_expansion`` defaults to ``settings.query_expansion`` (env
    ``SCIFORGE_QUERY_EXPANSION``, default true). Each query requests
    ``max(candidate_pool_per_query, max_results)`` records per database
    (``candidate_pool_per_query`` defaults to ``settings.candidate_pool_per_query``).
    ``max_selected`` (default ``max_results * 2``) is the number of verified
    records the selection aims for; ``accept_partially_verified`` lets
    ``partially_verified`` records count towards it (model opt-in).
    ``abstract_enrichment_limit`` / ``source_policy`` default to the settings
    (``SCIFORGE_ABSTRACT_ENRICHMENT_LIMIT`` / ``SCIFORGE_SOURCE_POLICY``).
    ``abstract_fetcher`` (tests) replaces the batched PubMed efetch used for enrichment.
    """
    query = question.strip()
    if not query:
        raise ValueError("research question must not be empty")
    settings = settings or Settings.from_env()
    expand = settings.query_expansion if query_expansion is None else query_expansion
    plan: QueryPlan = expand_queries(query) if expand else single_query_plan(query)
    pool = settings.candidate_pool_per_query if candidate_pool_per_query is None else candidate_pool_per_query
    if isinstance(pool, bool) or not isinstance(pool, int) or pool < 1:
        raise ValueError("candidate_pool_per_query must be a positive integer")
    per_query = max(pool, max_results)
    target = max_results * len(DATABASES) if max_selected is None else max_selected
    if isinstance(target, bool) or not isinstance(target, int) or target < 1:
        raise ValueError("max_selected must be a positive integer")
    limit = settings.abstract_enrichment_limit if abstract_enrichment_limit is None else abstract_enrichment_limit
    if isinstance(limit, bool) or not isinstance(limit, int) or not 0 <= limit <= MAX_ABSTRACT_ENRICHMENT_LIMIT:
        raise ValueError(f"abstract_enrichment_limit must be an integer between 0 and {MAX_ABSTRACT_ENRICHMENT_LIMIT}")
    policy = settings.source_policy if source_policy is None else source_policy
    if policy not in SOURCE_POLICIES:
        raise ValueError(f"source_policy must be one of {', '.join(SOURCE_POLICIES)}")
    started = now()
    run_dir = make_run_dir(Path(output_dir), started)
    run_log = RunLog(settings.secret_values())

    own_client = client is None
    http = client or httpx.Client(follow_redirects=True, timeout=settings.timeout_seconds)
    try:
        fetcher = HttpFetcher(http, settings, run_log, sleep=sleep or time.sleep)
        pubmed = PubMedClient(fetcher, settings)
        crossref = CrossrefClient(fetcher, settings)

        per_query_outcomes: dict[str, list[SearchOutcome]] = {"pubmed": [], "crossref": []}
        for pq in plan.queries:
            per_query_outcomes["pubmed"].append(_safe_search(
                lambda: pubmed.search(pq.pubmed, per_query, from_year, to_year), "pubmed", pq.pubmed, run_log))
            per_query_outcomes["crossref"].append(_safe_search(
                lambda: crossref.search(pq.crossref, per_query, from_year, to_year), "crossref", pq.crossref,
                run_log))
        origins = _query_origins(plan, per_query_outcomes)
        outcomes = []
        query_stats: dict[str, list[dict[str, Any]]] = {}
        pool_cap = per_query * len(plan.queries)       # no truncation: every candidate enters dedup
        for database in DATABASES:
            merged_outcome, stats = merge_query_outcomes(database, per_query_outcomes[database], pool_cap)
            for st in stats:
                st["requested"] = per_query
            outcomes.append(merged_outcome)
            query_stats[database] = stats
        retrieved = [r for o in outcomes for r in o.records]
        unique, merged = deduplicate(retrieved)
        by_id = {r.record_id: r for r in unique}
        # Title/provenance pre-score -> bounded abstract enrichment -> final scores.
        prescores = score_candidates(query, unique, origins)
        considered = choose_enrichment_candidates(prescores, limit)
        fetch = abstract_fetcher if abstract_fetcher is not None else make_pubmed_abstract_fetcher(fetcher, settings,
                                                                                                    pubmed.limiter)
        enrichment = enrich_abstracts(by_id, considered, limit=limit,
                                      inline_abstracts=_inline_abstracts(unique, crossref.search_abstracts),
                                      fetch_pubmed=fetch if limit > 0 else None)
        scores = score_candidates(query, unique, origins, abstracts=enrichment.abstracts)
        ranked = rank_candidates(scores)
        classes = _classify(unique, pubmed.source_metadata, crossref.source_metadata)
        policy_outcome = apply_source_policy([c.record_id for c in ranked],
                                             {rid: c.source_status for rid, c in classes.items()},
                                             {c.record_id: c.score.text_relevant for c in ranked}, policy)
        ranked_by_id = {c.record_id: c for c in ranked}
        ordered = [ranked_by_id[rid] for rid in policy_outcome.ordered_ids]
        if ordered:
            selection = select_and_verify(ordered, by_id, Verifier(pubmed, crossref).verify_all, target,
                                          accept_partially_verified=accept_partially_verified)
        else:
            selection = SelectionOutcome(target=target, checked=[], verification=[], selected_ids=[], rounds=[],
                                         max_checked=0, accepted_statuses=("verified", "partially_verified")
                                         if accept_partially_verified else ("verified",))
        if selection.error is not None:
            run_log.add_error(ErrorEntry(database="all", stage="verification", query=query, timestamp=iso_utc(),
                                         error_type="unexpected_error", message=selection.error))
    finally:
        if own_client:
            http.close()

    checked_ids = {r.record_id for r in selection.checked}
    records = [r for r in unique if r.record_id in checked_ids]          # deduplication order
    vmap = {v.record_id: v for v in selection.verification}
    verification = [vmap[r.record_id] for r in records if r.record_id in vmap]

    finished = now()
    params = {"max_results_per_source": max_results, "from_year": from_year, "to_year": to_year,
              "candidate_pool_per_query": pool, "records_requested_per_query": per_query,
              "max_selected": target, "abstract_enrichment_limit": limit, "source_policy": policy}
    summary = build_summary(query, params, started, finished, outcomes, retrieved, unique, merged, verification,
                            run_log)
    expansion_log = _expansion_log(plan, query_stats)
    dedup_log = _dedup_log(retrieved, unique, merged)
    selection_log = _selection_log(ranked, selection)
    enrichment_log = enrichment.to_json()
    policy_log = _policy_log(policy_outcome, classes, selection, target)
    for cand in selection_log["candidates"]:
        c = classes.get(cand["record_id"])
        cand["source_status"] = c.source_status if c else None
    summary["query_generation"] = _query_generation(plan)
    summary["queries_used"] = {"pubmed": [q.pubmed for q in plan.queries],
                               "crossref": [q.crossref for q in plan.queries]}
    summary["query_expansion"] = expansion_log
    summary["candidates_per_query"] = {db: [{"query_id": q.query_id, "retrieved": st.get("retrieved", 0),
                                             "new_candidates": st.get("contributed", 0)}
                                            for q, st in zip(plan.queries, query_stats[db])]
                                       for db in DATABASES}
    summary["deduplication"] = {k: dedup_log[k] for k in ("candidates_before", "unique_after", "duplicates_merged")}
    summary["selection"] = {**{k: v for k, v in selection_log.items() if k != "candidates"},
                            "selected_scores": [c for c in selection_log["candidates"]
                                                if c["record_id"] in selection.selected_ids]}
    summary["abstract_enrichment"] = enrichment.counts()
    summary["source_policy"] = {k: v for k, v in policy_log.items() if k not in ("classification", "tier_by_id",
                                                                                "order_after_policy")}
    summary["source_classification"] = {r.record_id: classes[r.record_id].source_status for r in records
                                        if r.record_id in classes}
    summary["verified_records"] = len(records)
    summary["run_directory"] = run_dir.name
    write_outputs(run_dir, query, params, summary, records, verification, run_log, settings.secret_values(),
                  query_expansion=expansion_log, deduplication=dedup_log, selection=selection_log,
                  abstract_enrichment=enrichment_log, source_policy=policy_log,
                  source_status={rid: c.to_json() for rid, c in classes.items()})
    return InvestigationResult(run_dir=run_dir, summary=summary, records=records, verification=verification)


def _inline_abstracts(unique: list[Record], raw: dict[str, str]) -> dict[str, str]:
    """record_id -> plain-text Crossref abstract from the search results (any merged provenance id)."""
    from sciforge.sourcetext import jats_to_text

    out: dict[str, str] = {}
    for rec in unique:
        for rid in [rec.record_id, *(p.source_record_id for p in rec.provenance)]:
            text = jats_to_text(raw.get(rid))
            if text:
                out[rec.record_id] = text
                break
    return out


def _classify(unique: list[Record], *metadata_maps: dict[str, dict[str, Any]]) -> dict[str, SourceClassification]:
    """Source-type classification of every unique record from the metadata of all its merged provenance ids."""
    out: dict[str, SourceClassification] = {}
    for rec in unique:
        ids = list(dict.fromkeys([rec.record_id, *(p.source_record_id for p in rec.provenance)]))
        meta = [m[rid] for rid in ids for m in metadata_maps if rid in m]
        out[rec.record_id] = classify_record(rec.record_id, meta, rec.doi)
    return out


def _policy_log(outcome: PolicyOutcome, classes: dict[str, SourceClassification], selection: SelectionOutcome,
                target: int) -> dict[str, Any]:
    selected = [{"record_id": rid, "source_status": classes[rid].source_status if rid in classes else "unknown"}
                for rid in selection.selected_ids]
    counts: dict[str, int] = {}
    for item in selected:
        counts[item["source_status"]] = counts.get(item["source_status"], 0) + 1
    shortfall = max(0, target - len(selection.selected_ids))
    note = None
    if shortfall:
        note = (f"Only {len(selection.selected_ids)} of {target} requested sources were selected"
                + (" after excluding every source that is not a peer-reviewed journal article (metadata-based) "
                   "under SCIFORGE_SOURCE_POLICY=peer_reviewed_only" if outcome.policy == POLICY_ONLY else "")
                + ".")
    return {**outcome.to_json(), "status_note": STATUS_NOTE,
            "selected_sources": selected, "selected_status_counts": dict(sorted(counts.items())),
            "preprints_selected": counts.get("preprint", 0),
            "requested": target, "selected": len(selection.selected_ids), "shortfall": shortfall,
            "fewer_than_requested": bool(shortfall), "shortfall_note": note,
            "classification": {rid: c.to_json() for rid, c in classes.items()}}


def _query_generation(plan: QueryPlan) -> str:
    if not plan.enabled:
        return ("none: query expansion disabled; the research question was searched verbatim on PubMed and "
                "Crossref (one query per database)")
    if len(plan.queries) > 1:
        return (f"deterministic rule-based expansion (no model): the question verbatim plus "
                f"{len(plan.queries) - 1} focused queries, each run on PubMed ([tiab] terms) and Crossref "
                f"(keywords); see query_expansion / queries_used")
    return ("deterministic rule-based expansion enabled (no model), but no curated concept matched: only the "
            "question verbatim was searched on PubMed and Crossref")


def _query_origins(plan: QueryPlan, per_query: dict[str, list[SearchOutcome]]
                   ) -> dict[str, list[tuple[str, str, int]]]:
    """record_id -> [(query_id, database, 1-based rank)] from the raw per-query results."""
    origins: dict[str, list[tuple[str, str, int]]] = {}
    for database, outs in per_query.items():
        for pq, outcome in zip(plan.queries, outs):
            for rank, rec in enumerate(outcome.records, start=1):
                origins.setdefault(rec.record_id, []).append((pq.query_id, database, rank))
    return origins


def _dedup_log(retrieved: list[Record], unique: list[Record], merged: int) -> dict[str, Any]:
    """Before/after counts and every merge (unique record <- merged record ids, matched rule)."""
    merges = [{"record_id": r.record_id,
               "merged_record_ids": [p.source_record_id for p in r.provenance[1:]],
               "matched_on": [p.matched_on for p in r.provenance[1:]],
               "source_databases": [p.source_database for p in r.provenance]}
              for r in unique if len(r.provenance) > 1]
    return {"method": "unchanged v0.2 dedup (DOI, PMID, or title+year+first author)",
            "candidates_before": len(retrieved), "unique_after": len(unique), "duplicates_merged": merged,
            "merges": merges}


def _selection_log(ranked: list[RankedCandidate], selection: SelectionOutcome) -> dict[str, Any]:
    return {"method": SELECTION_METHOD, "candidates_scored": len(ranked),
            "backfill_factor": VERIFY_BACKFILL_FACTOR, **selection.to_json(),
            "candidates": [c.to_json() for c in ranked]}


def _expansion_log(plan: QueryPlan, stats: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Query plan + per-query, per-database result counts (for search_log.json / summary.json)."""
    payload = plan.to_json()
    for i, q in enumerate(payload["queries"]):
        q["results"] = {db: stats[db][i] for db in stats if i < len(stats[db])}
    return payload


def build_summary(
    query: str,
    params: dict[str, Any],
    started: datetime,
    finished: datetime,
    outcomes: list[SearchOutcome],
    retrieved: list[Record],
    unique: list[Record],
    merged: int,
    verification: list[VerificationResult],
    run_log: RunLog,
) -> dict[str, Any]:
    """Build the ``summary.json`` payload."""
    status_counts = Counter(v.status for v in verification)
    return {
        "sciforge_version": __version__,
        "question": query,
        "query_used": query,
        "query_generation": "not recorded",
        "parameters": params,
        "started_at": iso_utc(started),
        "finished_at": iso_utc(finished),
        "databases_queried": list(DATABASES),
        "search_status": {o.database: o.status for o in outcomes},
        "total_hits_reported": {o.database: o.total_hits for o in outcomes},
        "retrieved_per_source": {o.database: len(o.records) for o in outcomes},
        "total_retrieved": len(retrieved),
        "unique_records": len(unique),
        "duplicates_merged": merged,
        "verification_counts": {s: status_counts.get(s, 0) for s in ("verified", "partially_verified", "not_verified")},
        "errors_count": len(run_log.errors),
        "failed_databases": [o.database for o in outcomes if o.status == "failed"],
        "partially_failed_databases": [o.database for o in outcomes if o.status == "partial"],
        "databases_with_errors": sorted({e.database for e in run_log.errors}),
    }


def _source_json(record: Record, source_status: dict[str, dict[str, Any]] | None) -> dict[str, Any]:
    data = record.model_dump()
    if source_status is not None:
        c = source_status.get(record.record_id) or {}
        data["source_status"] = c.get("source_status", "unknown")
        data["source_status_basis"] = c.get("basis", [])
    return data


def write_outputs(
    run_dir: Path,
    query: str,
    params: dict[str, Any],
    summary: dict[str, Any],
    records: list[Record],
    verification: list[VerificationResult],
    run_log: RunLog,
    secrets: list[str],
    query_expansion: dict[str, Any] | None = None,
    deduplication: dict[str, Any] | None = None,
    selection: dict[str, Any] | None = None,
    abstract_enrichment: dict[str, Any] | None = None,
    source_policy: dict[str, Any] | None = None,
    source_status: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Write search_log.json, sources.json, verification.json, summary.json."""
    _write_json(run_dir / "search_log.json", {
        "sciforge_version": __version__,
        "question": query,
        "parameters": params,
        "note": "Sensitive parameters (api_key, email, mailto) are redacted.",
        **({"query_expansion": query_expansion} if query_expansion is not None else {}),
        **({"deduplication": deduplication} if deduplication is not None else {}),
        **({"abstract_enrichment": abstract_enrichment} if abstract_enrichment is not None else {}),
        **({"selection": selection} if selection is not None else {}),
        **({"source_policy": source_policy} if source_policy is not None else {}),
        "requests": [e.model_dump() for e in run_log.entries],
        "errors": [e.model_dump() for e in run_log.errors],
    }, secrets)
    _write_json(run_dir / "sources.json", {
        "sciforge_version": __version__,
        "question": query,
        "count": len(records),
        "records": [_source_json(r, source_status) for r in records],
    }, secrets)
    _write_json(run_dir / "verification.json", {
        "sciforge_version": __version__,
        "title_similarity_threshold": TITLE_SIMILARITY_THRESHOLD,
        "journal_similarity_threshold": JOURNAL_SIMILARITY_THRESHOLD,
        "results": [v.model_dump() for v in verification],
    }, secrets)
    _write_json(run_dir / "summary.json", summary, secrets)
