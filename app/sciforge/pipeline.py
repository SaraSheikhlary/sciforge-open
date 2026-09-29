"""Orchestrates one v0.2 investigation: expand queries -> search -> merge -> dedup -> verify -> write files.

Query plan (:mod:`sciforge.query_expansion`, deterministic, no model): the
research question verbatim (always the first query, exactly as before) plus,
unless disabled (``SCIFORGE_QUERY_EXPANSION=false`` / ``query_expansion=False``),
focused anchor×facet queries from a curated concept map. Every query is run on
PubMed and Crossref; per database the hits are merged (identical hits kept once,
at most ``max_results``), then the unchanged v0.2 dedup and verification run.
The plan and per-query result counts are written to ``search_log.json`` and
``summary.json`` (``query_expansion``).
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
from sciforge.config import Settings
from sciforge.crossref import CrossrefClient
from sciforge.dedup import deduplicate
from sciforge.http_utils import HttpFetcher, Sleep
from sciforge.logging_utils import RunLog, iso_utc, redact_text, run_stamp, utc_now
from sciforge.models import ErrorEntry, Record, SearchOutcome, VerificationResult
from sciforge.pubmed import PubMedClient
from sciforge.query_expansion import QueryPlan, expand_queries, merge_query_outcomes, single_query_plan
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
) -> InvestigationResult:
    """Run a full retrieve-and-verify investigation and write output files.

    Never raises for network or API problems; they are recorded in the logs.
    ``sleep`` defaults to :func:`time.sleep`, looked up at call time.
    ``query_expansion`` defaults to ``settings.query_expansion`` (env
    ``SCIFORGE_QUERY_EXPANSION``, default true). ``max_results`` caps the merged
    records per database (each query also requests at most ``max_results``).
    """
    query = question.strip()
    if not query:
        raise ValueError("research question must not be empty")
    settings = settings or Settings.from_env()
    expand = settings.query_expansion if query_expansion is None else query_expansion
    plan: QueryPlan = expand_queries(query) if expand else single_query_plan(query)
    started = now()
    run_dir = make_run_dir(Path(output_dir), started)
    run_log = RunLog(settings.secret_values())

    own_client = client is None
    http = client or httpx.Client(follow_redirects=True, timeout=settings.timeout_seconds)
    try:
        fetcher = HttpFetcher(http, settings, run_log, sleep=sleep or time.sleep)
        pubmed = PubMedClient(fetcher, settings)
        crossref = CrossrefClient(fetcher, settings)

        per_query: dict[str, list[SearchOutcome]] = {"pubmed": [], "crossref": []}
        for pq in plan.queries:
            per_query["pubmed"].append(_safe_search(
                lambda: pubmed.search(pq.pubmed, max_results, from_year, to_year), "pubmed", pq.pubmed, run_log))
            per_query["crossref"].append(_safe_search(
                lambda: crossref.search(pq.crossref, max_results, from_year, to_year), "crossref", pq.crossref,
                run_log))
        outcomes = []
        query_stats: dict[str, list[dict[str, Any]]] = {}
        for database in DATABASES:
            merged_outcome, stats = merge_query_outcomes(database, per_query[database], max_results)
            outcomes.append(merged_outcome)
            query_stats[database] = stats
        retrieved = [r for o in outcomes for r in o.records]
        unique, merged = deduplicate(retrieved)
        try:
            verification = Verifier(pubmed, crossref).verify_all(unique)
        except Exception as exc:  # noqa: BLE001 - defensive; verify_all should not raise
            run_log.add_error(ErrorEntry(database="all", stage="verification", query=query, timestamp=iso_utc(),
                                         error_type="unexpected_error", message=f"{type(exc).__name__}: {exc}"))
            verification = []
    finally:
        if own_client:
            http.close()

    finished = now()
    params = {"max_results_per_source": max_results, "from_year": from_year, "to_year": to_year}
    summary = build_summary(query, params, started, finished, outcomes, retrieved, unique, merged, verification, run_log)
    expansion_log = _expansion_log(plan, query_stats)
    if plan.enabled and len(plan.queries) > 1:
        summary["query_generation"] = (f"deterministic rule-based expansion (no model): the question verbatim plus "
                                       f"{len(plan.queries) - 1} focused queries; see query_expansion")
    summary["query_expansion"] = expansion_log
    summary["run_directory"] = run_dir.name
    write_outputs(run_dir, query, params, summary, unique, verification, run_log, settings.secret_values(),
                  query_expansion=expansion_log)
    return InvestigationResult(run_dir=run_dir, summary=summary, records=unique, verification=verification)


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
        "query_generation": "none: v0.2 uses the research question verbatim as the search query",
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
) -> None:
    """Write search_log.json, sources.json, verification.json, summary.json."""
    _write_json(run_dir / "search_log.json", {
        "sciforge_version": __version__,
        "question": query,
        "parameters": params,
        "note": "Sensitive parameters (api_key, email, mailto) are redacted.",
        **({"query_expansion": query_expansion} if query_expansion is not None else {}),
        "requests": [e.model_dump() for e in run_log.entries],
        "errors": [e.model_dump() for e in run_log.errors],
    }, secrets)
    _write_json(run_dir / "sources.json", {
        "sciforge_version": __version__,
        "question": query,
        "count": len(records),
        "records": [r.model_dump() for r in records],
    }, secrets)
    _write_json(run_dir / "verification.json", {
        "sciforge_version": __version__,
        "title_similarity_threshold": TITLE_SIMILARITY_THRESHOLD,
        "journal_similarity_threshold": JOURNAL_SIMILARITY_THRESHOLD,
        "results": [v.model_dump() for v in verification],
    }, secrets)
    _write_json(run_dir / "summary.json", summary, secrets)
