"""Orchestrates one v0.2 investigation: search -> dedup -> verify -> write files.

The research question is used verbatim as the search query (no model-based
query generation in v0.2).
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
) -> InvestigationResult:
    """Run a full retrieve-and-verify investigation and write output files.

    Never raises for network or API problems; they are recorded in the logs.
    ``sleep`` defaults to :func:`time.sleep`, looked up at call time.
    """
    query = question.strip()
    if not query:
        raise ValueError("research question must not be empty")
    settings = settings or Settings.from_env()
    started = now()
    run_dir = make_run_dir(Path(output_dir), started)
    run_log = RunLog(settings.secret_values())

    own_client = client is None
    http = client or httpx.Client(follow_redirects=True, timeout=settings.timeout_seconds)
    try:
        fetcher = HttpFetcher(http, settings, run_log, sleep=sleep or time.sleep)
        pubmed = PubMedClient(fetcher, settings)
        crossref = CrossrefClient(fetcher, settings)

        outcomes = [
            _safe_search(lambda: pubmed.search(query, max_results, from_year, to_year), "pubmed", query, run_log),
            _safe_search(lambda: crossref.search(query, max_results, from_year, to_year), "crossref", query, run_log),
        ]
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
    summary["run_directory"] = run_dir.name
    write_outputs(run_dir, query, params, summary, unique, verification, run_log, settings.secret_values())
    return InvestigationResult(run_dir=run_dir, summary=summary, records=unique, verification=verification)


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
) -> None:
    """Write search_log.json, sources.json, verification.json, summary.json."""
    _write_json(run_dir / "search_log.json", {
        "sciforge_version": __version__,
        "question": query,
        "parameters": params,
        "note": "Sensitive parameters (api_key, email, mailto) are redacted.",
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
