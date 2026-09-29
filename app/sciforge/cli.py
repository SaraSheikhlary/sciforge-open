"""Command-line interface: ``sciforge investigate "question"``."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from typing import Any

import httpx

from sciforge import __version__
from sciforge.config import ConfigError, Settings
from sciforge.logging_utils import RedactingFilter, get_logger

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_ALL_SOURCES_FAILED = 3
MAX_RESULTS_LIMIT = 1000


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if not 1 <= value <= MAX_RESULTS_LIMIT:
        raise argparse.ArgumentTypeError(f"must be between 1 and {MAX_RESULTS_LIMIT}")
    return value


def _year(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a four-digit year") from exc
    if not 1800 <= value <= 2100:
        raise argparse.ArgumentTypeError("must be between 1800 and 2100")
    return value


def build_parser() -> argparse.ArgumentParser:
    """Argument parser for the ``sciforge`` command."""
    parser = argparse.ArgumentParser(
        prog="sciforge",
        description="SciForge v0.2: deterministic literature retrieval (PubMed, Crossref) and citation verification.",
    )
    parser.add_argument("--version", action="version", version=f"sciforge {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    inv = sub.add_parser(
        "investigate",
        help="search PubMed and Crossref for a question, deduplicate, and verify identifiers",
        description=(
            "Search PubMed and Crossref with the research question verbatim plus deterministic rule-based "
            "focused queries (no model), collect a candidate pool per query and database "
            "(SCIFORGE_CANDIDATE_POOL_PER_QUERY, default 10), deduplicate, rank the candidates "
            "deterministically for relevance and concept diversity, verify each selected DOI/PMID (with "
            "backfill if verification fails), and write JSON outputs. "
            "No conclusions or evidence extraction are produced in v0.2."
        ),
    )
    inv.add_argument("question", help="research question (always searched verbatim, plus rule-based focused queries)")
    inv.add_argument("--max-results", type=_positive_int, default=20,
                     help="per-database result size (default: 20): each query requests max(N, candidate pool) "
                          "records per database and up to 2*N selected records are verified")
    inv.add_argument("--from-year", type=_year, default=None, help="earliest publication year (inclusive)")
    inv.add_argument("--to-year", type=_year, default=None, help="latest publication year (inclusive)")
    inv.add_argument("--output-dir", default="runs", help="directory for run outputs (default: runs/)")
    inv.add_argument("--no-query-expansion", action="store_true",
                     help="search only the question verbatim (disable deterministic query expansion; "
                          "same as SCIFORGE_QUERY_EXPANSION=false)")
    inv.add_argument("-v", "--verbose", action="store_true", help="log requests and errors to stderr")
    return parser


def format_summary(summary: dict[str, Any], run_dir: str) -> str:
    """Short human-readable run summary."""
    v = summary["verification_counts"]
    qe = summary.get("query_expansion") or {}
    n_queries = len(qe.get("queries") or []) or 1
    lines = [
        f"SciForge {summary['sciforge_version']} — retrieval and verification only (no conclusions).",
        f"Question: {summary['question']}",
        f"Queries: {n_queries} per database (query expansion {'on' if qe.get('enabled') else 'off'}; "
        "see search_log.json)",
        "Candidates retrieved: " + ", ".join(f"{db} {n}" for db, n in summary["retrieved_per_source"].items())
        + f" (total {summary['total_retrieved']})",
        f"Unique records: {summary['unique_records']} (duplicates merged: {summary['duplicates_merged']})",
        f"Verification: verified {v['verified']}, partially_verified {v['partially_verified']}, "
        f"not_verified {v['not_verified']}",
        f"Errors recorded: {summary['errors_count']}",
    ]
    sel = summary.get("selection")
    if sel:
        lines.insert(5, f"Selected: {len(sel.get('selected_record_ids') or [])} of target {sel.get('target')} "
                        f"({sel.get('candidates_verified', 0)} candidates verified)")
    enr = summary.get("abstract_enrichment")
    if enr:
        lines.insert(len(lines) - 2, f"Abstract enrichment for ranking: {enr.get('with_abstract', 0)} of "
                                     f"{enr.get('considered', 0)} candidates had an abstract "
                                     f"(limit {enr.get('limit', 0)}; the rest were ranked on title only)")
    pol = summary.get("source_policy")
    if pol:
        lines.insert(len(lines) - 2, f"Source policy: {pol.get('policy')}; preprints selected: "
                                     f"{pol.get('preprints_selected', 0)}; excluded by policy: "
                                     f"{pol.get('excluded_count', 0)}")
        if pol.get("fewer_than_requested") and pol.get("shortfall_note"):
            lines.insert(len(lines) - 2, f"Note: {pol['shortfall_note']}")
    if summary["failed_databases"]:
        lines.append("Failed databases: " + ", ".join(summary["failed_databases"]))
    if summary["partially_failed_databases"]:
        lines.append("Partially failed databases: " + ", ".join(summary["partially_failed_databases"]))
    lines.append(f"Outputs: {run_dir}")
    return "\n".join(lines)


def _configure_logging(verbose: bool, settings: Settings) -> None:
    logger = get_logger()
    if not verbose or logger.handlers:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.addFilter(RedactingFilter(settings.secret_values()))
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def main(argv: Sequence[str] | None = None, *, client: httpx.Client | None = None) -> int:
    """Entry point. ``client`` may be injected (e.g. a MockTransport client in tests)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command != "investigate":
        parser.print_help()
        return EXIT_USAGE
    if not args.question.strip():
        parser.error("question must not be empty")
    if args.from_year is not None and args.to_year is not None and args.from_year > args.to_year:
        parser.error("--from-year must not be later than --to-year")
    try:
        settings = Settings.from_env()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    _configure_logging(args.verbose, settings)

    from sciforge.pipeline import run_investigation  # local import keeps --help fast

    result = run_investigation(
        args.question,
        max_results=args.max_results,
        from_year=args.from_year,
        to_year=args.to_year,
        output_dir=args.output_dir,
        settings=settings,
        client=client,
        query_expansion=False if args.no_query_expansion else None,
    )
    print(format_summary(result.summary, str(result.run_dir)))
    if len(result.summary["failed_databases"]) == len(result.summary["databases_queried"]):
        return EXIT_ALL_SOURCES_FAILED
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
