"""End-to-end runs against mocked PubMed and Crossref."""

import json
from datetime import datetime, timezone

import httpx

from conftest import (
    FAKE_API_KEY,
    FAKE_EMAIL,
    efetch_empty_response,
    crossref_list_payload,
    crossref_work,
    crossref_work_payload,
    esearch_payload,
    esummary_doc,
    esummary_payload,
    json_response,
    mock_client,
)
from sciforge.pipeline import run_investigation
from sciforge.query_expansion import expand_queries

# "platelet shear activation" expands to the verbatim question + 2 focused queries (run on both databases)
N_QUERIES = len(expand_queries("platelet shear activation").queries)

FIXED = datetime(2026, 9, 28, 23, 41, 0, tzinfo=timezone.utc)


def happy_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path.endswith("esearch.fcgi"):
        return json_response(esearch_payload(["111", "222"]))
    if path.endswith("esummary.fcgi"):
        return json_response(esummary_payload([
            esummary_doc("111", doi="10.1000/abc123"),
            esummary_doc("222", doi=None, title="Another platelet paper", authors=["Doe J"]),
        ]))
    if path == "/works":
        return json_response(crossref_list_payload([
            crossref_work("10.1000/ABC123"),  # duplicate of PMID 111 by DOI
            crossref_work("10.3000/other", title="Unrelated coral study", authors=[("Reef", "A")]),
        ]))
    if path == "/works/10.1000/abc123":
        return json_response(crossref_work_payload(crossref_work("10.1000/ABC123")))
    if path == "/works/10.3000/other":
        return httpx.Response(404)
    if path.endswith("efetch.fcgi"):
        return efetch_empty_response()
    return httpx.Response(500)


def run(tmp_path, handler, settings, **kw):
    return run_investigation("  platelet shear activation ", output_dir=tmp_path / "runs", settings=settings,
                             client=mock_client(handler), sleep=lambda s: None, now=lambda: FIXED, **kw)


def test_happy_path_outputs(tmp_path, settings):
    result = run(tmp_path, happy_handler, settings, max_results=5)
    assert result.run_dir == tmp_path / "runs" / "20260928T234100Z"
    files = sorted(p.name for p in result.run_dir.iterdir())
    assert files == ["search_log.json", "sources.json", "summary.json", "verification.json"]

    summary = json.loads((result.run_dir / "summary.json").read_text())
    assert summary["question"] == "platelet shear activation"
    assert summary["sciforge_version"] == "0.4.0"
    assert summary["retrieved_per_source"] == {"pubmed": 2, "crossref": 2}
    assert summary["unique_records"] == 3 and summary["duplicates_merged"] == 1
    assert summary["verification_counts"] == {"verified": 2, "partially_verified": 0, "not_verified": 1}
    assert summary["failed_databases"] == [] and summary["errors_count"] == 0
    assert summary["started_at"] == "2026-09-28T23:41:00Z"

    sources = json.loads((result.run_dir / "sources.json").read_text())
    merged = sources["records"][0]
    assert merged["doi"] == "10.1000/abc123" and merged["pmid"] == "111"
    assert [p["source_database"] for p in merged["provenance"]] == ["pubmed", "crossref"]

    ver = {v["record_id"]: v for v in json.loads((result.run_dir / "verification.json").read_text())["results"]}
    assert ver[merged["record_id"]]["status"] == "verified"
    assert len(ver[merged["record_id"]]["checks"]) == 2
    by_title = {r["title"]: ver[r["record_id"]] for r in sources["records"]}
    assert by_title["Another platelet paper"]["status"] == "verified"  # PMID-only record resolved via esummary
    coral = by_title["Unrelated coral study"]
    assert coral["status"] == "not_verified" and coral["reasons"][0] == "not_found"
    assert coral["checks"][0]["http_status"] == 404

    log = json.loads((result.run_dir / "search_log.json").read_text())
    stages = [e["stage"] for e in log["requests"]]
    assert N_QUERIES == 3
    assert stages.count("search") == 2 * N_QUERIES and "summary" in stages and "verification" in stages


def test_secrets_never_written(tmp_path, settings):
    result = run(tmp_path, happy_handler, settings)
    for path in result.run_dir.iterdir():
        text = path.read_text()
        assert FAKE_API_KEY not in text, path.name
        assert FAKE_EMAIL not in text, path.name


def test_all_sources_down_does_not_crash(tmp_path, settings):
    def handler(request):
        raise httpx.ConnectError("network unreachable", request=request)
    result = run(tmp_path, handler, settings)
    s = result.summary
    assert s["failed_databases"] == ["pubmed", "crossref"]
    assert s["unique_records"] == 0 and s["errors_count"] == 2 * N_QUERIES
    log = json.loads((result.run_dir / "search_log.json").read_text())
    assert {e["error_type"] for e in log["errors"]} == {"connection_error"}
    for err in log["errors"]:
        assert set(err) >= {"database", "query", "timestamp", "error_type", "http_status", "message"}


def test_one_source_down_other_continues(tmp_path, settings):
    def handler(request):
        if "eutils" in request.url.host:
            return httpx.Response(503)
        return happy_handler(request)
    result = run(tmp_path, handler, settings)
    s = result.summary
    assert s["failed_databases"] == ["pubmed"]
    assert s["retrieved_per_source"]["crossref"] == 2
    assert s["verification_counts"]["verified"] == 1  # Crossref DOI lookup still works


def test_unexpected_exception_in_client_is_contained(tmp_path, settings, monkeypatch):
    from sciforge import pubmed

    def boom(*a, **k):
        raise RuntimeError("bug")
    monkeypatch.setattr(pubmed.PubMedClient, "search", boom)
    result = run(tmp_path, happy_handler, settings)
    assert result.summary["failed_databases"] == ["pubmed"]
    assert result.summary["errors_count"] >= 1


def test_run_dir_collision(tmp_path, settings):
    a = run(tmp_path, happy_handler, settings)
    b = run(tmp_path, happy_handler, settings)
    assert a.run_dir != b.run_dir and b.run_dir.name == "20260928T234100Z-1"
