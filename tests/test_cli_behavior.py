"""CLI behaviour: exit codes, argument errors, output files, partial and total failure."""

import json

import httpx
import pytest

from conftest import FAKE_API_KEY, api_handler, mock_client
from sciforge.cli import EXIT_ALL_SOURCES_FAILED, EXIT_OK, EXIT_USAGE, main

OUTPUT_FILES = ["search_log.json", "sources.json", "summary.json", "verification.json"]


def only_run_dir(tmp_path):
    dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(dirs) == 1
    return dirs[0]


def load(run_dir, name):
    return json.loads((run_dir / name).read_text())


# ------------------------------------------------------------ malformed input -> exit 2

@pytest.mark.parametrize("extra", [
    ["--from-year", "abc"],
    ["--to-year", "20x4"],
    ["--from-year", "2020.5"],
    ["--from-year", "1799"],
    ["--to-year", "2101"],
    ["--from-year=-5"],
    ["--from-year", "2024", "--to-year", "2020"],
    ["--max-results", "0"],
    ["--max-results", "1001"],
    ["--max-results", "ten"],
])
def test_malformed_input_exits_2_without_traceback(tmp_path, capsys, extra):
    with pytest.raises(SystemExit) as exc:
        main(["investigate", "q", "--output-dir", str(tmp_path), *extra])
    assert exc.value.code == EXIT_USAGE
    err = capsys.readouterr().err
    assert "Traceback" not in err and "error:" in err
    assert list(tmp_path.iterdir()) == []  # nothing written


def test_empty_question_exits_2(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["investigate", "  ", "--output-dir", str(tmp_path)])
    assert exc.value.code == EXIT_USAGE
    assert "Traceback" not in capsys.readouterr().err


def test_invalid_env_config_exits_2(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("SCIFORGE_MAX_RETRIES", "lots")
    assert main(["investigate", "q", "--output-dir", str(tmp_path)]) == EXIT_USAGE
    assert "Traceback" not in capsys.readouterr().err


def test_missing_command_exits_2():
    assert main([]) == EXIT_USAGE


# ------------------------------------------------------------ success -> exit 0

def test_exit_0_successful_run_writes_sane_outputs(tmp_path, capsys, fast_sleep, monkeypatch):
    monkeypatch.setenv("NCBI_API_KEY", FAKE_API_KEY)
    code = main(["investigate", "platelet shear", "--max-results", "5", "--output-dir", str(tmp_path)],
                client=mock_client(api_handler))
    assert code == EXIT_OK
    run_dir = only_run_dir(tmp_path)
    assert sorted(p.name for p in run_dir.iterdir()) == OUTPUT_FILES

    summary = load(run_dir, "summary.json")
    assert summary["question"] == "platelet shear" and summary["sciforge_version"] == "0.4.0"
    assert summary["retrieved_per_source"] == {"pubmed": 2, "crossref": 2}
    assert summary["unique_records"] == 3 and summary["duplicates_merged"] == 1
    assert summary["verification_counts"] == {"verified": 3, "partially_verified": 0, "not_verified": 0}
    assert summary["errors_count"] == 0 and summary["failed_databases"] == []

    sources = load(run_dir, "sources.json")
    assert sources["count"] == 3 == len(sources["records"])
    for r in sources["records"]:
        assert r["record_id"].startswith("rec_") and r["source_database"] in {"pubmed", "crossref"}
        assert r["doi"] or r["pmid"]

    ver = load(run_dir, "verification.json")
    assert {v["record_id"] for v in ver["results"]} == {r["record_id"] for r in sources["records"]}
    assert ver["title_similarity_threshold"] == 0.9

    log = load(run_dir, "search_log.json")
    assert log["errors"] == [] and all(e["status"] == "ok" for e in log["requests"])
    assert all(e["params"].get("api_key") in (None, "[REDACTED]") for e in log["requests"])
    for name in OUTPUT_FILES:
        assert FAKE_API_KEY not in (run_dir / name).read_text()

    out = capsys.readouterr().out
    assert "Unique records: 3" in out and str(run_dir) in out


def test_python_dash_m_entry_uses_same_main():
    import sciforge.__main__ as entry
    from sciforge import cli

    assert entry.main is cli.main


# ------------------------------------------------------------ partial failure -> exit 0

def _pubmed_down(kind):
    def handler(request):
        if "eutils" in request.url.host:
            if kind == "timeout":
                raise httpx.ReadTimeout("timed out", request=request)
            return httpx.Response(503)
        return api_handler(request)
    return handler


def _crossref_search_down(kind):
    def handler(request):
        if request.url.path == "/works":
            if kind == "timeout":
                raise httpx.ReadTimeout("timed out", request=request)
            return httpx.Response(503)
        return api_handler(request)
    return handler


@pytest.mark.parametrize("kind, expected_type, expected_status", [
    ("5xx", "http_error", 503), ("timeout", "timeout", None)])
def test_partial_failure_pubmed_down_exit_0(tmp_path, fast_sleep, kind, expected_type, expected_status):
    code = main(["investigate", "q", "--output-dir", str(tmp_path)], client=mock_client(_pubmed_down(kind)))
    assert code == EXIT_OK
    run_dir = only_run_dir(tmp_path)
    summary = load(run_dir, "summary.json")
    assert summary["failed_databases"] == ["pubmed"] and summary["errors_count"] >= 1
    errors = load(run_dir, "search_log.json")["errors"]
    search_err = [e for e in errors if e["database"] == "pubmed" and e["stage"] == "search"][0]
    assert search_err["error_type"] == expected_type and search_err["http_status"] == expected_status
    assert search_err["query"] == "q" and search_err["message"]
    records = load(run_dir, "sources.json")["records"]
    assert len(records) == 2 and {r["source_database"] for r in records} == {"crossref"}


@pytest.mark.parametrize("kind", ["5xx", "timeout"])
def test_partial_failure_crossref_down_exit_0(tmp_path, fast_sleep, kind):
    code = main(["investigate", "q", "--output-dir", str(tmp_path)], client=mock_client(_crossref_search_down(kind)))
    assert code == EXIT_OK
    run_dir = only_run_dir(tmp_path)
    assert load(run_dir, "summary.json")["failed_databases"] == ["crossref"]
    records = load(run_dir, "sources.json")["records"]
    assert len(records) == 2 and {r["source_database"] for r in records} == {"pubmed"}
    assert any(e["database"] == "crossref" for e in load(run_dir, "search_log.json")["errors"])


# ------------------------------------------------------------ complete failure -> exit 3

@pytest.mark.parametrize("failure", ["5xx", "connection"])
def test_complete_failure_exit_3_outputs_still_written(tmp_path, capsys, fast_sleep, failure):
    def handler(request):
        if failure == "connection":
            raise httpx.ConnectError("unreachable", request=request)
        return httpx.Response(500)
    code = main(["investigate", "q", "--output-dir", str(tmp_path)], client=mock_client(handler))
    assert code == EXIT_ALL_SOURCES_FAILED
    run_dir = only_run_dir(tmp_path)
    assert sorted(p.name for p in run_dir.iterdir()) == OUTPUT_FILES
    summary = load(run_dir, "summary.json")
    assert summary["failed_databases"] == ["pubmed", "crossref"]
    assert summary["unique_records"] == 0 and summary["errors_count"] == 2
    assert load(run_dir, "sources.json")["records"] == []
    assert load(run_dir, "verification.json")["results"] == []
    assert len(load(run_dir, "search_log.json")["errors"]) == 2
    captured = capsys.readouterr()
    assert "Failed databases: pubmed, crossref" in captured.out and "Traceback" not in captured.err
