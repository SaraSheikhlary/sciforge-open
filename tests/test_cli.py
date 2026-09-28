"""CLI argument handling and output."""

import httpx
import pytest

from conftest import mock_client
from sciforge.cli import EXIT_ALL_SOURCES_FAILED, build_parser, main


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["investigate", "--help"])
    assert exc.value.code == 0
    assert "--max-results" in capsys.readouterr().out


def test_defaults():
    args = build_parser().parse_args(["investigate", "q"])
    assert args.max_results == 20 and args.output_dir == "runs" and args.from_year is None


@pytest.mark.parametrize("argv", [
    ["investigate", "q", "--max-results", "0"],
    ["investigate", "q", "--max-results", "abc"],
    ["investigate", "q", "--from-year", "20x"],
    ["investigate", "q", "--from-year", "2020", "--to-year", "2010"],
    ["investigate", "   "],
])
def test_invalid_args(argv):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2


def test_no_command_prints_help(capsys):
    assert main([]) == 2


def test_bad_env_is_reported(monkeypatch, capsys):
    monkeypatch.setenv("SCIFORGE_TIMEOUT_SECONDS", "nope")
    assert main(["investigate", "q"]) == 2
    assert "SCIFORGE_TIMEOUT_SECONDS" in capsys.readouterr().err


def test_run_with_all_sources_down(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("SCIFORGE_MAX_RETRIES", "0")

    def handler(request):
        return httpx.Response(503)
    code = main(["investigate", "q", "--output-dir", str(tmp_path)], client=mock_client(handler))
    assert code == EXIT_ALL_SOURCES_FAILED
    out = capsys.readouterr().out
    assert "Failed databases: pubmed, crossref" in out and "no conclusions" in out
    assert len(list(tmp_path.iterdir())) == 1
