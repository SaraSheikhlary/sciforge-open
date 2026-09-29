"""Publication date range filtering, from client parameters through pipeline and CLI."""

import pytest

from conftest import api_handler, mock_client
from sciforge.cli import main
from sciforge.pipeline import run_investigation


class Recorder:
    """Wraps a handler and keeps every request."""

    def __init__(self, handler=api_handler):
        self.handler = handler
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        return self.handler(request)

    def params_for(self, fragment):
        return [r.url.params for r in self.requests if fragment in r.url.path]


def test_pubmed_date_params_both_bounds(make_harness):
    rec = Recorder()
    h = make_harness(rec)
    h.pubmed.search("q", 3, from_year=2020, to_year=2024)
    p = rec.params_for("esearch")[0]
    assert (p["datetype"], p["mindate"], p["maxdate"]) == ("pdat", "2020", "2024")


def test_pubmed_date_params_only_to_year(make_harness):
    rec = Recorder()
    make_harness(rec).pubmed.search("q", 3, to_year=2010)
    p = rec.params_for("esearch")[0]
    assert (p["datetype"], p["mindate"], p["maxdate"]) == ("pdat", "1800", "2010")


def test_pubmed_no_date_params_without_years(make_harness):
    rec = Recorder()
    make_harness(rec).pubmed.search("q", 3)
    p = rec.params_for("esearch")[0]
    assert not {"datetype", "mindate", "maxdate"} & set(p.keys())


@pytest.mark.parametrize("from_year, to_year, expected", [
    (2020, 2024, "from-pub-date:2020,until-pub-date:2024"),
    (2020, None, "from-pub-date:2020"),
    (None, 2024, "until-pub-date:2024"),
])
def test_crossref_date_filter(make_harness, from_year, to_year, expected):
    rec = Recorder()
    make_harness(rec).crossref.search("q", 3, from_year=from_year, to_year=to_year)
    assert rec.params_for("/works")[0]["filter"] == expected


def test_crossref_no_filter_without_years(make_harness):
    rec = Recorder()
    make_harness(rec).crossref.search("q", 3)
    assert "filter" not in rec.params_for("/works")[0]


def test_date_filters_passed_through_pipeline(tmp_path, settings):
    rec = Recorder()
    result = run_investigation("q", from_year=2020, to_year=2024, output_dir=tmp_path, settings=settings,
                               client=mock_client(rec), sleep=lambda s: None)
    es = rec.params_for("esearch.fcgi")[0]
    assert (es["mindate"], es["maxdate"], es["datetype"]) == ("2020", "2024", "pdat")
    cr = [r.url.params for r in rec.requests if r.url.path == "/works"][0]
    assert cr["filter"] == "from-pub-date:2020,until-pub-date:2024"
    assert result.summary["parameters"] == {"max_results_per_source": 20, "from_year": 2020, "to_year": 2024,
                                            "candidate_pool_per_query": 10, "records_requested_per_query": 20,
                                            "max_selected": 40}


def test_date_filters_passed_through_cli(tmp_path, fast_sleep):
    rec = Recorder()
    code = main(["investigate", "q", "--from-year", "2020", "--to-year", "2024", "--max-results", "3",
                 "--output-dir", str(tmp_path)], client=mock_client(rec))
    assert code == 0
    es = rec.params_for("esearch.fcgi")[0]
    # each query requests max(candidate pool (default 10), --max-results) records before dedup/selection
    assert (es["mindate"], es["maxdate"], es["retmax"]) == ("2020", "2024", "10")
    cr = [r.url.params for r in rec.requests if r.url.path == "/works"][0]
    assert cr["filter"] == "from-pub-date:2020,until-pub-date:2024" and cr["rows"] == "10"


def test_same_from_and_to_year_allowed(tmp_path, fast_sleep):
    rec = Recorder()
    assert main(["investigate", "q", "--from-year", "2022", "--to-year", "2022", "--output-dir", str(tmp_path)],
                client=mock_client(rec)) == 0
    assert rec.params_for("esearch.fcgi")[0]["mindate"] == "2022"
