"""Deterministic query expansion before PubMed/Crossref retrieval (offline; MockTransport only)."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import httpx
import pytest

from conftest import (
    crossref_list_payload,
    crossref_work,
    crossref_work_payload,
    esearch_payload,
    esummary_doc,
    esummary_payload,
    json_response,
    mock_client,
)
from sciforge.cli import main
from sciforge.config import ConfigError, Settings
from sciforge.models import Record, SearchOutcome
from sciforge.pipeline import run_investigation
from sciforge.query_expansion import (
    CONCEPTS,
    MAX_FOCUSED_QUERIES,
    expand_queries,
    keywords,
    merge_query_outcomes,
    single_query_plan,
)

FIXED = datetime(2026, 9, 28, 23, 41, 0, tzinfo=timezone.utc)
PLATELET_Q = "What is the role of lipid-related changes in shear-mediated platelet activation?"
EXPECTED_CONCEPTS = ["platelet activation", "shear stress", "platelet membrane", "membrane lipids", "phospholipids",
                     "phosphatidylserine", "lipid signaling", "mechanotransduction"]
DOI_RE = re.compile(r"\b10\.\d{4,9}/\S+", re.I)


# ------------------------------------------------------------------ plan


def test_original_question_is_first_and_verbatim():
    plan = expand_queries(f"  {PLATELET_Q}  ")
    first = plan.queries[0]
    assert (first.query_id, first.kind, first.pubmed, first.crossref) == ("q1", "original", PLATELET_Q, PLATELET_Q)


def test_platelet_example_produces_focused_queries_for_all_listed_concepts():
    plan = expand_queries(PLATELET_Q)
    focused = [q for q in plan.queries if q.kind == "focused"]
    assert 3 <= len(focused) <= MAX_FOCUSED_QUERIES
    blob = " ".join(q.pubmed + " " + q.crossref for q in focused)
    for concept in EXPECTED_CONCEPTS:
        assert concept in blob, concept
    assert plan.anchor == "platelet activation"
    for q in focused:                                   # anchor combined with exactly one facet
        assert q.pubmed.startswith('"platelet activation"[tiab] AND ')
        assert q.crossref.startswith("platelet activation ")
        assert q.concepts[0] == "platelet_activation" and len(q.concepts) == 2
    assert [q.concepts[1] for q in focused] == ["shear_stress", "mechanotransduction", "platelet_membrane",
                                                "membrane_lipids", "phosphatidylserine", "lipid_signaling"]
    assert [q.query_id for q in plan.queries] == [f"q{i}" for i in range(1, len(plan.queries) + 1)]
    ids = {c["concept_id"] for c in plan.concepts}
    assert {"platelet_activation", "shear_stress", "mechanotransduction", "platelet_membrane", "membrane_lipids",
            "phosphatidylserine", "lipid_signaling"} <= ids


def test_pubmed_and_crossref_formats():
    q = expand_queries(PLATELET_Q).queries[1]
    assert q.pubmed == '"platelet activation"[tiab] AND ("shear stress"[tiab] OR "shear-induced"[tiab])'
    assert q.crossref == "platelet activation shear stress shear-induced"
    assert "[" not in q.crossref and "AND" not in q.crossref


def test_deterministic_deduplicated_and_capped():
    a, b = expand_queries(PLATELET_Q), expand_queries(PLATELET_Q)
    assert a == b and a.to_json() == b.to_json()
    keys = [(q.pubmed, q.crossref) for q in a.queries]
    assert len(keys) == len(set(keys))
    assert len(a.queries) <= 1 + MAX_FOCUSED_QUERIES
    many = expand_queries("platelet membrane lipid phosphatidylserine shear mechanosensing procoagulant activation")
    assert len(many.queries) <= 1 + MAX_FOCUSED_QUERIES


def test_no_concept_match_gives_only_the_original_question():
    plan = expand_queries("How does sleep affect memory consolidation?")
    assert len(plan.queries) == 1 and plan.queries[0].kind == "original"
    assert plan.notes and "only the original question" in plan.notes[0]
    assert single_query_plan("x y z").queries[0].pubmed == "x y z"


def test_facet_without_curated_anchor_uses_question_keywords():
    plan = expand_queries("Does lipid signaling regulate neuronal growth?")
    assert plan.anchor == "neuronal growth"
    focused = [q for q in plan.queries if q.kind == "focused"]
    assert focused and all(q.pubmed.startswith("(neuronal[tiab] AND growth[tiab]) AND ") for q in focused)
    assert all(q.concepts[0] == "keyword_anchor" for q in focused)


def test_queries_never_contain_identifiers_names_years_or_citations():
    q = ("Did Smith 2019 (doi 10.1234/abc.567, PMID 12345678, https://example.org/paper) show lipid-related "
         "platelet activation under shear in Blood journal?")
    plan = expand_queries(q)
    focused = [x for x in plan.queries if x.kind == "focused"]
    assert focused
    vocabulary = {w for c in CONCEPTS for t in c.terms for w in t.split()}
    for x in focused:
        text = x.pubmed + " " + x.crossref
        assert not DOI_RE.search(text) and not re.search(r"\d", text)
        for bad in ("Smith", "smith", "2019", "12345678", "Blood", "blood", "journal", "http", "doi"):
            assert bad not in text
        assert set(x.crossref.split()) <= vocabulary          # only curated concept terms
    assert not any(re.search(r"\d|/|\.", k) for k in keywords(q))
    assert "doi" not in json.dumps([c for c in plan.concepts])


def test_empty_question_rejected():
    with pytest.raises(ValueError):
        expand_queries("   ")


# ------------------------------------------------------------------ merge


def _rec(pmid: str, db: str = "pubmed") -> Record:
    return Record(title=f"t{pmid}", pmid=pmid, source_database=db, retrieval_timestamp="2026-01-01T00:00:00Z")


def test_merge_single_outcome_is_unchanged():
    o = SearchOutcome(database="pubmed", status="partial", records=[_rec("1"), _rec("1")], total_hits=9)
    merged, stats = merge_query_outcomes("pubmed", [o], 5)
    assert merged is o and stats == [{"status": "partial", "total_hits_reported": 9, "retrieved": 2,
                                      "contributed": 2}]


def test_merge_interleaves_skips_repeated_hits_and_caps():
    a = SearchOutcome("pubmed", "ok", [_rec("1"), _rec("2"), _rec("3")], 30)
    b = SearchOutcome("pubmed", "ok", [_rec("1"), _rec("4")], 20)
    c = SearchOutcome("pubmed", "failed")
    merged, stats = merge_query_outcomes("pubmed", [a, b, c], 3)
    assert [r.pmid for r in merged.records] == ["1", "2", "4"]           # rank 1 of each, then rank 2 of each; cap 3
    assert merged.status == "partial" and merged.total_hits == 30
    assert [s["contributed"] for s in stats] == [2, 1, 0]
    all_failed, _ = merge_query_outcomes("pubmed", [c, c], 3)
    assert all_failed.status == "failed" and all_failed.records == []


# ------------------------------------------------------------------ retrieval integration


def term_handler(pubmed_ids: dict[str, list[str]], crossref_dois: dict[str, list[str]]):
    """PubMed/Crossref mock whose search results depend on the query; lookups resolve everything."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = request.url.path
        if path.endswith("esearch.fcgi"):
            return json_response(esearch_payload(pubmed_ids.get(request.url.params["term"], [])))
        if path.endswith("esummary.fcgi"):
            ids = request.url.params["id"].split(",")
            return json_response(esummary_payload([esummary_doc(i, doi=f"10.1000/p{i}", title=f"Paper {i}")
                                                   for i in ids]))
        if path == "/works":
            dois = crossref_dois.get(request.url.params["query.bibliographic"], [])
            return json_response(crossref_list_payload([crossref_work(d, title=f"Paper {d[-3:].lstrip('p')}")
                                                        for d in dois]))
        if path.startswith("/works/"):
            doi = path[len("/works/"):]
            return json_response(crossref_work_payload(crossref_work(doi, title=f"Paper {doi[-3:].lstrip('p')}")))
        return httpx.Response(500)

    handler.requests = requests  # type: ignore[attr-defined]
    return handler


def run(tmp_path, question, handler, settings, **kw):
    return run_investigation(question, output_dir=tmp_path / "runs", settings=settings, client=mock_client(handler),
                             sleep=lambda s: None, now=lambda: FIXED, **kw)


def test_same_record_across_queries_is_kept_once_and_dedup_unchanged(tmp_path, settings):
    plan = expand_queries(PLATELET_Q)
    q1, q2, q3 = plan.queries[0], plan.queries[1], plan.queries[2]
    handler = term_handler(
        {q1.pubmed: ["111"], q2.pubmed: ["111", "333"], q3.pubmed: ["444"]},
        {q1.crossref: ["10.1000/p111"], q2.crossref: ["10.1000/p111"]})
    result = run(tmp_path, PLATELET_Q, handler, settings, max_results=10)
    s = result.summary
    pmids = [r.pmid for r in result.records]
    assert sorted(p for p in pmids if p) == ["111", "333", "444"]           # 111 found by 2 queries: once
    assert s["retrieved_per_source"] == {"pubmed": 3, "crossref": 1}
    assert s["duplicates_merged"] == 1                                       # only the cross-database DOI match
    rec111 = next(r for r in result.records if r.pmid == "111")
    assert [p.source_database for p in rec111.provenance] == ["pubmed", "crossref"]
    assert {v.status for v in result.verification} == {"verified"}


def test_expansion_changes_nothing_when_all_queries_return_the_same_hits(tmp_path, settings):
    from conftest import api_handler

    on = run(tmp_path / "on", PLATELET_Q, api_handler, settings, max_results=5)
    off = run(tmp_path / "off", PLATELET_Q, api_handler, settings, max_results=5, query_expansion=False)
    assert [r.model_dump() for r in on.records] == [r.model_dump() for r in off.records]
    for key in ("retrieved_per_source", "unique_records", "duplicates_merged", "verification_counts"):
        assert on.summary[key] == off.summary[key]
    assert [v.status for v in on.verification] == [v.status for v in off.verification]


def test_candidate_pool_is_larger_than_final_selection(tmp_path, settings):
    """Every query's hits enter the candidate pool; only max_selected (default 2 * max_results) are verified."""
    plan = expand_queries(PLATELET_Q)
    ids = {q.pubmed: [str(100 + 10 * i + k) for k in range(5)] for i, q in enumerate(plan.queries)}
    handler = term_handler(ids, {})
    result = run(tmp_path, PLATELET_Q, handler, settings, max_results=4)
    s = result.summary
    assert s["retrieved_per_source"]["pubmed"] == 5 * len(plan.queries)     # no truncation before dedup
    assert s["unique_records"] == 5 * len(plan.queries)
    assert len(result.records) == 8 and s["selection"]["target"] == 8
    assert len(s["selection"]["selected_record_ids"]) == 8
    esearch = [r for r in handler.requests if r.url.path.endswith("esearch.fcgi")]
    assert {r.url.params["retmax"] for r in esearch} == {"10"}                # max(pool 10, max_results 4)


def test_single_query_retrieval_still_works(tmp_path, settings):
    q = "How does sleep affect memory consolidation?"
    handler = term_handler({q: ["111"]}, {q: ["10.1000/p222"]})
    result = run(tmp_path, q, handler, settings)
    searches = [r for r in handler.requests if r.url.path.endswith("esearch.fcgi") or r.url.path == "/works"]
    assert len(searches) == 2
    assert result.summary["unique_records"] == 2 and result.summary["query_expansion"]["enabled"] is True
    assert len(result.summary["query_expansion"]["queries"]) == 1


def test_queries_recorded_in_search_log_and_summary(tmp_path, settings):
    from conftest import api_handler

    result = run(tmp_path, PLATELET_Q, api_handler, settings)
    plan = expand_queries(PLATELET_Q)
    for name in ("search_log.json", "summary.json"):
        data = json.loads((result.run_dir / name).read_text())
        qe = data["query_expansion"]
        assert qe["enabled"] is True and "no model" in qe["method"]
        assert [q["pubmed"] for q in qe["queries"]] == [q.pubmed for q in plan.queries]
        assert [q["concepts"] for q in qe["queries"]] == [list(q.concepts) for q in plan.queries]
        assert set(qe["queries"][0]["results"]) == {"pubmed", "crossref"}
        assert {c["concept_id"] for c in qe["concepts"]} >= {"platelet_activation", "lipid_signaling"}
    log = json.loads((result.run_dir / "search_log.json").read_text())
    searched = [e["query"] for e in log["requests"] if e["stage"] == "search"]
    assert searched == [x for q in plan.queries for x in (q.pubmed, q.crossref)]
    summary = json.loads((result.run_dir / "summary.json").read_text())
    assert summary["query_used"] == PLATELET_Q and "deterministic rule-based expansion" in summary["query_generation"]


def test_expansion_can_be_disabled_by_env_parameter_or_cli(tmp_path, settings, monkeypatch, fast_sleep):
    from conftest import api_handler

    assert Settings.from_env({}).query_expansion is True
    assert Settings.from_env({"SCIFORGE_QUERY_EXPANSION": "false"}).query_expansion is False
    with pytest.raises(ConfigError):
        Settings.from_env({"SCIFORGE_QUERY_EXPANSION": "maybe"})
    off = run(tmp_path / "p", PLATELET_Q, api_handler, settings, query_expansion=False)
    assert off.summary["query_expansion"]["enabled"] is False
    assert len(off.summary["query_expansion"]["queries"]) == 1
    assert off.summary["query_generation"].startswith("none")

    requests: list[httpx.Request] = []

    def recording(request):
        requests.append(request)
        return api_handler(request)

    assert main(["investigate", PLATELET_Q, "--no-query-expansion", "--output-dir", str(tmp_path / "cli")],
                client=mock_client(recording)) == 0
    assert len([r for r in requests if r.url.path.endswith("esearch.fcgi") or r.url.path == "/works"]) == 2
    monkeypatch.setenv("SCIFORGE_QUERY_EXPANSION", "false")
    requests.clear()
    assert main(["investigate", PLATELET_Q, "--output-dir", str(tmp_path / "cli2")], client=mock_client(recording)) == 0
    assert len([r for r in requests if r.url.path.endswith("esearch.fcgi") or r.url.path == "/works"]) == 2
    monkeypatch.delenv("SCIFORGE_QUERY_EXPANSION")
    requests.clear()
    assert main(["investigate", PLATELET_Q, "--output-dir", str(tmp_path / "cli3")], client=mock_client(recording)) == 0
    n = len(expand_queries(PLATELET_Q).queries)
    assert len([r for r in requests if r.url.path.endswith("esearch.fcgi") or r.url.path == "/works"]) == 2 * n
