"""Expanded-query candidate pool -> dedup -> deterministic selection -> verification backfill (offline only)."""

from __future__ import annotations

import json
import random
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
from sciforge.config import ConfigError, Settings
from sciforge.models import Record, VerificationResult
from sciforge.pipeline import run_investigation
from sciforge.query_expansion import expand_queries
from sciforge.source_selection import (
    NEAR_DUPLICATE_PENALTY,
    REDUNDANCY_PENALTY,
    VERIFY_BACKFILL_FACTOR,
    W_ANCHOR,
    W_CONCEPT,
    W_KEYWORD,
    W_NOVELTY,
    rank_candidates,
    score_candidates,
    select_and_verify,
)
from sciforge.stages.report import build_report

FIXED = datetime(2026, 9, 28, 23, 41, 0, tzinfo=timezone.utc)
Q = "What is the role of lipid-related changes in shear-mediated platelet activation?"
TS = "2026-09-28T00:00:00Z"


def rec(pmid: str, title: str | None, *, journal: str | None = None) -> Record:
    return Record(title=title, pmid=pmid, journal=journal, source_database="pubmed", retrieval_timestamp=TS)


def origins_for(records, query_id="q1"):
    return {r.record_id: [(query_id, "pubmed", i + 1)] for i, r in enumerate(records)}


# ------------------------------------------------------------------ scoring


def test_scores_are_deterministic_and_order_independent():
    recs = [rec(str(i), t) for i, t in enumerate([
        "Shear stress triggers platelet activation", "Phosphatidylserine exposure in procoagulant platelets",
        "Membrane lipids of platelets", "Unrelated cardiology note", "Shear stress triggers platelet activation in vitro",
        "Lipid signaling in platelets"])]
    origins = origins_for(recs)
    a = rank_candidates(score_candidates(Q, recs, origins))
    b = rank_candidates(score_candidates(Q, recs, origins))
    shuffled = list(recs)
    random.Random(7).shuffle(shuffled)
    c = rank_candidates(score_candidates(Q, shuffled, origins))
    assert [x.to_json() for x in a] == [x.to_json() for x in b]
    assert [x.record_id for x in a] == [x.record_id for x in c]
    assert [x.selection_score for x in a] == [x.selection_score for x in c]


def test_score_components_from_title_and_query_origin():
    r = rec("1", "Shear stress induces phosphatidylserine exposure during platelet activation")
    origins = {r.record_id: [("q1", "pubmed", 1), ("q2", "pubmed", 3), ("q6", "crossref", 2)]}
    (s,) = score_candidates(Q, [r], origins)
    assert s.anchor_matched is True
    assert set(s.concepts_matched) >= {"platelet_activation", "shear_stress", "phosphatidylserine"}
    assert s.components["anchor"] == W_ANCHOR
    assert s.components["concept_coverage"] == W_CONCEPT * len(s.facet_concepts)
    assert s.components["core_term_relevance"] == W_KEYWORD * len(s.keyword_hits)
    assert {"shear", "platel", "activa"} <= set(s.keyword_hits)
    assert s.components["query_origin"] == 2 + 3                    # verbatim query + 3 distinct queries
    assert s.query_ids == ("q1", "q2", "q6") and s.databases == ("crossref", "pubmed") and s.best_rank == 1
    assert s.base_score == sum(s.components.values())
    assert s.text_used == ("title",)


def test_scoring_uses_only_retrieved_title_or_abstract_text():
    no_title = rec("1", None, journal="Shear Stress and Platelet Activation Letters")
    (s,) = score_candidates(Q, [no_title], {})
    assert s.concepts_matched == () and s.keyword_hits == () and s.base_score == 0 and s.text_used == ()
    plain = rec("2", "A case report")
    (without,) = score_candidates(Q, [plain], {})
    (with_abs,) = score_candidates(Q, [plain], {}, abstracts={plain.record_id: "Phosphatidylserine was measured."})
    assert "phosphatidylserine" not in without.concepts_matched
    assert "phosphatidylserine" in with_abs.concepts_matched and with_abs.text_used == ("title", "abstract")


def test_greedy_selection_prefers_concept_diversity_over_near_duplicates():
    shear = [rec(str(i), f"Shear stress platelet activation study {i}") for i in range(1, 4)]
    ps = rec("10", "Phosphatidylserine exposure in platelets")
    lipids = rec("11", "Membrane phospholipids in platelets")
    recs = [*shear, ps, lipids]
    origins = {**{r.record_id: [("q1", "pubmed", i + 1), ("q2", "pubmed", i + 1)] for i, r in enumerate(shear)},
               ps.record_id: [("q6", "pubmed", 5)], lipids.record_id: [("q5", "pubmed", 5)]}
    scores = {s.record_id: s for s in score_candidates(Q, recs, origins)}
    assert all(scores[r.record_id].base_score > scores[ps.record_id].base_score for r in shear)   # shear wins raw
    ranked = rank_candidates(list(scores.values()))
    top3 = [c.record_id for c in ranked[:3]]
    assert ps.record_id in top3 and lipids.record_id in top3                  # diverse concepts beat duplicates
    assert sum(r.record_id in top3 for r in shear) == 1
    second_shear = next(c for c in ranked if c.record_id in {r.record_id for r in shear} and c.rank > 1)
    assert second_shear.same_profile_selected == 1 and second_shear.new_concepts == ()
    assert second_shear.selection_score == scores[second_shear.record_id].base_score - REDUNDANCY_PENALTY
    first = ranked[0]
    assert first.selection_score == first.score.base_score + W_NOVELTY * len(first.new_concepts)


def test_near_duplicate_titles_are_penalised():
    a = rec("1", "Shear stress activates platelets through membrane lipid remodeling")
    b = rec("2", "Shear stress activates platelets through membrane lipid remodelling")   # spelling variant
    ranked = rank_candidates(score_candidates(Q, [a, b], origins_for([a, b])))
    assert ranked[1].near_duplicate_of == ranked[0].record_id
    assert ranked[1].selection_score == (ranked[1].score.base_score - REDUNDANCY_PENALTY
                                         - NEAR_DUPLICATE_PENALTY)


def test_ties_are_broken_by_record_id():
    recs = [rec(str(i), "Identical title") for i in (5, 3, 9)]
    ranked = rank_candidates(score_candidates("How does sleep affect memory?", recs, {}))
    assert [c.record_id for c in ranked][0] == min(r.record_id for r in recs)


# ------------------------------------------------------------------ verification backfill


def ver(record: Record, status: str) -> VerificationResult:
    return VerificationResult(record_id=record.record_id, status=status, reasons=[], verified_at=TS)


def _ranked(n):
    recs = [rec(str(i), f"Shear stress platelet activation {i}") for i in range(n)]
    return recs, rank_candidates(score_candidates(Q, recs, origins_for(recs))), {r.record_id: r for r in recs}


def test_failed_verification_is_backfilled_from_the_ranking():
    recs, ranked, by_id = _ranked(10)
    order = [c.record_id for c in ranked]
    failing = {order[1], order[3]}
    calls = []

    def verify(batch):
        calls.append([r.record_id for r in batch])
        return [ver(r, "not_verified" if r.record_id in failing else "verified") for r in batch]

    out = select_and_verify(ranked, by_id, verify, 5)
    assert out.selected_ids == [x for x in order[:7] if x not in failing]      # still 5, from verified records
    assert calls == [order[:5], order[5:7]] and out.to_json()["backfilled"] is True
    assert out.to_json()["verification_failed_ids"] == [order[1], order[3]]


def test_backfill_is_bounded_and_partial_only_by_opt_in():
    recs, ranked, by_id = _ranked(30)
    out = select_and_verify(ranked, by_id, lambda b: [ver(r, "partially_verified") for r in b], 5)
    assert out.selected_ids == [] and len(out.checked) == 5 * VERIFY_BACKFILL_FACTOR
    opt_in = select_and_verify(ranked, by_id, lambda b: [ver(r, "partially_verified") for r in b], 5,
                               accept_partially_verified=True)
    assert len(opt_in.selected_ids) == 5 and len(opt_in.checked) == 5


def test_verification_exception_keeps_earlier_rounds():
    recs, ranked, by_id = _ranked(10)
    state = {"n": 0}

    def verify(batch):
        state["n"] += 1
        if state["n"] == 2:
            raise RuntimeError("boom")
        return [ver(r, "verified" if i else "not_verified") for i, r in enumerate(batch)]

    out = select_and_verify(ranked, by_id, verify, 5)
    assert len(out.selected_ids) == 4 and out.error == "RuntimeError: boom"


# ------------------------------------------------------------------ settings


def test_candidate_pool_setting_default_and_validation():
    assert Settings.from_env({}).candidate_pool_per_query == 10
    assert Settings.from_env({"SCIFORGE_CANDIDATE_POOL_PER_QUERY": " 25 "}).candidate_pool_per_query == 25
    assert Settings.from_env({"SCIFORGE_CANDIDATE_POOL_PER_QUERY": ""}).candidate_pool_per_query == 10
    for bad in ("0", "-1", "101", "ten", "2.5"):
        with pytest.raises(ConfigError, match="SCIFORGE_CANDIDATE_POOL_PER_QUERY"):
            Settings.from_env({"SCIFORGE_CANDIDATE_POOL_PER_QUERY": bad})


# ------------------------------------------------------------------ pipeline integration


def handler_for(pubmed_ids, crossref_dois, *, titles=None, not_found=()):
    titles = titles or {}
    requests: list[httpx.Request] = []

    def title(key):
        return titles.get(key, f"Paper {key}")

    def handler(request):
        requests.append(request)
        path = request.url.path
        if path.endswith("esearch.fcgi"):
            ids = pubmed_ids.get(request.url.params["term"], [])
            return json_response(esearch_payload(ids[:int(request.url.params["retmax"])]))
        if path.endswith("esummary.fcgi"):
            ids = [i for i in request.url.params["id"].split(",") if i not in not_found]
            missing = [i for i in request.url.params["id"].split(",") if i in not_found]
            return json_response(esummary_payload([esummary_doc(i, doi=f"10.1000/p{i}", title=title(i))
                                                   for i in ids], missing=missing))
        if path == "/works":
            dois = crossref_dois.get(request.url.params["query.bibliographic"], [])[:int(request.url.params["rows"])]
            return json_response(crossref_list_payload([crossref_work(d, title=title(d.rsplit("p", 1)[-1]))
                                                        for d in dois]))
        if path.startswith("/works/"):
            doi = path[len("/works/"):]
            if doi.rsplit("p", 1)[-1] in not_found:
                return httpx.Response(404)
            return json_response(crossref_work_payload(crossref_work(doi, title=title(doi.rsplit("p", 1)[-1]))))
        return httpx.Response(500)

    handler.requests = requests  # type: ignore[attr-defined]
    return handler


def run(tmp_path, handler, settings, **kw):
    return run_investigation(Q, output_dir=tmp_path / "runs", settings=settings, client=mock_client(handler),
                             sleep=lambda s: None, now=lambda: FIXED, **kw)


def test_larger_candidate_pool_is_collected_before_the_final_source_limit(tmp_path, settings):
    plan = expand_queries(Q)
    ids = {q.pubmed: [str(1000 + 100 * i + k) for k in range(12)] for i, q in enumerate(plan.queries)}
    handler = handler_for(ids, {})
    result = run(tmp_path, handler, settings, max_results=5, max_selected=5, candidate_pool_per_query=8)
    esearch = [r for r in handler.requests if r.url.path.endswith("esearch.fcgi")]
    assert len(esearch) == len(plan.queries) and {r.url.params["retmax"] for r in esearch} == {"8"}
    s = result.summary
    assert s["total_retrieved"] == 8 * len(plan.queries) > 5
    assert s["candidates_per_query"]["pubmed"][0] == {"query_id": "q1", "retrieved": 8, "new_candidates": 8}
    assert len(result.records) == 5 and len(s["selection"]["selected_record_ids"]) == 5
    assert s["parameters"]["records_requested_per_query"] == 8 and s["parameters"]["max_selected"] == 5


def test_dedup_runs_before_selection_and_is_logged(tmp_path, settings):
    plan = expand_queries(Q)
    q1, q2 = plan.queries[0], plan.queries[1]
    handler = handler_for({q1.pubmed: ["111", "222"], q2.pubmed: ["333"]},
                          {q1.crossref: ["10.1000/p111"], q2.crossref: ["10.1000/p444"]})
    result = run(tmp_path, handler, settings, max_results=5)
    log = json.loads((result.run_dir / "search_log.json").read_text())
    dd = log["deduplication"]
    assert (dd["candidates_before"], dd["unique_after"], dd["duplicates_merged"]) == (5, 4, 1)
    (merge,) = dd["merges"]
    assert merge["matched_on"] == ["doi"] and merge["source_databases"] == ["pubmed", "crossref"]
    sel = log["selection"]
    assert sel["candidates_scored"] == 4                                   # scored AFTER dedup
    merged = next(c for c in sel["candidates"] if c["record_id"] == merge["record_id"])
    assert merged["databases"] == ["crossref", "pubmed"] and merged["query_ids"] == ["q1"]
    assert {"anchor", "concept_coverage", "core_term_relevance", "query_origin"} == set(merged["components"])
    assert [c["rank"] for c in sel["candidates"]] == [1, 2, 3, 4]


def test_pipeline_backfill_when_top_candidate_fails_verification(tmp_path, settings, monkeypatch):
    from sciforge import verify as verify_mod

    plan = expand_queries(Q)
    titles = {"1": "Shear stress and phosphatidylserine in platelet activation",
              "2": "Membrane lipids in platelet activation", "3": "Platelet activation under shear"}
    handler = handler_for({plan.queries[0].pubmed: ["1", "2", "3"]}, {}, titles=titles)
    original = verify_mod.Verifier.verify_all
    first_rec: list[str] = []

    def failing_first(self, records):
        results = original(self, records)
        if not first_rec:
            first_rec.append(records[0].record_id)
            results[0] = results[0].model_copy(update={"status": "not_verified"})
        return results

    monkeypatch.setattr(verify_mod.Verifier, "verify_all", failing_first)
    result = run(tmp_path, handler, settings, max_results=5, max_selected=2)
    sel = result.summary["selection"]
    assert sel["backfilled"] is True and sel["candidates_verified"] == 3
    assert len(sel["selected_record_ids"]) == 2 and first_rec[0] not in sel["selected_record_ids"]
    assert sel["verification_failed_ids"] == first_rec
    assert len(result.records) == 3                                    # every verified candidate is reported


def test_search_log_and_summary_record_queries_counts_scores_and_selection(tmp_path, settings):
    from conftest import api_handler

    result = run(tmp_path, api_handler, settings)
    plan = expand_queries(Q)
    log = json.loads((result.run_dir / "search_log.json").read_text())
    summary = json.loads((result.run_dir / "summary.json").read_text())
    for q, logged in zip(plan.queries, log["query_expansion"]["queries"]):
        assert (logged["pubmed"], logged["crossref"]) == (q.pubmed, q.crossref)
        for db in ("pubmed", "crossref"):
            assert {"requested", "retrieved", "contributed"} <= set(logged["results"][db])
    assert summary["queries_used"] == {"pubmed": [q.pubmed for q in plan.queries],
                                       "crossref": [q.crossref for q in plan.queries]}
    assert set(summary["deduplication"]) == {"candidates_before", "unique_after", "duplicates_merged"}
    assert summary["selection"]["selected_record_ids"] == log["selection"]["selected_record_ids"]
    assert all("components" in c and "base_score" in c and "selection_score" in c
               for c in log["selection"]["candidates"])
    assert [c["record_id"] for c in summary["selection"]["selected_scores"]] == [
        c["record_id"] for c in log["selection"]["candidates"] if c["record_id"] in summary["selection"]["selected_record_ids"]]


def test_fallback_report_describes_every_query_and_selection(tmp_path, settings):
    from conftest import api_handler

    on = run(tmp_path / "on", api_handler, settings)
    off = run(tmp_path / "off", api_handler, settings, query_expansion=False)
    plan = expand_queries(Q)

    def section_b(summary):
        report, _ = build_report(question=Q, question_definition=None, search_summary=summary,
                                 source_texts={"sources": []}, evidence={"accepted": []}, gaps=None, hypotheses=None,
                                 narrative=None, records=[], verification=[], stage_notes=[])
        return report.split("## B.")[1].split("## C.")[0]

    b_on = section_b(on.summary)
    assert "Query used (verbatim)" not in b_on and "searches the question verbatim" not in b_on
    assert f"Query expansion:** enabled; {len(plan.queries)} queries per database" in b_on
    for q in plan.queries:
        assert f"  - {q.query_id} pubmed: " in b_on and f"  - {q.query_id} crossref: " in b_on
    assert "Candidate pool:" in b_on and "Selection:" in b_on and "retrieved" in b_on
    b_off = section_b(off.summary)
    assert "Query expansion:** disabled; 1 query per database" in b_off
    assert "query expansion disabled" in off.summary["query_generation"]


def test_web_live_mode_selects_at_most_max_sources_verified_records(tmp_path, monkeypatch):
    """Live path (MockTransport + FakeModelClient): big candidate pool, final set = model max_sources."""
    from conftest import FAKE_XAI_KEY, FAKE_XAI_MODEL, Sleeper
    from m2_support import question_output
    from sciforge import app_service as svc
    from sciforge.llm.fake import FakeModelClient

    monkeypatch.chdir(tmp_path)
    plan = expand_queries(Q)
    ids = {q.pubmed: [str(2000 + 100 * i + k) for k in range(12)] for i, q in enumerate(plan.queries)}
    handler = handler_for(ids, {})

    def live(request):
        if request.url.path.endswith("efetch.fcgi"):
            return httpx.Response(200, text="<PubmedArticleSet></PubmedArticleSet>")
        return handler(request)

    env = {"SCIFORGE_LIVE_ENABLED": "true", "XAI_API_KEY": FAKE_XAI_KEY, "XAI_MODEL": FAKE_XAI_MODEL,
           "SCIFORGE_MAX_SPEND_USD": "none", "SCIFORGE_MODEL_REASONING_EFFORT_EVIDENCE": "low"}
    clients = []

    def factory(model_settings):
        clients.append(FakeModelClient([question_output()] + [{"items": []}] * 10))
        return clients[-1]

    result = svc.run_web_investigation(
        svc.InvestigationRequest(question=Q, mode=svc.MODE_LIVE, max_sources=5), environ=env,
        live_http_client=mock_client(live), live_model_client_factory=factory, sleep=Sleeper())
    assert result.ok
    esearch = [r for r in handler.requests if r.url.path.endswith("esearch.fcgi")]
    assert {r.url.params["retmax"] for r in esearch} == {"10"}                 # pool 10 > max_sources 5
    assert len(result.sources) == 5                                           # 5 selected + verified records
    assert clients[0].reasoning_efforts[0] == "high"                          # question stage: global default
