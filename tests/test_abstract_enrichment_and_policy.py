"""Bounded abstract enrichment, metadata-based source classification and source policy (offline only)."""

from __future__ import annotations

import json
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
from sciforge.abstract_enrichment import choose_enrichment_candidates, enrich_abstracts
from sciforge.cli import format_summary
from sciforge.config import ConfigError, Settings
from sciforge.models import Record
from sciforge.pipeline import run_investigation
from sciforge.source_classification import (
    BOOK,
    CONFERENCE,
    JOURNAL_ARTICLE,
    PREPRINT,
    UNKNOWN,
    apply_source_policy,
    classify_metadata,
    classify_record,
    crossref_metadata,
    pubmed_metadata,
)
from sciforge.source_selection import score_candidates
from sciforge.stages.report import build_report

FIXED = datetime(2026, 9, 28, 23, 41, 0, tzinfo=timezone.utc)
Q = "What is the role of lipid-related changes in shear-mediated platelet activation?"
TS = "2026-09-28T00:00:00Z"
RELEVANT_ABSTRACT = ("Shear stress triggered platelet activation with phosphatidylserine exposure and membrane "
                     "lipid remodeling in shear-mediated platelet activation experiments.")


# ------------------------------------------------------------------ mocked APIs


class FakeAbstractFetcher:
    """Offline stand-in for the batched PubMed efetch (records every call)."""

    def __init__(self, abstracts: dict[str, str] | None = None, fail: bool = False) -> None:
        self.abstracts = abstracts or {}
        self.calls: list[list[str]] = []
        self.fail = fail

    def __call__(self, pmids: list[str]) -> dict[str, str | None]:
        self.calls.append(list(pmids))
        if self.fail:
            raise RuntimeError("efetch down")
        return {p: self.abstracts.get(p) for p in pmids}

    @property
    def requested(self) -> list[str]:
        return [p for c in self.calls for p in c]


def api(pubmed: dict[str, dict], crossref: dict[str, dict]):
    """pubmed: pmid -> {title, pubtype, journal}; crossref: doi -> {title, type, subtype, abstract, publisher}."""
    requests: list[httpx.Request] = []

    def doc(pmid):
        spec = pubmed[pmid]
        d = esummary_doc(pmid, doi=f"10.7777/pm{pmid}", title=spec["title"], journal=spec.get("journal", "J Test"))
        d["pubtype"] = spec.get("pubtype", ["Journal Article"])
        if spec.get("journal") == "":
            d["fulljournalname"], d["source"] = "", ""
        return d

    def work(doi):
        spec = crossref[doi]
        w = crossref_work(doi, title=spec["title"], journal=spec.get("container", "Journal of Tests"))
        for key in ("type", "subtype", "publisher", "abstract"):
            if spec.get(key):
                w[key] = spec[key]
        return w

    def handler(request):
        requests.append(request)
        path = request.url.path
        if path.endswith("esearch.fcgi"):
            return json_response(esearch_payload(sorted(pubmed)[:int(request.url.params["retmax"])]))
        if path.endswith("esummary.fcgi"):
            ids = request.url.params["id"].split(",")
            return json_response(esummary_payload([doc(i) for i in ids if i in pubmed]))
        if path == "/works":
            return json_response(crossref_list_payload([work(d) for d in sorted(crossref)]
                                                       [:int(request.url.params["rows"])]))
        if path.startswith("/works/"):
            doi = path[len("/works/"):]
            match = next((d for d in crossref if d.lower() == doi.lower()), None)
            if match:
                return json_response(crossref_work_payload(work(match)))
            pm = next((p for p in pubmed if f"10.7777/pm{p}" == doi.lower()), None)
            if pm:
                return json_response(crossref_work_payload(crossref_work(doi, title=pubmed[pm]["title"])))
            return httpx.Response(404)
        if path.endswith("efetch.fcgi"):
            raise AssertionError("tests inject an abstract fetcher; no efetch request expected")
        return httpx.Response(500)

    handler.requests = requests  # type: ignore[attr-defined]
    return handler


def run(tmp_path, handler, settings, **kw):
    kw.setdefault("query_expansion", False)
    return run_investigation(Q, output_dir=tmp_path / "runs", settings=settings, client=mock_client(handler),
                             sleep=lambda s: None, now=lambda: FIXED, **kw)


def log_of(result):
    return json.loads((result.run_dir / "search_log.json").read_text())


# ------------------------------------------------------------------ enrichment: limit, determinism, ranking


def rec(pmid, title, doi=None):
    return Record(title=title, pmid=pmid, doi=doi, source_database="pubmed", retrieval_timestamp=TS)


def test_enrichment_never_exceeds_limit_and_batches(tmp_path):
    recs = [rec(str(100 + i), f"Platelet paper {i}") for i in range(30)]
    by_id = {r.record_id: r for r in recs}
    pre = score_candidates(Q, recs, {r.record_id: [("q1", "pubmed", i + 1)] for i, r in enumerate(recs)})
    for limit in (0, 1, 7, 29, 30, 50):
        fetch = FakeAbstractFetcher({r.pmid: RELEVANT_ABSTRACT for r in recs})
        considered = choose_enrichment_candidates(pre, limit)
        assert len(considered) == min(limit, 30)
        out = enrich_abstracts(by_id, [r.record_id for r in recs], limit=limit, fetch_pubmed=fetch, batch_size=3)
        assert len(out.considered) <= limit and len(fetch.requested) <= limit
        assert all(len(c) <= 3 for c in fetch.calls)
        assert len(out.abstracts) == min(limit, 30)
    zero = FakeAbstractFetcher()
    enrich_abstracts(by_id, [r.record_id for r in recs], limit=0, fetch_pubmed=zero)
    assert zero.calls == []


def test_enrichment_choice_is_deterministic_and_failure_keeps_title_only():
    recs = [rec(str(i + 1), t) for i, t in enumerate(["Platelet activation under shear", "Coral reefs",
                                                  "Phosphatidylserine in platelets", "Soil bacteria"])]
    origins = {r.record_id: [("q1", "pubmed", i + 1)] for i, r in enumerate(recs)}
    a = choose_enrichment_candidates(score_candidates(Q, recs, origins), 2)
    b = choose_enrichment_candidates(score_candidates(Q, list(reversed(recs)), origins), 2)
    assert a == b and len(a) == 2
    with pytest.raises(ValueError):
        choose_enrichment_candidates([], -1)
    out = enrich_abstracts({r.record_id: r for r in recs}, a, limit=2, fetch_pubmed=FakeAbstractFetcher(fail=True))
    assert out.abstracts == {} and set(out.title_only) == set(a) and out.errors
    assert all("efetch failed" in reason for reason in out.title_only.values())


def test_abstract_relevant_pubmed_record_outranks_title_only_less_relevant_crossref(tmp_path, settings):
    pubmed = {"501": {"title": "A prospective cohort investigation"}}
    crossref = {"10.5555/cr1": {"title": "Platelet counts in adults", "type": "journal-article"}}
    fetch = FakeAbstractFetcher({"501": RELEVANT_ABSTRACT})
    with_abs = run(tmp_path / "a", api(pubmed, crossref), settings, max_results=1, max_selected=1,
                   abstract_fetcher=fetch, abstract_enrichment_limit=50)
    without = run(tmp_path / "b", api(pubmed, crossref), settings, max_results=1, max_selected=1,
                  abstract_fetcher=FakeAbstractFetcher(), abstract_enrichment_limit=0)
    pm_id = next(r.record_id for r in with_abs.records if r.pmid == "501")
    log = log_of(with_abs)
    cands = log["selection"]["candidates"]
    assert cands[0]["record_id"] == pm_id                                   # abstract lifts the PubMed record
    assert with_abs.summary["selection"]["selected_record_ids"] == [pm_id]
    top = cands[0]
    assert top["abstract_components"]["abstract_anchor"] > 0 and top["abstract_components"]["abstract_concept_coverage"] > 0
    assert "abstract" in top["text_used"]
    other = next(c for c in cands if c["record_id"] != pm_id)
    assert other["abstract_components"] == {} and other["text_used"] == ["title"]
    # without enrichment the title-only Crossref record wins
    wo = log_of(without)["selection"]["candidates"]
    assert [c["record_id"] for c in wo] == [other["record_id"], pm_id]
    assert without.summary["selection"]["selected_record_ids"] == [other["record_id"]]
    assert top["base_score"] > other["base_score"] > next(c for c in wo if c["record_id"] == pm_id)["base_score"]
    assert fetch.calls == [["501"]]                                         # one batched call, PMIDs only


def test_search_log_records_enrichment_and_is_deterministic(tmp_path, settings):
    pubmed = {str(600 + i): {"title": f"Platelet study {i}"} for i in range(12)}
    crossref = {"10.5555/abs": {"title": "Shear study", "type": "journal-article",
                                "abstract": "<jats:p>Phosphatidylserine exposure in platelets.</jats:p>"},
                "10.5555/noabs": {"title": "Lipid study", "type": "journal-article"}}
    logs = []
    for n in ("x", "y"):
        fetch = FakeAbstractFetcher({"600": RELEVANT_ABSTRACT, "601": RELEVANT_ABSTRACT})
        result = run(tmp_path / n, api(pubmed, crossref), settings, max_results=3, max_selected=3,
                     abstract_fetcher=fetch, abstract_enrichment_limit=5, candidate_pool_per_query=20)
        assert len(fetch.requested) <= 5
        logs.append(log_of(result))
        summary = result.summary
    a, b = logs
    assert a["abstract_enrichment"] == b["abstract_enrichment"] and a["selection"] == b["selection"]
    enr = a["abstract_enrichment"]
    assert enr["limit"] == 5 and len(enr["candidates_considered"]) == 5
    got = {x["record_id"] for x in enr["with_abstract"]}
    title_only = {x["record_id"] for x in enr["title_only"]}
    assert got | title_only == set(enr["candidates_considered"]) and not got & title_only
    assert all(set(x) == {"record_id", "abstract_origin", "abstract_chars"} for x in enr["with_abstract"])
    assert RELEVANT_ABSTRACT not in json.dumps(a)                            # text itself is never logged
    assert all("abstract_components" in c and "source_status" in c for c in a["selection"]["candidates"])
    assert summary["abstract_enrichment"]["considered"] == 5
    assert "Abstract enrichment for ranking" in format_summary(summary, "runs/x")


def test_crossref_inline_abstract_used_when_candidate_considered(tmp_path, settings):
    crossref = {"10.5555/abs": {"title": "Unrelated title words", "type": "journal-article",
                                "abstract": f"<jats:p>{RELEVANT_ABSTRACT}</jats:p>"}}
    result = run(tmp_path, api({}, crossref), settings, max_results=1, max_selected=1,
                 abstract_fetcher=FakeAbstractFetcher(), abstract_enrichment_limit=10)
    enr = log_of(result)["abstract_enrichment"]
    assert [x["abstract_origin"] for x in enr["with_abstract"]] == ["crossref_search_result"]


# ------------------------------------------------------------------ settings


def test_new_settings_defaults_and_validation(monkeypatch):
    s = Settings.from_env()
    assert s.abstract_enrichment_limit == 50 and s.source_policy == "peer_reviewed_preferred"
    for ok in ("0", "500", "7"):
        monkeypatch.setenv("SCIFORGE_ABSTRACT_ENRICHMENT_LIMIT", ok)
        assert Settings.from_env().abstract_enrichment_limit == int(ok)
    for bad in ("-1", "501", "abc", "1.5"):
        monkeypatch.setenv("SCIFORGE_ABSTRACT_ENRICHMENT_LIMIT", bad)
        with pytest.raises(ConfigError):
            Settings.from_env()
    monkeypatch.delenv("SCIFORGE_ABSTRACT_ENRICHMENT_LIMIT")
    for ok in ("allow_all", "peer_reviewed_preferred", "peer_reviewed_only"):
        monkeypatch.setenv("SCIFORGE_SOURCE_POLICY", ok)
        assert Settings.from_env().source_policy == ok
    for bad in ("peer_reviewed", "ALL", "none"):
        monkeypatch.setenv("SCIFORGE_SOURCE_POLICY", bad)
        with pytest.raises(ConfigError):
            Settings.from_env()


# ------------------------------------------------------------------ classification


@pytest.mark.parametrize("work, expected", [
    ({"type": "journal-article", "DOI": "10.1000/x"}, JOURNAL_ARTICLE),
    ({"type": "posted-content", "subtype": "preprint", "DOI": "10.9999/y"}, PREPRINT),
    ({"type": "posted-content", "DOI": "10.1101/2023.01.02.522222"}, PREPRINT),            # bioRxiv DOI pattern
    ({"type": "posted-content", "DOI": "10.9999/z"}, UNKNOWN),                             # no preprint subtype
    ({"type": "proceedings-article", "DOI": "10.1145/1"}, CONFERENCE),
    ({"type": "book-chapter", "DOI": "10.1007/978"}, BOOK),
    ({"type": "book", "DOI": "10.1007/979"}, BOOK),
    ({"type": "dataset", "DOI": "10.5061/dryad"}, UNKNOWN),
    ({"DOI": "10.1000/untyped"}, UNKNOWN),
    ({"type": "journal-article", "DOI": "10.21203/rs.3.rs-12345/v1"}, PREPRINT),          # Research Square
    ({"type": "journal-article", "DOI": "10.2139/ssrn.123456"}, PREPRINT),                # SSRN
    ({"type": "posted-content", "DOI": "10.20944/preprints202301.0001.v1"}, PREPRINT),
    ({"type": "posted-content", "DOI": "10.26434/chemrxiv-2023-abc"}, PREPRINT),
    ({"type": "posted-content", "DOI": "10.48550/arXiv.2301.00001"}, PREPRINT),
    ({"type": "journal-article", "DOI": "10.1101/gr.123456.111"}, JOURNAL_ARTICLE),       # CSHL journal, not bioRxiv
    ({"type": "journal-article", "DOI": "10.1101/cshperspect.a0001"}, JOURNAL_ARTICLE),
])
def test_crossref_classification(work, expected):
    status, basis = classify_metadata(crossref_metadata(work))
    assert status == expected and basis


@pytest.mark.parametrize("doc, doi, expected", [
    ({"pubtype": ["Journal Article"], "fulljournalname": "Blood"}, None, JOURNAL_ARTICLE),
    ({"pubtype": ["Review", "Journal Article"], "source": "Blood"}, None, JOURNAL_ARTICLE),
    ({"pubtype": ["Preprint"], "fulljournalname": "bioRxiv"}, None, PREPRINT),
    ({"pubtype": ["Journal Article", "Preprint"], "fulljournalname": "Some Journal"}, None, PREPRINT),
    ({"pubtype": ["Journal Article"], "fulljournalname": "medRxiv"}, None, PREPRINT),       # known server name
    ({"pubtype": ["Journal Article"], "fulljournalname": "X"}, "10.1101/2021.03.04.433333", PREPRINT),
    ({"pubtype": ["Journal Article"]}, None, UNKNOWN),                                      # no journal: ambiguous
    ({"pubtype": [], "fulljournalname": "Blood"}, None, UNKNOWN),
    ({"pubtype": ["Editorial"], "fulljournalname": "Blood"}, None, UNKNOWN),
    ({"pubtype": ["Congress"], "fulljournalname": "Blood"}, None, CONFERENCE),
])
def test_pubmed_classification(doc, doi, expected):
    assert classify_metadata(pubmed_metadata(doc, doi))[0] == expected


def test_title_wording_never_changes_classification():
    for title in ("A preprint about platelets", "Conference proceedings on shear", "Book chapter: lipids"):
        work = {"type": "journal-article", "DOI": "10.1000/t", "title": [title]}
        assert classify_metadata(crossref_metadata(work))[0] == JOURNAL_ARTICLE


def test_merged_record_classification():
    journal = crossref_metadata({"type": "journal-article", "DOI": "10.1000/a"})
    pm_journal = pubmed_metadata({"pubtype": ["Journal Article"], "fulljournalname": "Blood"})
    pm_pre = pubmed_metadata({"pubtype": ["Preprint"], "fulljournalname": "bioRxiv"})
    conf = crossref_metadata({"type": "proceedings-article", "DOI": "10.1000/a"})
    assert classify_record("r", [journal, pm_journal]).source_status == JOURNAL_ARTICLE
    assert classify_record("r", [journal, pm_pre]).source_status == PREPRINT          # preprint signal wins
    assert classify_record("r", [conf, pm_journal]).source_status == UNKNOWN          # conflict -> unknown
    assert classify_record("r", []).source_status == UNKNOWN


# ------------------------------------------------------------------ source policy


STATUSES = {"pre": PREPRINT, "j1": JOURNAL_ARTICLE, "conf": CONFERENCE, "book": BOOK, "unk": UNKNOWN,
            "j2": JOURNAL_ARTICLE}
RANKED = ["pre", "j1", "conf", "book", "unk", "j2"]


def test_policy_allow_all_keeps_ranking():
    out = apply_source_policy(RANKED, STATUSES, {r: True for r in RANKED}, "allow_all")
    assert out.ordered_ids == RANKED and out.excluded == []


def test_policy_preferred_puts_journal_articles_first_but_keeps_others():
    relevant = {r: True for r in RANKED} | {"j2": False}
    out = apply_source_policy(RANKED, STATUSES, relevant, "peer_reviewed_preferred")
    assert out.ordered_ids == ["j1", "pre", "conf", "book", "unk", "j2"] and out.excluded == []


def test_policy_only_excludes_everything_but_journal_articles():
    out = apply_source_policy(RANKED, STATUSES, {r: True for r in RANKED}, "peer_reviewed_only")
    assert out.ordered_ids == ["j1", "j2"]
    assert {e["record_id"]: e["source_status"] for e in out.excluded} == {
        "pre": PREPRINT, "conf": CONFERENCE, "book": BOOK, "unk": UNKNOWN}
    with pytest.raises(ValueError):
        apply_source_policy(RANKED, STATUSES, {}, "bogus")


MIXED_PUBMED = {"701": {"title": "Shear stress platelet activation phosphatidylserine", "pubtype": ["Preprint"],
                        "journal": "bioRxiv"},
                "702": {"title": "Platelet activation under shear stress", "pubtype": ["Journal Article"]}}
MIXED_CROSSREF = {"10.5555/conf": {"title": "Shear-mediated platelet activation lipids", "type": "proceedings-article"},
                  "10.5555/book": {"title": "Platelet lipid biology", "type": "book-chapter"}}


def _policy_run(tmp_path, settings, policy, max_selected):
    return run(tmp_path / policy, api(MIXED_PUBMED, MIXED_CROSSREF), settings, max_results=4,
               max_selected=max_selected, abstract_fetcher=FakeAbstractFetcher(), source_policy=policy)


def _report(summary, records, verification):
    report, validation = build_report(question=Q, question_definition=None, search_summary=summary,
                                      source_texts={"sources": []}, evidence={"accepted": []}, gaps=None,
                                      hypotheses=None, narrative=None, records=records, verification=verification,
                                      stage_notes=[])
    return report


@pytest.mark.parametrize("policy", ["allow_all", "peer_reviewed_preferred", "peer_reviewed_only"])
def test_pipeline_source_policies(tmp_path, settings, policy):
    result = _policy_run(tmp_path, settings, policy, 4)
    s = result.summary
    statuses = s["source_classification"]
    selected = s["selection"]["selected_record_ids"]
    sel_status = [statuses[r] for r in selected]
    assert s["source_policy"]["policy"] == policy == s["parameters"]["source_policy"]
    sources = json.loads((result.run_dir / "sources.json").read_text())
    assert all(src["source_status"] in (JOURNAL_ARTICLE, PREPRINT, CONFERENCE, BOOK, UNKNOWN)
               for src in (sources if isinstance(sources, list) else sources.get("records", [])))
    if policy == "peer_reviewed_only":
        assert sel_status == [JOURNAL_ARTICLE]
        assert s["source_policy"]["excluded_count"] == 3
        assert s["source_policy"]["fewer_than_requested"] and "peer_reviewed_only" in s["source_policy"]["shortfall_note"]
        log = log_of(result)["source_policy"]
        assert {e["source_status"] for e in log["excluded_by_policy"]} == {PREPRINT, CONFERENCE, BOOK}
        report = _report(s, result.records, result.verification)
        assert "Fewer sources than requested" in report
        assert "Note: Only 1 of 4" in format_summary(s, "runs/x")
    else:
        assert sorted(sel_status) == sorted([JOURNAL_ARTICLE, PREPRINT, CONFERENCE, BOOK])
        assert s["source_policy"]["preprints_selected"] == 1 and not s["source_policy"]["fewer_than_requested"]
        report = _report(s, result.records, result.verification)
        assert "Preprints:** 1 selected" in report and 'labelled "Preprint — not peer-reviewed"' in report
    if policy == "peer_reviewed_preferred":
        assert sel_status[0] == JOURNAL_ARTICLE


def test_preferred_policy_picks_journal_article_when_only_one_slot(tmp_path, settings):
    pref = _policy_run(tmp_path, settings, "peer_reviewed_preferred", 1)
    assert [pref.summary["source_classification"][r] for r in pref.summary["selection"]["selected_record_ids"]] \
        == [JOURNAL_ARTICLE]
