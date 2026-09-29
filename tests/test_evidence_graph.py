"""v0.4 Evidence-to-Hypothesis Graph: deterministic build, validation, DOT rendering, UI (offline)."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from conftest import FAKE_XAI_KEY, model_env
from m3_support import QUOTE_1, critic_review, evidence_items, full_script, gap, hypothesis, load, records, run
from sciforge.config import ModelSettings
from sciforge.evidence_graph import (
    EDGE_TYPES,
    GRAPH_FILE,
    GRAPH_VERSION,
    PREPRINT_LABEL,
    GraphInputs,
    build_evidence_graph,
    build_validated_graph,
    dot_escape,
    graph_json,
    graph_to_dot,
    node_details,
    node_id,
    node_options,
    validate_evidence_graph,
)
from sciforge.hypothesis_validation import HYPOTHESIS_NOTICE


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SCIFORGE_DEBUG_KEEP_REJECTED_RAW", raising=False)
    monkeypatch.delenv("SCIFORGE_MAX_SOURCE_CHARS", raising=False)


def pipeline(tmp_path, settings, script=None, **kw):
    recs, ver = records()
    result, _ = run(tmp_path, settings, script or full_script(recs), recs=recs, ver=ver, **kw)
    return result, recs, ver


def inputs_from(result, recs, ver, **overrides) -> GraphInputs:
    base = dict(records=recs, verification=ver,
                source_texts=load(result, "source_texts")["sources"],
                accepted_evidence=load(result, "evidence")["accepted"],
                gaps=load(result, "gaps")["accepted"], hypotheses=load(result, "hypotheses")["accepted"],
                source_types={}, report_citations=result.report_validation["citations"])
    base.update(overrides)
    return GraphInputs(**base)


def codes(errors):
    return sorted({e["code"] for e in errors})


@pytest.fixture
def built(tmp_path, settings):
    result, recs, ver = pipeline(tmp_path, settings)
    inputs = inputs_from(result, recs, ver)
    return result, recs, ver, inputs, build_validated_graph(inputs)


def nodes_of(graph, t):
    return [n for n in graph["nodes"] if n["type"] == t]


# ================================================================== ids + reproducibility


def test_graph_written_by_pipeline_and_validated(built):
    result, recs, _, _, graph = built
    on_disk = json.loads((result.run_dir / GRAPH_FILE).read_text(encoding="utf-8"))
    assert on_disk == graph and graph["graph_version"] == GRAPH_VERSION
    assert graph["validation"] == {**graph["validation"], "status": "passed", "errors": []}
    assert load(result, "report_validation")["evidence_graph"]["status"] == "passed"
    assert graph["references"] == {"source_record_ids": sorted(r.record_id for r in recs),
                                   "evidence_ids": ["ev_0001", "ev_0002"], "gap_ids": ["gap_01"],
                                   "hypothesis_ids": ["hyp_01"]}
    assert "generated_at" not in (result.run_dir / GRAPH_FILE).read_text(encoding="utf-8")


def test_deterministic_node_and_edge_ids(built):
    _, recs, _, _, graph = built
    ids = [n["id"] for n in graph["nodes"]]
    assert ids == sorted(ids)
    assert set(ids) == {*(f"src:{r.record_id}" for r in recs), "ev:ev_0001", "ev:ev_0002", "gap:gap_01",
                        "hyp:hyp_01", "pred:hyp_01", "fals:hyp_01"}
    assert node_id("prediction", "hyp_07") == "pred:hyp_07" and node_id("source", "rec_x") == "src:rec_x"
    assert {e["id"] for e in graph["edges"]} == {
        f"supports|src:{recs[0].record_id}|ev:ev_0001", f"supports|src:{recs[1].record_id}|ev:ev_0002",
        "supports|ev:ev_0001|hyp:hyp_01", "supports|ev:ev_0002|hyp:hyp_01", "motivates|ev:ev_0001|gap:gap_01",
        "motivates|ev:ev_0002|gap:gap_01", "addresses|hyp:hyp_01|gap:gap_01", "tests|pred:hyp_01|hyp:hyp_01",
        "tests|fals:hyp_01|hyp:hyp_01"}
    assert {e["type"] for e in graph["edges"]} <= set(EDGE_TYPES)


def test_reproducible_byte_identical(built, tmp_path, settings):
    result, recs, ver, inputs, graph = built
    assert graph_json(build_validated_graph(inputs)) == graph_json(graph)
    shuffled = inputs_from(result, list(reversed(recs)), list(reversed(ver)),
                           accepted_evidence=list(reversed(inputs.accepted_evidence)),
                           source_texts=list(reversed(inputs.source_texts)))
    assert graph_json(build_validated_graph(shuffled)) == graph_json(graph)
    again, _, _ = pipeline(tmp_path / "second", settings)
    assert (again.run_dir / GRAPH_FILE).read_bytes() == (result.run_dir / GRAPH_FILE).read_bytes()


# ================================================================== provenance


def test_source_provenance_from_verified_record_only(built):
    _, recs, _, inputs, graph = built
    src = {n["record_id"]: n for n in nodes_of(graph, "source")}
    r = recs[1]
    n = src[r.record_id]
    assert (n["title"], n["authors"], n["year"], n["doi"], n["pmid"], n["journal"]) == \
        (r.title, r.authors, r.year, r.doi, r.pmid, r.journal)
    assert n["verification_status"] == "verified" and n["abstract_only"] is True
    assert n["source_type"] == "unknown" and n["preprint_label"] is None
    bad = copy.deepcopy(graph)
    [s for s in bad["nodes"] if s["id"] == f"src:{r.record_id}"][0]["doi"] = "10.9999/invented"
    assert "bibliographic_field_mismatch" in codes(validate_evidence_graph(bad, inputs))


def test_evidence_from_non_eligible_source_excluded(built):
    result, recs, ver, inputs, _ = built
    st = [s for s in inputs.source_texts if s["record_id"] != recs[1].record_id]
    graph = build_validated_graph(inputs_from(result, recs, ver, source_texts=st, report_citations=None))
    assert {"item": "ev:ev_0002", "reason": "source_not_verified_record"} in graph["excluded"]
    assert f"src:{recs[1].record_id}" not in {n["id"] for n in graph["nodes"]}
    assert graph["validation"]["status"] == "passed_with_exclusions"
    assert "hyp:hyp_01" not in {n["id"] for n in graph["nodes"]}          # cited excluded evidence -> dropped


def test_evidence_provenance_and_exact_quote(built):
    _, recs, _, inputs, graph = built
    ev = {n["evidence_id"]: n for n in nodes_of(graph, "evidence")}
    assert ev["ev_0001"]["verified_quote"] == QUOTE_1 and ev["ev_0001"]["source_record_id"] == recs[0].record_id
    bad = copy.deepcopy(graph)
    [n for n in bad["nodes"] if n["id"] == "ev:ev_0001"][0]["verified_quote"] = "not the verified quote"
    assert "evidence_mismatch" in codes(validate_evidence_graph(bad, inputs))
    bad = copy.deepcopy(graph)
    e = [e for e in bad["edges"] if e["id"] == f"supports|src:{recs[0].record_id}|ev:ev_0001"][0]
    e["source"] = f"src:{recs[1].record_id}"
    e["id"] = f"supports|src:{recs[1].record_id}|ev:ev_0001"
    assert {"wrong_source_link", "evidence_source_link"} <= set(codes(validate_evidence_graph(bad, inputs)))


def test_gap_provenance(built):
    result, recs, ver, inputs, graph = built
    [g] = nodes_of(graph, "gap")
    assert g["gap_id"] == "gap_01" and g["supporting_evidence_ids"] == ["ev_0001", "ev_0002"]
    bad = copy.deepcopy(graph)
    bad["nodes"].append({**g, "id": "gap:gap_09", "gap_id": "gap_09"})
    assert "unresolved_gap" in codes(validate_evidence_graph(bad, inputs))
    dangling = inputs_from(result, recs, ver, gaps=[gap(supporting_evidence_ids=["ev_0001", "ev_0404"])],
                           report_citations=None)
    graph2 = build_validated_graph(dangling)
    assert {"item": "gap:gap_01", "reason": "gap_references_missing_evidence"} in graph2["excluded"]
    assert {"item": "hyp:hyp_01", "reason": "hypothesis_references_missing_gap"} in graph2["excluded"]


def test_hypothesis_provenance_and_rejected_hypotheses_excluded(tmp_path, settings):
    recs, _ = records()
    script = full_script(recs, hyps={"hypotheses": [hypothesis(), hypothesis(evidence_ids=["ev_0404"])]},
                         critic={"reviews": [critic_review("hyp_01")]})
    result, recs, ver = pipeline(tmp_path, settings, script)
    graph = result.evidence_graph
    assert load(result, "hypotheses")["counts"]["rejected"] == 1
    [h] = nodes_of(graph, "hypothesis")
    assert h["id"] == "hyp:hyp_01" and h["warning"] == HYPOTHESIS_NOTICE
    assert h["warning"] == ("Unvalidated, AI-generated hypothesis for further investigation — not a validated "
                            "discovery or scientific finding.")
    for key in ("confidence", "mechanistic_claim_level", "evidence_ids", "research_gap_id",
                "alternative_explanation", "critic_status", "revision_status"):
        assert h[key] not in (None, "", [])
    assert h["critic_status"] == "completed" and h["revision_status"] == "no revision required"
    inputs = inputs_from(result, recs, ver)
    bad = copy.deepcopy(graph)
    [n for n in bad["nodes"] if n["id"] == "hyp:hyp_01"][0]["warning"] = "Validated discovery."
    assert "hypothesis_warning" in codes(validate_evidence_graph(bad, inputs))
    bad = copy.deepcopy(graph)
    bad["nodes"].append({**h, "id": "hyp:hyp_02", "hypothesis_id": "hyp_02"})
    assert "unresolved_hypothesis" in codes(validate_evidence_graph(bad, inputs))


def test_prediction_and_falsification_linkage(built):
    result, recs, ver, inputs, graph = built
    pred = [n for n in graph["nodes"] if n["id"] == "pred:hyp_01"][0]
    fals = [n for n in graph["nodes"] if n["id"] == "fals:hyp_01"][0]
    assert pred["parent_hypothesis_id"] == "hyp_01" and pred["text"] == inputs.hypotheses[0]["prediction"]
    assert fals["weakening_result"] == inputs.hypotheses[0]["falsification_test"]["weakening_result"]
    bad = copy.deepcopy(graph)
    bad["edges"] = [e for e in bad["edges"] if e["id"] != "tests|fals:hyp_01|hyp:hyp_01"]
    assert "hypothesis_test_links" in codes(validate_evidence_graph(bad, inputs))
    bad = copy.deepcopy(graph)
    [n for n in bad["nodes"] if n["id"] == "pred:hyp_01"][0]["parent_hypothesis_id"] = "hyp_02"
    assert "orphan_test_node" in codes(validate_evidence_graph(bad, inputs))
    no_pred = [dict(inputs.hypotheses[0], prediction="")]
    g2 = build_validated_graph(inputs_from(result, recs, ver, hypotheses=no_pred, report_citations=None))
    assert {"item": "hyp:hyp_01", "reason": "hypothesis_missing_prediction_or_falsification"} in g2["excluded"]


# ================================================================== contradictions


def test_contradiction_only_where_gap_records_it(built):
    result, recs, ver, inputs, graph = built
    assert [e for e in graph["edges"] if e["type"] == "contradicts"] == []
    ev = [dict(e) for e in inputs.accepted_evidence]
    ev[1]["evidence_category"] = "conflicting"                      # flagged conflicting, but no pair recorded
    g1 = build_validated_graph(inputs_from(result, recs, ver, accepted_evidence=ev))
    assert [e for e in g1["edges"] if e["type"] == "contradicts"] == []
    assert [n for n in g1["nodes"] if n["id"] == "ev:ev_0002"][0]["conflict_note"].startswith("category")
    gaps = [gap(supporting_evidence_ids=["ev_0001"], conflicting_evidence_ids=["ev_0002"])]
    g2 = build_validated_graph(inputs_from(result, recs, ver, gaps=gaps, hypotheses=[]))
    [c] = [e for e in g2["edges"] if e["type"] == "contradicts"]
    assert c == {"id": "contradicts|ev:ev_0001|ev:ev_0002", "type": "contradicts", "source": "ev:ev_0001",
                 "target": "ev:ev_0002", "bidirectional": True, "basis": "research_gap_record",
                 "recorded_by_gaps": ["gap_01"]}
    assert g2["validation"]["status"] == "passed"


def test_invented_contradiction_is_a_validation_error(built):
    _, _, _, inputs, graph = built
    bad = copy.deepcopy(graph)
    bad["edges"].append({"id": "contradicts|ev:ev_0001|ev:ev_0002", "type": "contradicts", "source": "ev:ev_0001",
                         "target": "ev:ev_0002", "bidirectional": True, "basis": "research_gap_record",
                         "recorded_by_gaps": ["gap_01"]})
    assert "invented_contradiction" in codes(validate_evidence_graph(bad, inputs))


# ================================================================== rejected evidence / preprints / report


def test_rejected_evidence_never_appears(tmp_path, settings):
    recs, _ = records()
    ev = evidence_items(recs)
    ev[0]["items"].append({**ev[0]["items"][0], "quote": "REJECTED QUOTE not present in the abstract",
                           "claim": "REJECTED CLAIM that must never reach the graph"})
    script = full_script(recs)
    script[1:3] = ev
    result, recs, ver = pipeline(tmp_path, settings, script)
    assert load(result, "evidence")["counts"]["rejected"] >= 1
    text = (result.run_dir / GRAPH_FILE).read_text(encoding="utf-8")
    assert "REJECTED" not in text and result.evidence_graph["validation"]["status"] == "passed"
    inputs = inputs_from(result, recs, ver, rejected_evidence_texts=["REJECTED CLAIM that must never reach the graph"])
    bad = copy.deepcopy(result.evidence_graph)
    [n for n in bad["nodes"] if n["id"] == "ev:ev_0001"][0]["finding"] = "REJECTED CLAIM that must never reach the graph"
    assert "rejected_evidence_present" in codes(validate_evidence_graph(bad, inputs))
    bad["nodes"].append({**bad["nodes"][0], "id": "ev:ev_0099", "type": "evidence", "evidence_id": "ev_0099"})
    assert "unresolved_evidence" in codes(validate_evidence_graph(bad, inputs))


def test_preprint_labelling_and_source_type_preserved(built):
    result, recs, ver, _, _ = built
    types = {recs[0].record_id: "preprint", recs[1].record_id: "peer-reviewed journal article"}
    inputs = inputs_from(result, recs, ver, source_types=types)
    graph = build_validated_graph(inputs)
    src = {n["record_id"]: n for n in nodes_of(graph, "source")}
    assert src[recs[0].record_id]["preprint_label"] == PREPRINT_LABEL == "Preprint — not peer-reviewed"
    assert src[recs[0].record_id]["source_type_label"] == PREPRINT_LABEL
    assert src[recs[1].record_id]["preprint_label"] is None
    assert src[recs[1].record_id]["source_type"] == "peer-reviewed journal article"
    bad = copy.deepcopy(graph)
    [n for n in bad["nodes"] if n["id"] == f"src:{recs[0].record_id}"][0]["source_type"] = \
        "peer-reviewed journal article"
    assert {"source_type_mismatch"} <= set(codes(validate_evidence_graph(bad, inputs)))
    bad = copy.deepcopy(graph)
    [n for n in bad["nodes"] if n["id"] == f"src:{recs[0].record_id}"][0]["preprint_label"] = None
    assert "preprint_label" in codes(validate_evidence_graph(bad, inputs))


def test_pipeline_uses_search_summary_classification(tmp_path, settings):
    from m3_support import SEARCH_SUMMARY

    recs, _ = records()
    summary = {**SEARCH_SUMMARY, "source_classification": {recs[1].record_id: "preprint"}}
    result, _, _ = pipeline(tmp_path, settings, search_summary=summary)
    src = {n["record_id"]: n for n in nodes_of(result.evidence_graph, "source")}
    assert src[recs[1].record_id]["preprint_label"] == PREPRINT_LABEL
    assert result.evidence_graph["validation"]["status"] == "passed"


def test_report_and_graph_cite_same_verified_records(built):
    result, recs, ver, inputs, graph = built
    report_ids = {c["record_id"] for c in result.report_validation["citations"] if c["status"] == "resolved"}
    assert report_ids == set(graph["references"]["source_record_ids"])
    extra = [*inputs.report_citations, {"ref": "S9", "record_id": "rec_" + "f" * 16, "status": "resolved"}]
    errs = validate_evidence_graph(graph, inputs_from(result, recs, ver, report_citations=extra))
    assert "report_graph_citation_mismatch" in codes(errs)


# ================================================================== missing nodes / adversarial


def test_dangling_references_are_errors_not_crashes(built):
    result, recs, ver, inputs, graph = built
    bad = copy.deepcopy(graph)
    bad["nodes"] = [n for n in bad["nodes"] if n["id"] != "gap:gap_01"]
    errs = validate_evidence_graph(bad, inputs)
    assert {"dangling_edge", "hypothesis_gap_link"} <= set(codes(errs))
    bad["edges"].append({"id": "motivates|ev:ev_0404|gap:gap_01", "type": "motivates", "source": "ev:ev_0404",
                         "target": "gap:gap_01"})
    assert "dangling_edge" in codes(validate_evidence_graph(bad, inputs))
    bad["edges"].append({"id": "x", "type": "causes", "source": "ev:ev_0001", "target": "ev:ev_0002"})
    assert "invalid_edge_type" in codes(validate_evidence_graph(bad, inputs))
    wrong_dir = copy.deepcopy(graph)
    wrong_dir["edges"].append({"id": "addresses|gap:gap_01|hyp:hyp_01", "type": "addresses", "source": "gap:gap_01",
                               "target": "hyp:hyp_01"})
    assert "invalid_edge_direction" in codes(validate_evidence_graph(wrong_dir, inputs))
    missing_ev = [dict(inputs.hypotheses[0], evidence_ids=["ev_0001", "ev_0404"])]
    g = build_validated_graph(inputs_from(result, recs, ver, hypotheses=missing_ev))
    assert {"item": "hyp:hyp_01", "reason": "hypothesis_references_missing_evidence"} in g["excluded"]
    assert validate_evidence_graph({"nodes": [{"id": 5}], "edges": [{}]}, inputs)       # no exception


@pytest.mark.parametrize("field,value", [
    ("hypothesis", "GPIb may matter, see doi 10.9999/fake.2024.001."),
    ("rationale", "Reported by Smith et al. in a flow study."),
    ("rationale", "As in Marchetti-Oyelaran (1993)."),
    ("prediction", "Lower levels, see https://example.org/x."),
])
def test_fake_citation_in_hypothesis_is_rejected(built, field, value):
    result, recs, ver, inputs, graph = built
    injected = [dict(inputs.hypotheses[0], **{field: value})]
    g = build_validated_graph(inputs_from(result, recs, ver, hypotheses=injected))
    [ex] = [x for x in g["excluded"] if x["item"] == "hyp:hyp_01"]
    assert ex["reason"] == "bibliographic_text_in_model_text"
    assert value not in graph_json(g)
    assert not [n for n in g["nodes"] if n["type"] in ("hypothesis", "prediction", "falsification_test")]
    # a tampered graph that carries the fake citation fails validation
    bad = copy.deepcopy(graph)
    target = "pred:hyp_01" if field == "prediction" else "hyp:hyp_01"
    [n for n in bad["nodes"] if n["id"] == target][0]["text" if field == "prediction" else field] = value
    assert "fabricated_citation" in codes(validate_evidence_graph(bad, inputs))


# ================================================================== secrets / reasoning


def test_no_secrets_or_reasoning_in_graph(tmp_path, settings):
    ms = ModelSettings.from_env(model_env(SCIFORGE_MODEL_MAX_ATTEMPTS="30"))
    result, _, _ = pipeline(tmp_path, settings, model_settings=ms, tracker=None)
    text = (result.run_dir / GRAPH_FILE).read_text(encoding="utf-8")
    for needle in (FAKE_XAI_KEY, settings.ncbi_api_key, settings.contact_email):
        assert needle not in text
    for key in ('"reasoning"', '"chain_of_thought"', '"thinking"', '"encrypted_content"', '"critic_findings"',
                '"explanation": "Looks'):
        assert key not in text


# ================================================================== DOT + navigation


def test_dot_escaping_prevents_injection():
    evil = 'A "quoted" \\ label\nwith newline"]; evil -> x [label="pwn'
    esc = dot_escape(evil)
    assert "\n" not in esc and '\\"' in esc and '\\\\' in esc
    assert re.search(r'(?<!\\)"', esc) is None                          # every quote is escaped
    graph = {"nodes": [{"id": 'ev:ev_0001', "type": "evidence", "evidence_id": "ev_0001", "claim": evil}],
             "edges": []}
    dot = graph_to_dot(graph)
    line = [ln for ln in dot.splitlines() if ln.strip().startswith('"ev:ev_0001"')][0]
    body = line.split("[label=", 1)[1]
    label = re.match(r'"((?:[^"\\]|\\.)*)"', body).group(1)
    assert "evil" in label                                              # stays inside the label string
    assert dot.count("{") == dot.count("}")
    assert "\x00" not in dot_escape("a\x00b") and dot_escape(None) == ""


def test_dot_has_styles_and_legend(built):
    graph = built[4]
    dot = graph_to_dot(graph)
    assert dot.startswith("digraph evidence_graph {") and "cluster_legend" in dot
    for t in EDGE_TYPES:
        assert f'label="{t}"' in dot
    assert 'color="#D93025", style=dashed, dir=both' in dot            # contradicts style
    assert graph_to_dot(graph) == dot


def test_node_options_and_details_for_every_type(tmp_path, settings):
    from sciforge import app_service as svc
    from sciforge.demo_data import DEMO_QUESTION

    r = svc.run_web_investigation(svc.InvestigationRequest(question=DEMO_QUESTION), environ={})
    g = r.evidence_graph
    opts = node_options(g)
    assert [nid.split(":")[0] for nid, _ in opts] == sorted(
        [nid.split(":")[0] for nid, _ in opts], key=["src", "ev", "gap", "hyp", "pred", "fals"].index)
    src = node_details(g, [n for n, _ in opts if n.startswith("src:")][0])
    assert src["fields"]["Title"].startswith("[SYNTHETIC DEMO]") and src["related"]["Evidence from this source"]
    ev = node_details(g, "ev:ev_0001")
    assert ev["fields"]["Exact verified quote"].startswith("Preregistered studies reported")
    assert ev["related"]["Supporting source"] and ev["related"]["Research gaps"]
    assert ev["related"]["Contradicts (recorded by a gap)"] == ["ev:ev_0003"]
    gp = node_details(g, "gap:gap_02")
    assert gp["related"]["Hypotheses addressing it"] == ["hyp:hyp_02"]
    hp = node_details(g, "hyp:hyp_02")
    assert hp["fields"]["Warning"] == HYPOTHESIS_NOTICE and hp["fields"]["Revision"] == "revised"
    assert hp["fields"]["Critic"] == "completed" and "basis: inference" in hp["fields"]["Alternative explanation"]
    assert "weakening result" in hp["fields"]["Falsification test"] and hp["fields"]["Prediction"]
    assert hp["related"]["Tested by"] == ["fals:hyp_02", "pred:hyp_02"]
    assert node_details(g, "pred:hyp_01")["related"]["Tests hypothesis"] == ["hyp:hyp_01"]
    assert node_details(g, "fals:hyp_01")["fields"]["Parent hypothesis warning"] == HYPOTHESIS_NOTICE
    assert node_details(g, "nope") is None


def test_demo_graph_meaningful_and_valid():
    from sciforge import app_service as svc
    from sciforge.demo_data import DEMO_QUESTION

    r = svc.run_web_investigation(svc.InvestigationRequest(question=DEMO_QUESTION), environ={})
    g = r.evidence_graph
    assert g["validation"]["status"] == "passed" and r.validation["evidence_graph"]["status"] == "passed"
    assert g["counts"] == {**g["counts"], "source": 3, "evidence": 3, "gap": 2, "hypothesis": 2, "prediction": 2,
                           "falsification_test": 2, "edges_contradicts": 1}
    assert "DEMO-REJECTED" not in json.dumps(g)
    [c] = [e for e in g["edges"] if e["type"] == "contradicts"]
    assert c["recorded_by_gaps"] == ["gap_01"]


def test_ui_evidence_graph_tab_renders_and_navigates():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py"), default_timeout=60)
    at.run()
    at.button(key="investigate").click().run()
    assert not at.exception
    assert "Evidence Graph" in [t.label for t in at.tabs]
    charts = at.get("graphviz_chart")
    assert charts and "digraph evidence_graph" in charts[0].proto.spec
    sel = at.selectbox(key="graph_node")
    assert any("hyp:hyp_02" in str(o) for o in sel.options)
    sel.set_value("hyp:hyp_02").run()
    assert not at.exception
    text = "\n".join(df.value.to_csv() for df in at.dataframe)
    assert HYPOTHESIS_NOTICE in text and "revised" in text
    assert any("Legend" in m.value for m in at.markdown)
