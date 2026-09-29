"""v0.4 Evidence-to-Hypothesis Graph (deterministic; no model calls).

Built ONLY from structured, already-validated pipeline outputs: eligible verified source records (v0.2
records + ``source_texts.json`` entries), accepted evidence, accepted research gaps and accepted hypotheses
(v0.4 engine records). Rejected evidence / gaps / hypotheses never enter the graph.

Node ids (stable, deterministic)::

    src:<record_id>   ev:<evidence_id>   gap:<gap_id>   hyp:<hypothesis_id>   pred:<hypothesis_id>   fals:<hypothesis_id>

Edge types and directions (exactly five types)::

    supports     src -> ev    evidence item was extracted (exact verified quote) from this verified source
    supports     ev  -> hyp   accepted hypothesis cites this accepted evidence item
    motivates    ev  -> gap   evidence listed by the accepted gap (role "supporting" or "conflicting")
    addresses    hyp -> gap   hypothesis addresses its research gap
    tests        pred -> hyp  measurable prediction tests its parent hypothesis
    tests        fals -> hyp  falsification test tests its parent hypothesis
    contradicts  ev <-> ev    ONLY where an accepted gap explicitly records the conflict: an id in the gap's
                              ``conflicting_evidence_ids`` versus an id in the same gap's
                              ``supporting_evidence_ids`` (undirected; stored once with source < target)

Bibliographic fields appear only on source nodes and are copied from the v0.2 record. All model-derived node
text is re-checked with the existing identifier / citation detectors; an offending gap or hypothesis (and
everything that depends on it) is dropped and recorded in ``excluded`` with the reason.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from sciforge.hypothesis_validation import HYPOTHESIS_LABEL, HYPOTHESIS_NOTICE, has_bibliographic_text
from sciforge.models import Record, VerificationResult

GRAPH_VERSION = "v0.4-evidence-graph-1"
GRAPH_FILE = "evidence_graph.json"
PREPRINT_LABEL = "Preprint — not peer-reviewed"
SOURCE_TYPES = ("peer-reviewed journal article", "preprint", "conference paper", "book/chapter", "unknown")
NODE_TYPES = ("source", "evidence", "gap", "hypothesis", "prediction", "falsification_test")
NODE_PREFIX = {"source": "src", "evidence": "ev", "gap": "gap", "hypothesis": "hyp", "prediction": "pred",
               "falsification_test": "fals"}
EDGE_TYPES = ("supports", "contradicts", "motivates", "addresses", "tests")
# allowed (source node type, target node type) per edge type
EDGE_RULES: dict[str, frozenset[tuple[str, str]]] = {
    "supports": frozenset({("source", "evidence"), ("evidence", "hypothesis")}),
    "contradicts": frozenset({("evidence", "evidence")}),
    "motivates": frozenset({("evidence", "gap")}),
    "addresses": frozenset({("hypothesis", "gap")}),
    "tests": frozenset({("prediction", "hypothesis"), ("falsification_test", "hypothesis")}),
}
BIB_FIELDS = ("title", "authors", "year", "journal", "doi", "pmid", "source_url")
_ID_RE = {
    "source": re.compile(r"^src:rec_[0-9a-f]{16}$"), "evidence": re.compile(r"^ev:ev_\d{4}$"),
    "gap": re.compile(r"^gap:gap_\d{2}$"), "hypothesis": re.compile(r"^hyp:hyp_\d{2}$"),
    "prediction": re.compile(r"^pred:hyp_\d{2}$"), "falsification_test": re.compile(r"^fals:hyp_\d{2}$"),
}
# model-derived text fields per node type (re-checked for bibliographic identity)
MODEL_TEXT_FIELDS = {
    "evidence": ("claim", "finding"),
    "gap": ("gap_statement", "why_unresolved"),
    "hypothesis": ("hypothesis", "rationale", "assumptions", "evidence_limitations", "alternative_explanation"),
    "prediction": ("text",),
    "falsification_test": ("manipulated_or_compared", "measured", "weakening_result", "supporting_result"),
}


def node_id(node_type: str, item_id: str) -> str:
    return f"{NODE_PREFIX[node_type]}:{item_id}"


def edge_id(edge_type: str, source: str, target: str) -> str:
    return f"{edge_type}|{source}|{target}"


def source_type_badge(status: str | None) -> str:
    from sciforge.stages.report import source_type_badge as badge

    return badge(status)


@dataclass
class GraphInputs:
    """Structured, already-validated pipeline outputs (no model calls, no network)."""

    records: Sequence[Record]
    verification: Sequence[VerificationResult]
    source_texts: Sequence[Mapping[str, Any]]           # source_texts.json "sources" entries
    accepted_evidence: Sequence[Mapping[str, Any]]
    gaps: Sequence[Mapping[str, Any]] = ()
    hypotheses: Sequence[Mapping[str, Any]] = ()
    source_types: Mapping[str, str] = field(default_factory=dict)
    report_citations: Sequence[Mapping[str, Any]] | None = None
    rejected_evidence_texts: Sequence[str] = ()          # raw text of rejected evidence (leak check only)


def _texts(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [t for v in value.values() for t in _texts(v)]
    if isinstance(value, (list, tuple)):
        return [t for v in value for t in _texts(v)]
    return []


def model_text_problems(node: Mapping[str, Any]) -> list[str]:
    """Fields of a node whose model-derived text contains identifier / citation-like text."""
    return [f for f in MODEL_TEXT_FIELDS.get(node.get("type"), ())
            if any(has_bibliographic_text(t) for t in _texts(node.get(f)))]


def _eligible(inputs: GraphInputs) -> dict[str, Mapping[str, Any]]:
    return {s["record_id"]: s for s in inputs.source_texts
            if isinstance(s, Mapping) and s.get("status", "ok") == "ok" and isinstance(s.get("record_id"), str)}


# ------------------------------------------------------------------ build


def build_evidence_graph(inputs: GraphInputs) -> dict[str, Any]:
    """Deterministic graph dict (no timestamps). Call :func:`graph_json` for the byte-stable file form."""
    records = {r.record_id: r for r in inputs.records}
    verification = {v.record_id: v for v in inputs.verification}
    eligible = _eligible(inputs)
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[str, dict[str, Any]] = {}
    excluded: list[dict[str, Any]] = []

    def add_edge(etype: str, source: str, target: str, **attrs: Any) -> None:
        eid = edge_id(etype, source, target)
        edge = edges.setdefault(eid, {"id": eid, "type": etype, "source": source, "target": target, **attrs})
        for k, v in attrs.items():
            if isinstance(v, list):
                edge[k] = sorted(set(edge.get(k, [])) | set(v))

    # ---- evidence (+ their sources)
    evidence: dict[str, Mapping[str, Any]] = {}
    for ev in inputs.accepted_evidence:
        eid, rid = ev.get("evidence_id"), ev.get("source_record_id")
        if not isinstance(eid, str) or not re.match(r"^ev_\d{4}$", eid):
            excluded.append({"item": "evidence", "reason": "invalid_evidence_id"})
            continue
        if rid not in records or rid not in eligible:
            excluded.append({"item": node_id("evidence", eid), "reason": "source_not_verified_record"})
            continue
        node = {"id": node_id("evidence", eid), "type": "evidence", "evidence_id": eid, "source_record_id": rid,
                "claim": ev.get("claim"), "finding": ev.get("finding"), "verified_quote": ev.get("quote"),
                "evidence_category": ev.get("evidence_category"), "confidence": ev.get("confidence"),
                "access_level": ev.get("access_level"), "abstract_only": bool(ev.get("abstract_only", True)),
                "conflict_note": ("category 'conflicting' (pairwise conflicts only where a research gap records "
                                  "them)" if ev.get("evidence_category") == "conflicting" else None),
                "provenance": {"kind": "accepted_evidence", "evidence_id": eid, "record_id": rid}}
        bad = model_text_problems(node)
        if bad:
            excluded.append({"item": node["id"], "reason": "bibliographic_text_in_model_text", "fields": bad})
            continue
        nodes[node["id"]] = node
        evidence[eid] = ev

    # ---- source nodes: sources of accepted evidence + (if given) report-cited verified records
    source_ids = {ev["source_record_id"] for ev in evidence.values()}
    for c in inputs.report_citations or ():
        rid = c.get("record_id")
        if c.get("status") == "resolved" and isinstance(rid, str) and rid in records and rid in eligible:
            source_ids.add(rid)
    for rid in sorted(source_ids):
        rec, st = records[rid], eligible[rid]
        stype = inputs.source_types.get(rid) or "unknown"
        stype = stype if stype in SOURCE_TYPES else "unknown"
        node = {"id": node_id("source", rid), "type": "source", "record_id": rid,
                **{f: (list(getattr(rec, f)) if f == "authors" else getattr(rec, f)) for f in BIB_FIELDS},
                "verification_status": verification[rid].status if rid in verification else None,
                "source_type": stype, "source_type_label": source_type_badge(stype),
                "preprint_label": PREPRINT_LABEL if stype == "preprint" else None,
                "abstract_only": bool(st.get("abstract_only", True)), "access_level": st.get("access_level"),
                "provenance": {"kind": "verified_v02_record", "record_id": rid,
                               "bibliographic_fields": "copied from the v0.2 record only"}}
        nodes[node["id"]] = node
    for eid, ev in evidence.items():
        add_edge("supports", node_id("source", ev["source_record_id"]), node_id("evidence", eid))

    # ---- gaps
    gaps: dict[str, Mapping[str, Any]] = {}
    for g in inputs.gaps:
        gid = g.get("gap_id")
        if not isinstance(gid, str) or not re.match(r"^gap_\d{2}$", gid):
            excluded.append({"item": "gap", "reason": "invalid_gap_id"})
            continue
        sup = [e for e in g.get("supporting_evidence_ids") or [] if e in evidence]
        con = [e for e in g.get("conflicting_evidence_ids") or [] if e in evidence]
        missing = [e for e in [*(g.get("supporting_evidence_ids") or []), *(g.get("conflicting_evidence_ids") or [])]
                   if e not in evidence]
        node = {"id": node_id("gap", gid), "type": "gap", "gap_id": gid, "label": g.get("label", "inference"),
                "gap_statement": g.get("gap_statement"), "why_unresolved": g.get("why_unresolved"),
                "confidence": g.get("confidence"), "supporting_evidence_ids": sup, "conflicting_evidence_ids": con,
                "provenance": {"kind": "accepted_gap", "gap_id": gid}}
        if missing or not sup:
            excluded.append({"item": node["id"], "reason": "gap_references_missing_evidence"})
            continue
        bad = model_text_problems(node)
        if bad:
            excluded.append({"item": node["id"], "reason": "bibliographic_text_in_model_text", "fields": bad})
            continue
        nodes[node["id"]] = node
        gaps[gid] = g
        for e in sup:
            add_edge("motivates", node_id("evidence", e), node["id"], role="supporting")
        for e in con:
            add_edge("motivates", node_id("evidence", e), node["id"], role="conflicting")
        for s in sup:
            for c in con:
                if s == c:
                    continue
                a, b = sorted((node_id("evidence", s), node_id("evidence", c)))
                add_edge("contradicts", a, b, bidirectional=True, basis="research_gap_record",
                         recorded_by_gaps=[gid])

    # ---- hypotheses (+ prediction / falsification nodes)
    for h in inputs.hypotheses:
        hid = h.get("hypothesis_id")
        if not isinstance(hid, str) or not re.match(r"^hyp_\d{2}$", hid):
            excluded.append({"item": "hypothesis", "reason": "invalid_hypothesis_id"})
            continue
        hnode_id = node_id("hypothesis", hid)
        ev_ids = list(h.get("evidence_ids") or [])
        gid = h.get("research_gap_id")
        alt = h.get("alternative_explanation") or {}
        st = h.get("stress_test") or {}
        conf = h.get("confidence_detail") or {}
        node = {"id": hnode_id, "type": "hypothesis", "hypothesis_id": hid, "label": HYPOTHESIS_LABEL,
                "warning": HYPOTHESIS_NOTICE, "hypothesis": h.get("hypothesis"), "rationale": h.get("rationale"),
                "evidence_ids": ev_ids, "research_gap_id": gid,
                "mechanistic_claim_level": h.get("mechanistic_claim_level"),
                "supported_claim_level": h.get("supported_claim_level"),
                "causality_statement": h.get("causality_statement"),
                "confidence": h.get("confidence"),
                "confidence_detail": {k: conf.get(k) for k in ("proposed_by_model", "deterministic_ceiling",
                                                               "final", "reasons")},
                "alternative_explanation": {"explanation": alt.get("explanation"), "basis": alt.get("basis"),
                                            "evidence_ids": list(alt.get("evidence_ids") or [])},
                "assumptions": list(h.get("assumptions") or []),
                "evidence_limitations": list(h.get("evidence_limitations") or []),
                "source_quality_summary": h.get("source_quality_summary"),
                "critic_status": st.get("critic_status", "not_run"),
                "critic_verdicts": {k: (v or {}).get("verdict") for k, v in (st.get("critic_findings") or {}).items()},
                "revision_status": st.get("revision_status", "not_run"),
                "stress_test_note": st.get("note"),
                "provenance": {"kind": "accepted_hypothesis", "hypothesis_id": hid}}
        reason = None
        if not ev_ids or any(e not in evidence for e in ev_ids) or \
                any(e not in evidence for e in node["alternative_explanation"]["evidence_ids"]):
            reason = "hypothesis_references_missing_evidence"
        elif gid not in gaps:
            reason = "hypothesis_references_missing_gap"
        fals = h.get("falsification_test") or {}
        pred = {"id": node_id("prediction", hid), "type": "prediction", "parent_hypothesis_id": hid,
                "text": h.get("prediction"), "provenance": {"kind": "accepted_hypothesis_field",
                                                             "hypothesis_id": hid, "field": "prediction"}}
        fnode = {"id": node_id("falsification_test", hid), "type": "falsification_test", "parent_hypothesis_id": hid,
                 **{k: fals.get(k) for k in MODEL_TEXT_FIELDS["falsification_test"]},
                 "provenance": {"kind": "accepted_hypothesis_field", "hypothesis_id": hid,
                                "field": "falsification_test"}}
        if reason is None and (not isinstance(pred["text"], str) or not pred["text"].strip()
                               or not all(isinstance(fnode[k], str) and fnode[k].strip()
                                          for k in MODEL_TEXT_FIELDS["falsification_test"])):
            reason = "hypothesis_missing_prediction_or_falsification"
        bad = model_text_problems(node) + model_text_problems(pred) + model_text_problems(fnode)
        if reason is None and bad:
            excluded.append({"item": hnode_id, "reason": "bibliographic_text_in_model_text", "fields": bad})
            continue
        if reason is not None:
            excluded.append({"item": hnode_id, "reason": reason})
            continue
        nodes[hnode_id] = node
        nodes[pred["id"]] = pred
        nodes[fnode["id"]] = fnode
        for e in ev_ids:
            add_edge("supports", node_id("evidence", e), hnode_id)
        add_edge("addresses", hnode_id, node_id("gap", gid))
        add_edge("tests", pred["id"], hnode_id)
        add_edge("tests", fnode["id"], hnode_id)

    node_list = [nodes[k] for k in sorted(nodes)]
    edge_list = [edges[k] for k in sorted(edges)]
    content = {"nodes": node_list, "edges": edge_list,
               "excluded": sorted(excluded, key=lambda x: json.dumps(x, sort_keys=True))}
    refs = {t: sorted(n[k] for n in node_list if n["type"] == t) for t, k in
            (("source", "record_id"), ("evidence", "evidence_id"), ("gap", "gap_id"), ("hypothesis", "hypothesis_id"))}
    return {
        "graph_version": GRAPH_VERSION,
        "notice": ("Deterministic graph built by code from validated pipeline outputs (no model calls). Hypotheses "
                   f"are: {HYPOTHESIS_NOTICE}"),
        "id_scheme": {t: f"{p}:<id>" for t, p in NODE_PREFIX.items()},
        "edge_types": {t: sorted(f"{a}->{b}" for a, b in rules) for t, rules in EDGE_RULES.items()},
        "contradiction_policy": ("contradicts edges only where an accepted research gap explicitly records the "
                                 "conflict (conflicting_evidence_ids vs supporting_evidence_ids of the same gap); "
                                 "no conflict is inferred"),
        "references": {"source_record_ids": refs["source"], "evidence_ids": refs["evidence"],
                       "gap_ids": refs["gap"], "hypothesis_ids": refs["hypothesis"]},
        "counts": {**{t: sum(1 for n in node_list if n["type"] == t) for t in NODE_TYPES},
                   **{f"edges_{t}": sum(1 for e in edge_list if e["type"] == t) for t in EDGE_TYPES},
                   "excluded": len(excluded)},
        **content,
        "content_sha256": hashlib.sha256(_canonical(content).encode("utf-8")).hexdigest(),
    }


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def graph_json(graph: Mapping[str, Any]) -> str:
    """Byte-stable serialisation (sorted keys, fixed indentation, trailing newline)."""
    return json.dumps(graph, sort_keys=True, ensure_ascii=False, indent=2) + "\n"


# ------------------------------------------------------------------ validate


def validate_evidence_graph(graph: Mapping[str, Any], inputs: GraphInputs) -> list[dict[str, Any]]:
    """Deterministic checks; returns a list of errors (empty = valid)."""
    errors: list[dict[str, Any]] = []

    def err(code: str, detail: str, **extra: Any) -> None:
        errors.append({"code": code, "detail": detail, **extra})

    nodes = {n.get("id"): n for n in graph.get("nodes") or []}
    edges = list(graph.get("edges") or [])
    records = {r.record_id: r for r in inputs.records}
    eligible = _eligible(inputs)
    accepted_ev = {e.get("evidence_id"): e for e in inputs.accepted_evidence}
    gaps = {g.get("gap_id"): g for g in inputs.gaps}
    hyps = {h.get("hypothesis_id"): h for h in inputs.hypotheses}

    if graph.get("graph_version") != GRAPH_VERSION:
        err("graph_version", "unexpected graph version")
    # ids + references resolve
    for nid, n in nodes.items():
        t = n.get("type")
        if t not in NODE_TYPES or not isinstance(nid, str) or not _ID_RE[t].match(nid):
            err("invalid_node_id", "node id does not follow the deterministic scheme", node=str(nid))
            continue
        if t == "source":
            rid = n.get("record_id")
            if nid != node_id("source", str(rid)) or rid not in records or rid not in eligible:
                err("unresolved_source", "source node does not resolve to an eligible verified record", node=nid)
                continue
            rec = records[rid]
            for f in BIB_FIELDS:
                expected = list(getattr(rec, f)) if f == "authors" else getattr(rec, f)
                if n.get(f) != expected:
                    err("bibliographic_field_mismatch", "source bibliographic field differs from the v0.2 record",
                        node=nid, field=f)
            expected_type = inputs.source_types.get(rid) or "unknown"
            expected_type = expected_type if expected_type in SOURCE_TYPES else "unknown"
            if n.get("source_type") != expected_type:
                err("source_type_mismatch", "source type does not match the classification", node=nid)
            if (n.get("preprint_label") == PREPRINT_LABEL) != (expected_type == "preprint"):
                err("preprint_label", "preprint label missing or wrongly applied", node=nid)
        elif t == "evidence":
            eid = n.get("evidence_id")
            ev = accepted_ev.get(eid)
            if nid != node_id("evidence", str(eid)) or ev is None:
                err("unresolved_evidence", "evidence node is not an accepted evidence item (rejected evidence "
                                           "never appears)", node=nid)
                continue
            if n.get("verified_quote") != ev.get("quote") or n.get("source_record_id") != ev.get("source_record_id"):
                err("evidence_mismatch", "evidence node quote/source differs from the accepted record", node=nid)
        elif t == "gap":
            if n.get("gap_id") not in gaps or nid != node_id("gap", str(n.get("gap_id"))):
                err("unresolved_gap", "gap node is not an accepted gap", node=nid)
        elif t == "hypothesis":
            hid = n.get("hypothesis_id")
            if hid not in hyps or nid != node_id("hypothesis", str(hid)):
                err("unresolved_hypothesis", "hypothesis node is not an accepted hypothesis", node=nid)
            if n.get("warning") != HYPOTHESIS_NOTICE or n.get("label") != HYPOTHESIS_LABEL:
                err("hypothesis_warning", "hypothesis node lacks the fixed warning/label", node=nid)
        elif t in ("prediction", "falsification_test"):
            parent = n.get("parent_hypothesis_id")
            if nid != node_id(t, str(parent)) or node_id("hypothesis", str(parent)) not in nodes:
                err("orphan_test_node", "prediction/falsification node does not map to its hypothesis", node=nid)
        bad = model_text_problems(n)
        if bad:
            err("fabricated_citation", "model-derived node text contains identifier/citation-like text", node=nid,
                fields=bad)

    # edges
    by_type: dict[str, list[dict[str, Any]]] = {t: [] for t in EDGE_TYPES}
    for e in edges:
        t, s, d = e.get("type"), e.get("source"), e.get("target")
        if t not in EDGE_TYPES:
            err("invalid_edge_type", "unknown edge type", edge=str(e.get("id")))
            continue
        if s not in nodes or d not in nodes:
            err("dangling_edge", "edge endpoint is not a node", edge=str(e.get("id")))
            continue
        if (nodes[s]["type"], nodes[d]["type"]) not in EDGE_RULES[t]:
            err("invalid_edge_direction", "edge type not allowed between these node types", edge=str(e.get("id")))
            continue
        if e.get("id") != edge_id(t, s, d):
            err("invalid_edge_id", "edge id does not follow the deterministic scheme", edge=str(e.get("id")))
        by_type[t].append(e)

    for e in by_type["supports"]:
        s, d = nodes[e["source"]], nodes[e["target"]]
        if s["type"] == "source" and d.get("source_record_id") != s.get("record_id"):
            err("wrong_source_link", "evidence linked to a source it was not extracted from", edge=e["id"])
    for n in nodes.values():
        if n.get("type") == "evidence":
            incoming = [e for e in by_type["supports"] if e["target"] == n["id"]]
            if [e["source"] for e in incoming] != [node_id("source", str(n.get("source_record_id")))]:
                err("evidence_source_link", "evidence needs exactly one supports edge from its source", node=n["id"])
        if n.get("type") == "hypothesis":
            cited = sorted(e["source"] for e in by_type["supports"] if e["target"] == n["id"])
            expected = sorted(node_id("evidence", x) for x in n.get("evidence_ids") or [])
            if not expected or cited != expected:
                err("hypothesis_evidence_links", "hypothesis evidence edges do not match its evidence ids",
                    node=n["id"])
            gap_edges = [e["target"] for e in by_type["addresses"] if e["source"] == n["id"]]
            if gap_edges != [node_id("gap", str(n.get("research_gap_id")))]:
                err("hypothesis_gap_link", "hypothesis must address exactly its research gap", node=n["id"])
            tests = sorted(e["source"] for e in by_type["tests"] if e["target"] == n["id"])
            hid = n.get("hypothesis_id")
            if tests != sorted([node_id("falsification_test", str(hid)), node_id("prediction", str(hid))]):
                err("hypothesis_test_links", "hypothesis needs exactly its prediction and falsification test",
                    node=n["id"])
    for e in by_type["tests"]:
        if nodes[e["source"]].get("parent_hypothesis_id") != nodes[e["target"]].get("hypothesis_id"):
            err("test_parent_mismatch", "prediction/falsification linked to the wrong hypothesis", edge=e["id"])
    for e in by_type["motivates"]:
        g = gaps.get(nodes[e["target"]].get("gap_id")) or {}
        eid = nodes[e["source"]].get("evidence_id")
        if eid not in [*(g.get("supporting_evidence_ids") or []), *(g.get("conflicting_evidence_ids") or [])]:
            err("unrecorded_motivation", "motivates edge not recorded by the gap", edge=e["id"])
    for e in by_type["contradicts"]:
        a, b = nodes[e["source"]].get("evidence_id"), nodes[e["target"]].get("evidence_id")
        recorded = False
        for gid in e.get("recorded_by_gaps") or []:
            g = gaps.get(gid) or {}
            sup, con = set(g.get("supporting_evidence_ids") or []), set(g.get("conflicting_evidence_ids") or [])
            if (a in sup and b in con) or (b in sup and a in con):
                recorded = True
        if not recorded or e["source"] >= e["target"]:
            err("invented_contradiction", "contradicts edge without an explicit conflict record", edge=e["id"])

    # rejected evidence text never appears
    accepted_texts = {t for ev in inputs.accepted_evidence for t in _texts(dict(ev))}
    blob = graph_json({"nodes": graph.get("nodes"), "edges": graph.get("edges")})
    for t in inputs.rejected_evidence_texts:
        if len(t) >= 12 and t not in accepted_texts and json.dumps(t, ensure_ascii=False)[1:-1] in blob:
            err("rejected_evidence_present", "text of rejected evidence appears in the graph")
            break
    # report / graph citation consistency
    if inputs.report_citations is not None:
        report_ids = {c.get("record_id") for c in inputs.report_citations if c.get("status") == "resolved"}
        graph_ids = {n.get("record_id") for n in nodes.values() if n.get("type") == "source"}
        if report_ids != graph_ids:
            err("report_graph_citation_mismatch", "report citations and graph sources are different record sets",
                only_in_report=sorted(str(x) for x in report_ids - graph_ids),
                only_in_graph=sorted(str(x) for x in graph_ids - report_ids))
    # reproducibility
    rebuilt = build_evidence_graph(inputs)
    core = {k: graph.get(k) for k in ("nodes", "edges", "excluded", "content_sha256")}
    if graph_json(core) != graph_json({k: rebuilt.get(k) for k in core}):
        err("not_reproducible", "rebuilding the graph from the same inputs gives a different result")
    return errors


def build_validated_graph(inputs: GraphInputs) -> dict[str, Any]:
    """Build + validate; the validation result is stored inside the graph (deterministic, no timestamps)."""
    graph = build_evidence_graph(inputs)
    errors = validate_evidence_graph(graph, inputs)
    status = "failed" if errors else ("passed_with_exclusions" if graph["excluded"] else "passed")
    graph["validation"] = {"status": status, "errors": errors, "checks": list(VALIDATION_CHECKS)}
    return graph


VALIDATION_CHECKS = (
    "node ids follow the deterministic scheme and every reference resolves (verified record / accepted evidence / "
    "accepted gap / accepted hypothesis)",
    "no rejected evidence (ids or text)",
    "each hypothesis has supports edges exactly for its evidence ids and one addresses edge to its gap",
    "each prediction / falsification test tests exactly its parent hypothesis",
    "source types and preprint labels match the classification; bibliographic fields equal the v0.2 record",
    "no identifier or citation-like text in model-derived node text",
    "contradicts edges only where an accepted gap records the conflict",
    "report citations and graph sources are the same verified record set (when the report is available)",
    "rebuilding from the same inputs is byte-identical",
)


def graph_summary(graph: Mapping[str, Any]) -> dict[str, Any]:
    v = graph.get("validation") or {}
    return {"graph_version": graph.get("graph_version"), "file": GRAPH_FILE, "status": v.get("status"),
            "errors": len(v.get("errors") or []), "counts": graph.get("counts"),
            "content_sha256": graph.get("content_sha256")}


# ------------------------------------------------------------------ DOT rendering (Streamlit st.graphviz_chart)


NODE_STYLE = {
    "source": 'shape=note, style=filled, fillcolor="#E8F0FE"',
    "evidence": 'shape=box, style="rounded,filled", fillcolor="#E6F4EA"',
    "gap": 'shape=hexagon, style=filled, fillcolor="#FEF7E0"',
    "hypothesis": 'shape=ellipse, style=filled, fillcolor="#F3E8FD"',
    "prediction": 'shape=box, style="filled,dashed", fillcolor="#FFFFFF"',
    "falsification_test": 'shape=box, style="filled,dotted", fillcolor="#FFFFFF"',
}
EDGE_STYLE = {
    "supports": 'color="#1E8E3E", penwidth=1.5',
    "contradicts": 'color="#D93025", style=dashed, dir=both, penwidth=2',
    "motivates": 'color="#F29900"',
    "addresses": 'color="#9334E6", penwidth=1.5',
    "tests": 'color="#5F6368", style=dotted',
}
LEGEND = {
    "supports": "source → evidence (extracted from); evidence → hypothesis (cited)",
    "contradicts": "evidence ↔ evidence, only where a research gap records the conflict (red, dashed)",
    "motivates": "evidence → research gap",
    "addresses": "hypothesis → research gap",
    "tests": "prediction / falsification test → hypothesis",
}
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def dot_escape(text: Any) -> str:
    """Escape for a DOT double-quoted string: no raw quotes, backslashes, newlines or control characters."""
    s = _CONTROL.sub(" ", str(text if text is not None else ""))
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _wrap(text: str, width: int = 28, max_lines: int = 3) -> list[str]:
    words, lines, cur = str(text).split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][: width - 1] + "…"
    return lines


def node_title(node: Mapping[str, Any]) -> str:
    t = node.get("type")
    if t == "source":
        extra = f" [{PREPRINT_LABEL}]" if node.get("preprint_label") else ""
        return f"{node.get('title') or node.get('record_id')}{extra}"
    if t == "evidence":
        return str(node.get("claim") or node.get("evidence_id"))
    if t == "gap":
        return str(node.get("gap_statement") or node.get("gap_id"))
    if t == "hypothesis":
        return str(node.get("hypothesis") or node.get("hypothesis_id"))
    if t == "prediction":
        return str(node.get("text") or "")
    return str(node.get("measured") or "")


def _short_id(node: Mapping[str, Any]) -> str:
    t = node.get("type")
    return {"source": "Source", "evidence": node.get("evidence_id"), "gap": node.get("gap_id"),
            "hypothesis": f"{node.get('hypothesis_id')} (unvalidated)",
            "prediction": f"Prediction {node.get('parent_hypothesis_id')}",
            "falsification_test": f"Falsification {node.get('parent_hypothesis_id')}"}.get(t, str(node.get("id")))


def graph_to_dot(graph: Mapping[str, Any]) -> str:
    """DOT string for ``st.graphviz_chart`` (client-side rendering; no graphviz package needed)."""
    out = ["digraph evidence_graph {", '  rankdir=LR; graph [fontname="Helvetica", fontsize=10];',
           '  node [fontname="Helvetica", fontsize=9]; edge [fontname="Helvetica", fontsize=8];']
    for n in graph.get("nodes") or []:
        lines = [str(_short_id(n)), *_wrap(node_title(n))]
        label = "\\n".join(dot_escape(x) for x in lines)
        out.append(f'  "{dot_escape(n["id"])}" [label="{label}", {NODE_STYLE.get(n.get("type"), "")}];')
    for e in graph.get("edges") or []:
        out.append(f'  "{dot_escape(e["source"])}" -> "{dot_escape(e["target"])}" '
                   f'[label="{dot_escape(e["type"])}", {EDGE_STYLE.get(e["type"], "")}];')
    out.append('  subgraph cluster_legend { label="Legend"; fontsize=9; style=dashed;')
    for i, t in enumerate(EDGE_TYPES):
        out.append(f'    "legend_{i}_a" [label="", shape=point, width=0.05]; '
                   f'"legend_{i}_b" [label="{dot_escape(t)}", shape=plaintext];')
        out.append(f'    "legend_{i}_a" -> "legend_{i}_b" [{EDGE_STYLE[t]}];')
    out.append("  }")
    out.append("}")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ selector-based navigation (pure functions)


TYPE_LABELS = {"source": "Source", "evidence": "Evidence", "gap": "Research gap", "hypothesis": "Hypothesis",
               "prediction": "Prediction", "falsification_test": "Falsification test"}


def node_options(graph: Mapping[str, Any]) -> list[tuple[str, str]]:
    """(node_id, display label) grouped by type in pipeline order."""
    order = {t: i for i, t in enumerate(NODE_TYPES)}
    nodes = sorted(graph.get("nodes") or [], key=lambda n: (order.get(n.get("type"), 99), n.get("id")))
    out = []
    for n in nodes:
        title = node_title(n)
        title = title if len(title) <= 70 else title[:69] + "…"
        out.append((n["id"], f"{TYPE_LABELS.get(n.get('type'), n.get('type'))} · {n['id']} — {title}"))
    return out


def node_details(graph: Mapping[str, Any], nid: str) -> dict[str, Any] | None:
    """Details panel content for one node (plain data; the UI escapes when rendering)."""
    nodes = {n["id"]: n for n in graph.get("nodes") or []}
    n = nodes.get(nid)
    if n is None:
        return None
    edges = graph.get("edges") or []

    def linked(etype: str, *, to: str | None = None, frm: str | None = None) -> list[dict[str, Any]]:
        ids = [e["source"] for e in edges if e["type"] == etype and to and e["target"] == to] + \
              [e["target"] for e in edges if e["type"] == etype and frm and e["source"] == frm]
        return [nodes[i] for i in sorted(set(ids)) if i in nodes]

    t = n["type"]
    d: dict[str, Any] = {"id": nid, "type": t, "type_label": TYPE_LABELS[t], "title": node_title(n), "fields": {},
                         "related": {}}
    if t == "source":
        d["fields"] = {"Source type": n.get("source_type_label"), "Preprint": n.get("preprint_label") or "no",
                       "Title": n.get("title"), "Authors": "; ".join(n.get("authors") or []) or None,
                       "Journal": n.get("journal"), "Year": n.get("year"), "DOI": n.get("doi"),
                       "PMID": n.get("pmid"), "URL": n.get("source_url"),
                       "Verification (v0.2)": n.get("verification_status"),
                       "Abstract only": n.get("abstract_only"), "Record": n.get("record_id")}
        d["related"]["Evidence from this source"] = [x["id"] for x in linked("supports", frm=nid)]
    elif t == "evidence":
        d["fields"] = {"Claim": n.get("claim"), "Exact verified quote": n.get("verified_quote"),
                       "Category": n.get("evidence_category"), "Confidence": n.get("confidence"),
                       "Abstract only": n.get("abstract_only"), "Conflict note": n.get("conflict_note")}
        d["related"]["Supporting source"] = [x["id"] for x in linked("supports", to=nid) if x["type"] == "source"]
        d["related"]["Research gaps"] = [x["id"] for x in linked("motivates", frm=nid)]
        d["related"]["Hypotheses citing it"] = [x["id"] for x in linked("supports", frm=nid)]
        d["related"]["Contradicts (recorded by a gap)"] = [x["id"] for x in linked("contradicts", to=nid)
                                                            + linked("contradicts", frm=nid)]
    elif t == "gap":
        d["fields"] = {"Gap": n.get("gap_statement"), "Why unresolved": n.get("why_unresolved"),
                       "Label": n.get("label"), "Confidence": n.get("confidence")}
        d["related"]["Motivating evidence"] = [x["id"] for x in linked("motivates", to=nid)]
        d["related"]["Hypotheses addressing it"] = [x["id"] for x in linked("addresses", to=nid)]
    elif t == "hypothesis":
        alt = n.get("alternative_explanation") or {}
        fals = nodes.get(node_id("falsification_test", n["hypothesis_id"])) or {}
        pred = nodes.get(node_id("prediction", n["hypothesis_id"])) or {}
        d["fields"] = {"Warning": n.get("warning"), "Hypothesis": n.get("hypothesis"),
                       "Claim level": n.get("mechanistic_claim_level"),
                       "Causality": n.get("causality_statement"), "Confidence": n.get("confidence"),
                       "Research gap": n.get("research_gap_id"),
                       "Alternative explanation": f"{alt.get('explanation')} (basis: {alt.get('basis')})",
                       "Prediction": pred.get("text"),
                       "Falsification test": "; ".join(f"{k.replace('_', ' ')}: {fals.get(k)}" for k in
                                                       MODEL_TEXT_FIELDS["falsification_test"]),
                       "Critic": n.get("critic_status"),
                       "Critic verdicts": ", ".join(f"{k}: {v}" for k, v in sorted(
                           (n.get("critic_verdicts") or {}).items())) or None,
                       "Revision": n.get("revision_status"), "Source quality": n.get("source_quality_summary")}
        d["related"]["Evidence"] = [x["id"] for x in linked("supports", to=nid)]
        d["related"]["Addresses gap"] = [x["id"] for x in linked("addresses", frm=nid)]
        d["related"]["Tested by"] = [x["id"] for x in linked("tests", to=nid)]
    else:
        parent = node_id("hypothesis", str(n.get("parent_hypothesis_id")))
        d["fields"] = ({"Prediction": n.get("text")} if t == "prediction" else
                       {k.replace("_", " ").capitalize(): n.get(k) for k in MODEL_TEXT_FIELDS["falsification_test"]})
        d["fields"]["Parent hypothesis warning"] = HYPOTHESIS_NOTICE
        d["related"]["Tests hypothesis"] = [parent] if parent in nodes else []
    d["fields"] = {k: v for k, v in d["fields"].items() if v is not None}
    return d
