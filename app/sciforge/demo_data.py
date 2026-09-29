"""Bundled SYNTHETIC demo dataset for the web app's Demo Mode (fully offline).

Everything in this module is invented for demonstration purposes. The records
are shaped like v0.2 records (:class:`~sciforge.models.Record` +
:class:`~sciforge.models.VerificationResult`) so the real v0.3 pipeline can run
on them, but they are NOT real publications:

* DOIs use the unassigned prefix ``10.0000/demo.*`` (they resolve nowhere);
* titles start with ``[SYNTHETIC DEMO]``; authors are ``Demo Author ...``;
  the journal is ``SciForge Synthetic Demo Journal (not a real journal)``;
* abstracts start with ``SYNTHETIC DEMO ABSTRACT`` and contain invented numbers;
* verification statuses are assigned by this file, not checked against any
  database (each VerificationResult says so in ``reasons``).

The scripted model (:func:`demo_model_client`) is a
:class:`~sciforge.llm.fake.FakeModelClient` whose responses are chosen from the
request stage. It sees exactly what a real model would see (opaque record ids
and abstracts, no bibliographic data) and builds its answers only from the ids in
the request, so the pipeline's deterministic checks run for real. One extraction
item deliberately quotes text that is not in the abstract, so the demo shows the
exact-quote check rejecting it (its text never reaches the rendered output).

No network access, no API key, no private data.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

import httpx

from sciforge.llm.client import ModelRequest
from sciforge.llm.fake import FakeModelClient
from sciforge.models import Record, VerificationResult

DEMO_LABEL = "SYNTHETIC DEMO DATA — not real findings, not real publications"
DEMO_TOPIC = "study preregistration and reported effect sizes (metascience of the published literature)"
DEMO_QUESTION = ("Is study preregistration associated with smaller reported effect sizes in the published "
                 "scientific literature?")
DEMO_TIMESTAMP = "2026-01-01T00:00:00Z"
DEMO_JOURNAL = "SciForge Synthetic Demo Journal (not a real journal)"
DEMO_MODEL_NAME = "demo-fake-model"
DEMO_VERIFICATION_NOTE = ("SYNTHETIC DEMO: status assigned by the bundled demo dataset; "
                          "not checked against PubMed or Crossref")

# key -> synthetic record fields, abstract, demo verification status
_DEMO_SOURCES: list[dict[str, Any]] = [
    {
        "key": "demo-1", "year": 2021, "status": "verified",
        "title": "[SYNTHETIC DEMO] Effect sizes in simulated preregistered versus conventional studies",
        "authors": ["Demo Author A", "Demo Author B"],
        "abstract": ("SYNTHETIC DEMO ABSTRACT. We compared 120 simulated preregistered studies with 120 simulated "
                     "conventional studies. Preregistered studies reported a median standardized effect size of "
                     "0.21, compared with 0.39 in conventional studies. The simulated data set is not real."),
    },
    {
        "key": "demo-2", "year": 2022, "status": "verified",
        "title": "[SYNTHETIC DEMO] Positive-result rates in a simulated sample of journals",
        "authors": ["Demo Author C"],
        "abstract": ("SYNTHETIC DEMO ABSTRACT. Across a simulated sample of 80 journals, positive results were "
                     "reported in 44% of preregistered reports and in 91% of conventional reports. Journal "
                     "policies were not modelled."),
    },
    {
        "key": "demo-3", "year": 2023, "status": "verified",
        "title": "[SYNTHETIC DEMO] No effect-size difference in a simulated field with strict measurement standards",
        "authors": ["Demo Author D", "Demo Author E"],
        "abstract": ("SYNTHETIC DEMO ABSTRACT. In a simulated field with strict measurement standards, reported "
                     "effect sizes did not differ between preregistered and conventional studies. The simulated "
                     "sample was small."),
    },
    {
        "key": "demo-4", "year": 2020, "status": "partially_verified",
        "title": "[SYNTHETIC DEMO] A partially verified placeholder record",
        "authors": ["Demo Author F"],
        "abstract": "SYNTHETIC DEMO ABSTRACT. This record is excluded because it is only partially verified.",
    },
    {
        "key": "demo-5", "year": 2019, "status": "not_verified",
        "title": "[SYNTHETIC DEMO] An unverified placeholder record",
        "authors": ["Demo Author G"],
        "abstract": "SYNTHETIC DEMO ABSTRACT. This record is excluded because it could not be verified.",
    },
]


def _doi(key: str) -> str:
    return f"10.0000/{key.replace('-', '.')}"


def demo_records(from_year: int | None = None, to_year: int | None = None) -> tuple[list[Record], list[VerificationResult]]:
    """Synthetic v0.2-shaped records + verification results (inclusive year filter, like v0.2)."""
    records: list[Record] = []
    verification: list[VerificationResult] = []
    for src in _DEMO_SOURCES:
        if from_year is not None and src["year"] < from_year:
            continue
        if to_year is not None and src["year"] > to_year:
            continue
        rec = Record(title=src["title"], authors=list(src["authors"]), year=src["year"], doi=_doi(src["key"]),
                     pmid=None, journal=DEMO_JOURNAL, source_database="crossref", source_url=None,
                     retrieval_timestamp=DEMO_TIMESTAMP)
        records.append(rec)
        verification.append(VerificationResult(record_id=rec.record_id, status=src["status"],
                                               reasons=[DEMO_VERIFICATION_NOTE], verified_at=DEMO_TIMESTAMP))
    return records, verification


def _abstract_by_doi() -> dict[str, str]:
    return {_doi(s["key"]): s["abstract"] for s in _DEMO_SOURCES}


def demo_http_client() -> httpx.Client:
    """Offline httpx client that serves the synthetic abstracts in Crossref ``/works/{doi}`` shape.

    Uses ``httpx.MockTransport``: no socket is ever opened. Any other URL gets HTTP 404.
    """
    abstracts = _abstract_by_doi()

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.startswith("/works/"):
            doi = path[len("/works/"):]
            if doi in abstracts:
                payload = {"status": "ok", "message-type": "work",
                           "message": {"DOI": doi, "abstract": f"<jats:p>{abstracts[doi]}</jats:p>"}}
                return httpx.Response(200, content=json.dumps(payload).encode(),
                                      headers={"Content-Type": "application/json"})
        return httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handler))


# ------------------------------------------------------------------ scripted model

# Evidence the scripted model returns per synthetic source (quotes are exact substrings of the abstracts).
_DEMO_EVIDENCE: dict[str, list[dict[str, Any]]] = {
    "demo-1": [
        {"claim": "Preregistered studies reported smaller median effect sizes than conventional studies.",
         "quote": ("Preregistered studies reported a median standardized effect size of 0.21, compared with 0.39 in "
                   "conventional studies."),
         "finding": "Median standardized effect size 0.21 (preregistered) versus 0.39 (conventional).",
         "methods": "Comparison of 120 simulated preregistered and 120 simulated conventional studies.",
         "limitations": "Synthetic demo abstract only.",
         "relevance": "Directly compares effect sizes by preregistration status.",
         "evidence_category": "established", "confidence": "moderate"},
        # Deliberately invalid: the quote is NOT in the abstract -> rejected by the exact-quote check.
        {"claim": "DEMO-REJECTED-CLAIM preregistration removes every source of bias entirely",
         "quote": "DEMO-REJECTED-QUOTE preregistration eliminates all publication bias in every field",
         "finding": None, "methods": None, "limitations": None, "relevance": None,
         "evidence_category": "established", "confidence": "high"},
    ],
    "demo-2": [
        {"claim": "Positive results were less frequent among preregistered reports than conventional reports.",
         "quote": ("positive results were reported in 44% of preregistered reports and in 91% of conventional "
                   "reports"),
         "finding": "Positive-result rate 44% (preregistered) versus 91% (conventional).",
         "methods": "Simulated sample of 80 journals.",
         "limitations": "Journal policies were not modelled; synthetic demo abstract only.",
         "relevance": "Indicates a possible reporting-bias mechanism behind larger conventional effects.",
         "evidence_category": "established", "confidence": "moderate"},
    ],
    "demo-3": [
        {"claim": "In a field with strict measurement standards, effect sizes did not differ by preregistration status.",
         "quote": ("reported effect sizes did not differ between preregistered and conventional studies"),
         "finding": "No difference in reported effect sizes between preregistered and conventional studies.",
         "methods": "Simulated field with strict measurement standards.",
         "limitations": "The simulated sample was small; synthetic demo abstract only.",
         "relevance": "Conflicts with the smaller-effects finding and suggests field-dependence.",
         "evidence_category": "conflicting", "confidence": "low"},
    ],
}


def _payload(request: ModelRequest) -> dict[str, Any]:
    try:
        data = json.loads(request.messages[0].content)
    except (ValueError, IndexError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def _record_keys() -> dict[str, str]:
    records, _ = demo_records()
    return {r.record_id: s["key"] for r, s in zip(records, _DEMO_SOURCES)}


def _question(request: ModelRequest) -> dict[str, Any]:
    question = str(_payload(request).get("research_question_verbatim") or DEMO_QUESTION)
    return {
        "research_question": question,
        "scope": "Demo Mode: the bundled synthetic example dataset only (no literature was searched).",
        "assumptions": ["All demo sources are synthetic and illustrate the pipeline only."],
        "key_concepts": ["preregistration", "effect size", "publication bias"],
        "ambiguities": ["Demo Mode does not interpret the entered question; it always uses the example dataset."],
    }


def _extraction(request: ModelRequest) -> dict[str, Any]:
    keys = _record_keys()
    items: list[dict[str, Any]] = []
    for src in _payload(request).get("sources") or [_payload(request).get("source") or {}]:
        if not isinstance(src, dict):
            continue
        rid = src.get("record_id")
        for item in _DEMO_EVIDENCE.get(keys.get(rid, ""), []):
            items.append({"source_record_id": rid, **item})
    return {"items": items}


def _evidence_by_key(request: ModelRequest) -> dict[str, str]:
    """demo source key -> evidence id, from the evidence actually sent in this request."""
    keys = _record_keys()
    out: dict[str, str] = {}
    for ev in _payload(request).get("evidence") or []:
        if isinstance(ev, dict) and ev.get("source_record_id") in keys:
            out.setdefault(keys[ev["source_record_id"]], ev.get("evidence_id"))
    return out


def _gaps(request: ModelRequest) -> dict[str, Any]:
    ev = _evidence_by_key(request)
    gaps: list[dict[str, Any]] = []
    if "demo-1" in ev and "demo-3" in ev:
        gaps.append({"gap_id": "g1",
                     "gap_statement": "Which field characteristics explain why the smaller-effects pattern appears "
                                      "in one simulated setting but not in another is unresolved.",
                     "supporting_evidence_ids": [ev["demo-1"]], "conflicting_evidence_ids": [ev["demo-3"]],
                     "why_unresolved": "The two synthetic sources studied different simulated fields and did not "
                                       "compare measurement standards directly.",
                     "confidence": "moderate"})
    if "demo-1" in ev and "demo-2" in ev:
        gaps.append({"gap_id": "g2",
                     "gap_statement": "How much of the effect-size difference is explained by selective reporting "
                                      "of positive results is untested.",
                     "supporting_evidence_ids": [ev["demo-1"], ev["demo-2"]], "conflicting_evidence_ids": [],
                     "why_unresolved": "Effect sizes and positive-result rates were reported in separate synthetic "
                                       "samples.",
                     "confidence": "low"})
    return {"gaps": gaps}


def _hypotheses(request: ModelRequest) -> dict[str, Any]:
    ev = _evidence_by_key(request)
    gap_ids = [g.get("gap_id") for g in _payload(request).get("research_gaps") or [] if isinstance(g, dict)]
    hyps: list[dict[str, Any]] = []
    if gap_ids and "demo-1" in ev:
        support = [ev["demo-1"]] + ([ev["demo-3"]] if "demo-3" in ev else [])
        hyps.append({"hypothesis_id": "h1", "label": "hypothesis",
                     "statement": "Strict measurement standards may reduce the effect-size gap between "
                                  "preregistered and conventional studies.",
                     "supporting_evidence_ids": support, "research_gap_ids": [gap_ids[0]],
                     "rationale": "The gap was absent in the simulated field with strict measurement standards.",
                     "predicted_observable_outcome": "Fields with stricter measurement standards would show a "
                                                     "smaller effect-size gap by preregistration status.",
                     "assumptions": ["Measurement standards can be compared across fields."],
                     "confidence": "low"})
    return {"hypotheses": hyps}


def _narrative(request: ModelRequest) -> dict[str, Any]:
    ev = _evidence_by_key(request)
    paragraphs: list[dict[str, Any]] = []
    if "demo-1" in ev:
        paragraphs.append({"section": "key_findings", "label": "established",
                           "text": f"In the synthetic demo data, preregistered studies reported smaller median "
                                   f"effect sizes than conventional studies [{ev['demo-1']}].",
                           "evidence_ids": [ev["demo-1"]], "source_ids": []})
    if "demo-2" in ev:
        paragraphs.append({"section": "key_findings", "label": "inference",
                           "text": f"Lower positive-result rates among preregistered reports may partly explain the "
                                   f"difference [{ev['demo-2']}].",
                           "evidence_ids": [ev["demo-2"]], "source_ids": []})
    if "demo-3" in ev:
        paragraphs.append({"section": "conflicting_evidence", "label": "conflicting",
                           "text": f"One synthetic source found no difference in a field with strict measurement "
                                   f"standards [{ev['demo-3']}].",
                           "evidence_ids": [ev["demo-3"]], "source_ids": []})
    paragraphs.append({"section": "limitations", "label": "inference",
                       "text": "All sources in this demo are synthetic; nothing here is a real finding.",
                       "evidence_ids": [], "source_ids": []})
    paragraphs.append({"section": "next_steps", "label": "hypothesis",
                       "text": "Run the same investigation in Live Mode on real, verified literature.",
                       "evidence_ids": [], "source_ids": []})
    return {"paragraphs": paragraphs}


_RESPONDERS: dict[str | None, Callable[[ModelRequest], dict[str, Any]]] = {
    "question": _question, "extraction": _extraction, "gaps": _gaps, "hypotheses": _hypotheses,
    "report": _narrative,
}


def _respond(request: ModelRequest) -> dict[str, Any]:
    responder = _RESPONDERS.get(request.stage)
    if responder is None:  # pragma: no cover - defensive; unknown stage
        return {}
    return responder(request)


def demo_model_client(max_calls: int = 200) -> FakeModelClient:
    """Offline scripted model for Demo Mode (a FakeModelClient; never performs I/O)."""
    return FakeModelClient([_respond] * max_calls, model=DEMO_MODEL_NAME)


def demo_sources() -> Sequence[dict[str, Any]]:
    """Read-only view of the synthetic source definitions (for docs/tests)."""
    return tuple(dict(s) for s in _DEMO_SOURCES)
