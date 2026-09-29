"""Stage S5 — report: model narrative (validated) + deterministic Markdown template (product spec §5, A–J).

The model supplies ONLY narrative paragraphs for C (key findings), E (conflicting
evidence), F (limitations, appended to code-written limitations) and I (next
steps), each with a statement label and references by ``ev_`` / ``rec_`` id.
Everything else is code: A (question definition), B (v0.2 search metadata),
D (evidence matrix), G (accepted gaps), H (accepted hypotheses, prefixed
"Hypothesis:"), J (sources).

J is rendered entirely by code from the ORIGINAL v0.2 records (authors, title,
journal, year, DOI, PMID, URL, v0.2 verification status) looked up by
``record_id``. An id that cannot be resolved fails closed: the line reads
``[UNRESOLVED CITATION: <id>]`` and ``report_validation.json`` records an
``unresolved_citation`` issue — nothing is ever guessed. Source numbers
``[S1]``... are assigned by code in order of first use.

Narrative checks (deterministic, override any model judgement): allowed
section/label; every referenced id known (``unknown_evidence_id``,
``unknown_source_id``); key-findings and conflicting-evidence paragraphs must
reference evidence (``missing_evidence_reference``); no bibliographic keys
(``fabricated_bibliographic_field``), identifiers or citation-like text; no
number absent from the referenced evidence (``unsupported_claim``). Rejected
paragraphs never appear in report.md (safe diagnostics only in
report_validation.json).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Literal

from pydantic import Field

from sciforge.llm.parsing import StrictModel, strict_json_schema
from sciforge.models import Record, VerificationResult
from sciforge.prompts_synthesis import REPORT_INSTRUCTIONS, SYNTHESIS_PROMPT_VERSION
from sciforge.stages.claim_checks import claim_check_lines, claim_check_summary
from sciforge.stages.common import CallContext
from sciforge.stages.hypotheses import gap_view
from sciforge.stages.synthesis_checks import (
    STATEMENT_LABELS,
    check_enum,
    check_id_list,
    check_keys,
    check_numbers_supported,
    check_text,
    evidence_reference_text,
    model_evidence_view,
    safe_id,
)
from sciforge.stages.synthesis_common import SynthesisResult, run_item_stage
from sciforge.stages.validation import DeterministicResult, combine_support

STAGE = "report"
MAX_PARAGRAPHS = 12
NARRATIVE_SECTIONS = ("key_findings", "conflicting_evidence", "limitations", "next_steps")
EVIDENCE_REQUIRED_SECTIONS = ("key_findings", "conflicting_evidence")
PARAGRAPH_FIELDS = ("section", "label", "text", "evidence_ids", "source_ids")
SECTION_TITLES = {
    "A": "A. Research Question", "B": "B. Search Strategy", "C": "C. Key Findings", "D": "D. Evidence Matrix",
    "E": "E. Conflicting Evidence", "F": "F. Limitations", "G": "G. Research Gaps", "H": "H. Candidate Hypotheses",
    "I": "I. Proposed Next Steps", "J": "J. Sources",
}
_SOURCE_ID_RE = re.compile(r"^rec_[0-9a-f]{16}$")


class ModelParagraph(StrictModel):
    section: Literal["key_findings", "conflicting_evidence", "limitations", "next_steps"]
    label: Literal["established", "conflicting", "inference", "hypothesis"]
    text: str
    evidence_ids: list[str]
    source_ids: list[str]


class ModelNarrative(StrictModel):
    paragraphs: list[ModelParagraph] = Field(max_length=MAX_PARAGRAPHS)


NARRATIVE_SCHEMA = strict_json_schema(ModelNarrative)


# ------------------------------------------------------------------ narrative validation


def validate_paragraph(raw: Any, *, evidence_by_id: Mapping[str, Mapping[str, Any]], known_sources: set[str],
                       next_id: str, known_gaps: set[str] | None = None,
                       known_hypotheses: set[str] | None = None) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    reasons = check_keys(raw, PARAGRAPH_FIELDS)
    if reasons and reasons[0]["code"] == "malformed_item":
        return None, reasons
    known_ev = set(evidence_by_id)
    section = raw.get("section")
    reasons += check_enum("section", section, NARRATIVE_SECTIONS, "invalid_section")
    reasons += check_enum("label", raw.get("label"), STATEMENT_LABELS, "invalid_label")
    r, ev_ids = check_id_list("evidence_ids", raw.get("evidence_ids"), known_ev, unknown_code="unknown_evidence_id",
                              empty_code=None)
    reasons += r
    r, src_ids = check_id_list("source_ids", raw.get("source_ids"), known_sources, unknown_code="unknown_source_id",
                               empty_code=None)
    reasons += r
    r, inline = check_text("text", raw.get("text"), known_evidence=known_ev, known_sources=known_sources,
                           known_gaps=set(known_gaps or ()), known_hypotheses=set(known_hypotheses or ()))
    reasons += r
    refs = [*ev_ids, *sorted(inline - set(ev_ids))]
    if section in EVIDENCE_REQUIRED_SECTIONS and not refs:
        reasons.append({"code": "missing_evidence_reference",
                        "detail": f"{section} paragraphs must reference at least one evidence id", "field": "evidence_ids"})
    reasons += check_numbers_supported("text", raw.get("text"), evidence_reference_text(refs, evidence_by_id))
    if reasons:
        return None, reasons
    text = raw["text"]
    return {
        "id": next_id, "paragraph_id": next_id, "section": section, "label": raw["label"], "text": text,
        "evidence_ids": refs, "source_ids": src_ids,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "support": combine_support(DeterministicResult(passed=True), None),
    }, []


def run_narrative_stage(ctx: CallContext, question_context: dict[str, Any], accepted_evidence: list[dict[str, Any]],
                        gaps: list[dict[str, Any]], hypotheses: list[dict[str, Any]],
                        known_sources: set[str]) -> SynthesisResult:
    evidence_by_id = {e["evidence_id"]: e for e in accepted_evidence}
    payload = {
        "question_definition": question_context,
        "evidence": model_evidence_view(accepted_evidence),
        "research_gaps": gap_view(gaps),
        "hypotheses": [{"hypothesis_id": h["hypothesis_id"], "hypothesis": h["hypothesis"],
                        "evidence_ids": h["evidence_ids"], "research_gap_id": h["research_gap_id"],
                        "mechanistic_claim_level": h["mechanistic_claim_level"]}
                       for h in hypotheses],
        "source_ids": sorted(known_sources),
    }
    counter = {"n": 0}

    def validate(raw: Any, index: int):
        record, reasons = validate_paragraph(raw, evidence_by_id=evidence_by_id, known_sources=known_sources,
                                             next_id=f"par_{counter['n'] + 1:02d}",
                                             known_gaps={g["gap_id"] for g in gaps},
                                             known_hypotheses={h["hypothesis_id"] for h in hypotheses})
        if record is not None:
            counter["n"] += 1
        return record, reasons

    return run_item_stage(ctx, stage=STAGE, instructions=REPORT_INSTRUCTIONS, payload=payload,
                          schema_name="report_narrative", json_schema=NARRATIVE_SCHEMA, key="paragraphs",
                          allowed_fields=PARAGRAPH_FIELDS, id_field=None, validate=validate, max_items=MAX_PARAGRAPHS)


# ------------------------------------------------------------------ citations (code only)


def _md(text: Any) -> str:
    """Escape Markdown-significant characters in code-rendered values."""
    s = str(text).replace("\r", " ").replace("\n", " ")
    return re.sub(r"([\\`*_\[\]|<>#])", r"\\\1", s)


ACCESS_LABELS = {"pubmed_abstract": "abstract only (PubMed abstract)",
                 "crossref_abstract": "abstract only (Crossref abstract)"}


PREPRINT_BADGE = "Preprint — not peer-reviewed"
SOURCE_TYPE_BADGES = {
    "preprint": PREPRINT_BADGE,
    "peer-reviewed journal article": "Peer-reviewed journal article (metadata label)",
    "conference paper": "Conference paper",
    "book/chapter": "Book/chapter",
    "unknown": "Source type unknown",
}
HYPOTHESIS_DISCLAIMER = ("Unvalidated, AI-generated hypotheses for further investigation — not validated "
                         "discoveries and not scientific findings.")


def source_type_badge(source_status: str | None) -> str:
    """Short, prominent source-type label shown FIRST for every source (preprints never look peer reviewed)."""
    return SOURCE_TYPE_BADGES.get(source_status or "unknown", SOURCE_TYPE_BADGES["unknown"])


def source_type_label(source_status: str | None) -> str:
    """Code-built source-type detail (metadata-based; preprints always flagged)."""
    if source_status == "preprint":
        return f"Source type: **{PREPRINT_BADGE}** (from bibliographic metadata)."
    if source_status == "peer-reviewed journal article":
        return "Source type: peer-reviewed journal article (metadata-based label, not a guarantee of peer review)."
    if source_status in ("conference paper", "book/chapter"):
        return f"Source type: {source_status} (from bibliographic metadata)."
    return "Source type: unknown (no conclusive bibliographic type metadata)."


def render_citation(ref: str, record: Record, verification: VerificationResult | None,
                    access_level: str | None, source_status: str | None = None) -> str:
    """One J-section line built ONLY from the v0.2 record (never from model text)."""
    authors = "; ".join(_md(a) for a in record.authors) if record.authors else "authors not available"
    title = f"*{_md(record.title)}*" if record.title else "title not available"
    journal = _md(record.journal) if record.journal else "journal not available"
    year = str(record.year) if record.year is not None else "year not available"
    parts = [f"- **[{ref}]** **[{source_type_badge(source_status)}]** {authors}. {title}. {journal}. {year}."]
    parts.append(f"DOI: [{_md(record.doi)}](https://doi.org/{record.doi})." if record.doi else "DOI: not available.")
    parts.append(f"PMID: [{record.pmid}](https://pubmed.ncbi.nlm.nih.gov/{record.pmid}/)." if record.pmid
                 else "PMID: not available.")
    parts.append(f"URL: <{record.source_url}>." if record.source_url else "URL: not available.")
    parts.append(f"v0.2 verification: {verification.status if verification else 'not available'}.")
    parts.append(f"Access: {ACCESS_LABELS.get(access_level or '', 'not accessed')}.")
    parts.append(source_type_label(source_status))
    parts.append(f"Record: `{record.record_id}`.")
    return " ".join(parts)


def unresolved_marker(record_id: Any) -> str:
    shown = record_id if isinstance(record_id, str) and _SOURCE_ID_RE.match(record_id) else safe_id(record_id)
    return f"[UNRESOLVED CITATION: {shown}]"


class SourceIndex:
    """Assigns [S#] numbers in order of first use and resolves ids against v0.2 records (fail closed)."""

    def __init__(self, records: Sequence[Record], verification: Iterable[VerificationResult],
                 access_levels: Mapping[str, str], source_statuses: Mapping[str, str] | None = None) -> None:
        self.records = {r.record_id: r for r in records}
        self.verification = {v.record_id: v for v in verification}
        self.access = dict(access_levels)
        self.statuses = dict(source_statuses or {})
        self.order: list[str] = []

    def ref(self, record_id: str) -> str:
        if record_id not in self.records:
            if record_id not in self.order:
                self.order.append(record_id)
            return unresolved_marker(record_id)
        if record_id not in self.order:
            self.order.append(record_id)
        n = [i for i in self.order if i in self.records].index(record_id) + 1
        return f"S{n}"

    def render(self) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
        lines: list[str] = []
        citations: list[dict[str, Any]] = []
        issues: list[dict[str, Any]] = []
        n = 0
        for rid in self.order:
            record = self.records.get(rid)
            if record is None:
                lines.append(f"- {unresolved_marker(rid)} — no v0.2 record with this id; no citation rendered.")
                citations.append({"ref": None, "record_id": safe_id(rid), "status": "unresolved"})
                issues.append({"code": "unresolved_citation", "record_id": safe_id(rid),
                               "detail": "id not found among the v0.2 records; citation not rendered (fail closed)"})
                continue
            n += 1
            lines.append(render_citation(f"S{n}", record, self.verification.get(rid), self.access.get(rid),
                                         self.statuses.get(rid)))
            citations.append({"ref": f"S{n}", "record_id": rid, "status": "resolved",
                              "v02_verification_status": self.verification[rid].status if rid in self.verification else None,
                              "source_status": self.statuses.get(rid) or "unknown"})
        return lines, citations, issues


# ------------------------------------------------------------------ report template


def _ev_refs(ids: Iterable[str], evidence_by_id: Mapping[str, Mapping[str, Any]], index: SourceIndex) -> str:
    out = []
    for eid in ids:
        ev = evidence_by_id.get(eid)
        if ev is None:
            continue
        out.append(f"{eid} [{index.ref(ev['source_record_id'])}]")
    return ", ".join(out) if out else "none"


def _paragraph_line(p: Mapping[str, Any], evidence_by_id: Mapping[str, Mapping[str, Any]], index: SourceIndex) -> str:
    refs = _ev_refs(p["evidence_ids"], evidence_by_id, index)
    extra = [index.ref(s) for s in p["source_ids"]]
    tail = f" — evidence: {refs}" + (f"; sources: {', '.join(f'[{x}]' for x in extra)}" if extra else "")
    return f"- **[{p['label']}]** {_md(p['text'].strip())}{tail}"



def _search_queries(w: Callable[[str], None], summary: Mapping[str, Any], question: str) -> None:
    """Section B query lines: every query actually sent to each database (not just the question)."""
    qe = summary.get("query_expansion") or {}
    queries = qe.get("queries") or []
    w(f"- **Query generation:** {_md(summary.get('query_generation', 'not recorded'))}")
    if not queries:
        w(f"- **Query used (verbatim):** `{_md(summary.get('query_used', question))}`")
        return
    enabled = "enabled" if qe.get("enabled") else "disabled"
    w(f"- **Query expansion:** {enabled}; {len(queries)} "
      f"quer{'y' if len(queries) == 1 else 'ies'} per database (the question verbatim is always q1)")
    for q in queries:
        results = q.get("results") or {}
        for db in ("pubmed", "crossref"):
            if db not in q:
                continue
            r = results.get(db) or {}
            counts = (f"{r.get('retrieved', 'n/a')} retrieved, {r.get('contributed', 'n/a')} new candidates"
                      if r else "counts not recorded")
            w(f"  - {_md(str(q.get('query_id', '?')))} {db}: {_md(str(q[db]))} — {counts}")


def _selection_lines(w: Callable[[str], None], summary: Mapping[str, Any]) -> None:
    sel = summary.get("selection") or {}
    if not sel:
        return
    params = summary.get("parameters") or {}
    w(f"- **Candidate pool:** {params.get('records_requested_per_query', 'n/a')} records requested per query and "
      f"database; {summary.get('total_retrieved', 'n/a')} candidates before deduplication")
    enr = summary.get("abstract_enrichment") or {}
    if enr:
        if enr.get("limit"):
            w(f"- **Abstract enrichment (for ranking only):** limit {enr.get('limit')} candidates; "
              f"{enr.get('considered', 0)} considered, {enr.get('with_abstract', 0)} with an abstract "
              f"(batched PubMed efetch or Crossref abstract field), "
              f"{enr.get('title_only_considered', 0)} considered but title-only")
        else:
            w("- **Abstract enrichment:** disabled (SCIFORGE_ABSTRACT_ENRICHMENT_LIMIT=0); ranking used titles only")
    w(f"- **Selection:** {_md(str(sel.get('method', 'deterministic')))}; target {sel.get('target', 'n/a')} verified "
      f"records, {sel.get('candidates_verified', 'n/a')} candidates sent to verification"
      + (" (backfilled after failed verification)" if sel.get("backfilled") else "")
      + f", {len(sel.get('selected_record_ids') or [])} selected")
    pol = summary.get("source_policy") or {}
    if pol:
        counts = pol.get("selected_status_counts") or {}
        w(f"- **Source policy:** {_md(str(pol.get('policy')))}; excluded by policy: {pol.get('excluded_count', 0)}; "
          "selected source types (metadata-based): "
          + (", ".join(f"{_md(k)} {v}" for k, v in counts.items()) or "none"))
        if pol.get("preprints_selected"):
            w(f"- **Preprints:** {pol['preprints_selected']} selected source(s) are preprints (not peer reviewed); "
              f"they are labelled \"{PREPRINT_BADGE}\" in sections D and J")
        if pol.get("fewer_than_requested"):
            w(f"- **Fewer sources than requested:** {_md(str(pol.get('shortfall_note') or ''))}")

CRITIC_CHECK_LABELS = {
    "evidence_supports_mechanism": "Evidence supports the proposed mechanism",
    "causal_language_exceeds_evidence": "Causal language within what the evidence supports",
    "distinct_from_evidence": "Distinct from simply restating the evidence",
    "prediction_measurable": "Prediction is measurable",
    "prediction_discriminates": "Prediction discriminates from the alternative",
    "falsification_meaningful": "Falsification test is meaningful",
    "ignored_contradictions_or_missing_evidence": "No contradictions or missing evidence ignored",
    "confidence_consistent": "Confidence consistent with the evidence",
}


def _hypothesis_section(w: Callable[[str], None], hypotheses: SynthesisResult | None,
                        evidence_by_id: Mapping[str, Any], index: SourceIndex, gaps: SynthesisResult | None) -> None:
    """Section H: every hypothesis with why it was proposed, alternatives, falsification, limitations,
    critic findings / revision status and validation status (all structure rendered by code)."""
    from sciforge.hypothesis_validation import HYPOTHESIS_NOTICE

    engine = getattr(hypotheses, "critic_status", None)
    if hypotheses is not None and engine is not None:
        w(f"- **Stress test:** generation -> critic ({hypotheses.critic_status}"
          + (f", {_md(str(hypotheses.critic_skip_reason))}" if hypotheses.critic_skip_reason else "")
          + f") -> revision ({hypotheses.revision_status.replace('_', ' ')}"
          + (f", {_md(str(hypotheses.revision_skip_reason))}" if hypotheses.revision_skip_reason else "")
          + f"). Candidates {hypotheses.candidates}, accepted {len(hypotheses.accepted)}, rejected "
          f"{len(hypotheses.rejected)}. Deterministic validation outranks the model and the critic.")
        w("")
    gap_text = {g["gap_id"]: g["gap_statement"] for g in (gaps.accepted if gaps else [])}
    if not (hypotheses and hypotheses.accepted):
        w("No candidate hypotheses passed the deterministic checks.")
    for h in (hypotheses.accepted if hypotheses else []):
        if "hypothesis" not in h:        # defensive: pre-v0.4 record shape
            continue
        w(f"### {h['hypothesis_id']} — {HYPOTHESIS_NOTICE}")
        w("")
        w(f"- **Label:** {h['label']}")
        w(f"- **Candidate hypothesis:** {_md(h['hypothesis'])}")
        w(f"- **Claim level:** {h['mechanistic_claim_level'].replace('_', ' ')} (cited evidence supports at most "
          f"{h['supported_claim_level'].replace('_', ' ')}). {_md(h['causality_statement'])}")
        w(f"- **Why it was proposed:** {_md(h['rationale'])} — research gap {h['research_gap_id']}"
          + (f" ({_md(gap_text[h['research_gap_id']])})" if h['research_gap_id'] in gap_text else "")
          + f"; linked evidence: {_ev_refs(h['evidence_ids'], evidence_by_id, index)}.")
        w(f"- **Prediction (measurable):** {_md(h['prediction'])}")
        alt = h["alternative_explanation"]
        basis = (f"evidence: {_ev_refs(alt['evidence_ids'], evidence_by_id, index)}" if alt["basis"] == "evidence"
                 else "inference (not tied to specific evidence)")
        w(f"- **Alternative explanation:** {_md(alt['explanation'])} — basis: {basis}.")
        f = h["falsification_test"]
        fx = {k: _md(v).rstrip(". ") for k, v in f.items()}
        w(f"- **How it could be falsified:** compare/manipulate: {fx['manipulated_or_compared']}; measure: "
          f"{fx['measured']}; would weaken it: {fx['weakening_result']}; would support it: "
          f"{fx['supporting_result']}.")
        w("- **Assumptions:** " + ("; ".join(_md(a).rstrip(".") for a in h["assumptions"]) or "none stated") + ".")
        w("- **Limitations:** " + "; ".join(_md(a).rstrip(".") for a in h["evidence_limitations"]) + ".")
        w(f"- **Source quality:** {_md(h['source_quality_summary'])}")
        c = h["confidence_detail"]
        w(f"- **Confidence (qualitative):** {c['final']} (model proposed {c['proposed_by_model']}; deterministic "
          f"ceiling {c['deterministic_ceiling']}: " + "; ".join(_md(r) for r in c["reasons"]) + ").")
        st = h["stress_test"]
        w(f"- **Critic review:** {st['critic_status'].replace('_', ' ')}"
          + ("" if st["note"] == "critic review completed" else f" — {_md(st['note'])}") + ".")
        for name, finding in (st.get("critic_findings") or {}).items():
            w(f"  - {CRITIC_CHECK_LABELS.get(name, name)}: **{finding['verdict']}** — {_md(finding['explanation'])}")
        if st.get("substantive_problems"):
            w("  - Substantive problems raised: " + "; ".join(_md(p) for p in st["substantive_problems"]))
        w(f"- **Revision status:** {st['revision_status']}"
          + (f" (initial deterministic flags: {', '.join(h['validation']['initial_flags'])})"
             if h["validation"].get("initial_flags") else "") + ".")
        w("- **Validation status:** passed deterministic validation (evidence links, research gap, prediction, "
          "alternative, falsification test, quotes, bibliographic checks, claim level). This is not scientific "
          "validation.")
        w("")
    rejected = hypotheses.rejected if hypotheses else []
    if rejected:
        w("**Rejected candidate hypotheses** (not shown; safe diagnostics only):")
        for r in rejected:
            label = r.get("hypothesis_id") or f"item {r['item_index']}"
            w(f"- {label} — rejected at {r.get('rejected_at', 'generation')}: "
              + ", ".join(dict.fromkeys(r["reason_codes"])) + ".")


def build_report(
    *,
    question: str,
    question_definition: Mapping[str, Any] | None,
    search_summary: Mapping[str, Any] | None,
    source_texts: Mapping[str, Any],
    evidence: Mapping[str, Any],
    gaps: SynthesisResult | None,
    hypotheses: SynthesisResult | None,
    narrative: SynthesisResult | None,
    records: Sequence[Record],
    verification: Sequence[VerificationResult],
    stage_notes: list[str],
    extra_issues: list[dict[str, Any]] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Render report.md and the report_validation.json payload."""
    accepted = list(evidence.get("accepted", []))
    evidence_by_id = {e["evidence_id"]: e for e in accepted}
    access = {s["record_id"]: s.get("access_level") for s in source_texts.get("sources", [])}
    statuses = (search_summary or {}).get("source_classification") or {}
    index = SourceIndex(records, verification, access, statuses)
    synthesis = [r for r in (gaps, hypotheses, narrative) if r is not None]
    check_inputs = {"evidence_accepted": len(accepted), "evidence_rejected": list(evidence.get("rejected", [])),
                    "synthesis_accepted": sum(len(r.accepted) for r in synthesis),
                    "synthesis_rejected": [x for r in synthesis for x in r.rejected]}
    paragraphs = narrative.accepted if narrative else []
    by_section: dict[str, list[dict[str, Any]]] = {s: [p for p in paragraphs if p["section"] == s]
                                                   for s in NARRATIVE_SECTIONS}
    out: list[str] = []
    w = out.append

    w("# SciForge research report")
    w("")
    w("> Development build (v0.3 model layer + v0.4 hypothesis engine, not released). Sections B, D, G, H and J are built by code; C, E, I "
      "and part of F contain model-written narrative that passed deterministic checks. Every citation in J is "
      "rendered by code from verified v0.2 records; the model cannot create citations. All evidence is from "
      "abstracts only.")
    w("")

    # A
    w(f"## {SECTION_TITLES['A']}")
    w("")
    w(f"- **Question (verbatim):** {_md(question)}")
    if question_definition:
        w(f"- **Restatement (model-proposed):** {_md(question_definition.get('research_question', ''))}")
        w(f"- **Scope (model-proposed):** {_md(question_definition.get('scope', ''))}")
        for label, key in (("Assumptions", "assumptions"), ("Ambiguities", "ambiguities")):
            items = question_definition.get(key) or []
            w(f"- **{label} (model-proposed):** " + ("; ".join(_md(i) for i in items) if items else "none stated"))
    else:
        w("- Question definition unavailable; the raw question was used.")
    w("")

    # B
    w(f"## {SECTION_TITLES['B']}")
    w("")
    if search_summary:
        params = search_summary.get("parameters") or {}
        _search_queries(w, search_summary, question)
        w(f"- **Databases:** {', '.join(search_summary.get('databases_queried') or []) or 'not recorded'}")
        w(f"- **Run:** started {search_summary.get('started_at', 'n/a')}, finished {search_summary.get('finished_at', 'n/a')} (UTC)")
        w(f"- **Year filter:** from {params.get('from_year') or 'any'} to {params.get('to_year') or 'any'}")
        hits = search_summary.get("total_hits_reported") or {}
        got = search_summary.get("retrieved_per_source") or {}
        for db in search_summary.get("databases_queried") or []:
            w(f"- **{db}:** status {(search_summary.get('search_status') or {}).get(db, 'n/a')}, "
              f"hits reported {hits.get(db, 'n/a')}, candidates retrieved {got.get(db, 'n/a')}")
        w(f"- **Unique records after deduplication:** {search_summary.get('unique_records', 'n/a')}")
        _selection_lines(w, search_summary)
        vc = search_summary.get("verification_counts") or {}
        w(f"- **v0.2 verification:** " + ", ".join(f"{k} {v}" for k, v in vc.items()))
    else:
        w("- v0.2 search metadata was not provided to the report stage.")
    concepts = (question_definition or {}).get("key_concepts") or []
    w(f"- **Concepts (model-proposed, not used for searching):** " + ("; ".join(_md(c) for c in concepts) or "none"))
    if (search_summary or {}).get("selection"):
        criteria = ("year filter (if any) and the source policy shown above; candidates were deduplicated, ranked "
                    "deterministically (no model) from retrieved titles, abstracts retrieved by bounded enrichment "
                    "and query provenance, and DOI/PMID-verified before use (see above). ")
    else:
        criteria = ("year filter (if any); no candidate ranking or source-policy step was recorded for this run "
                    "(see the query/record lines above). ")
    w("- **Inclusion/exclusion criteria:** " + criteria + "Model eligibility: "
      f"{source_texts.get('eligibility_policy', 'verified')} records, at most {source_texts.get('max_sources', 'n/a')} "
      f"sources, abstracts capped at {source_texts.get('max_source_chars', 'n/a')} characters.")
    w("")

    # C
    w(f"## {SECTION_TITLES['C']}")
    w("")
    if by_section["key_findings"]:
        for p in by_section["key_findings"]:
            w(_paragraph_line(p, evidence_by_id, index))
    elif accepted:
        w("_Code-generated fallback (no validated narrative): accepted evidence claims._")
        for ev in accepted:
            w(f"- **[{ev['evidence_category']}]** {_md(ev['claim'])} — evidence: {_ev_refs([ev['evidence_id']], evidence_by_id, index)}")
    else:
        w("No validated evidence; no key findings can be stated.")
    w("")

    # D
    w(f"## {SECTION_TITLES['D']}")
    w("")
    if accepted:
        w("| Evidence | Claim | Supporting quote (verbatim) | Category | Confidence | Access | Source | Source type |")
        w("|---|---|---|---|---|---|---|---|")
        for ev in accepted:
            w(f"| {ev['evidence_id']} | {_md(ev['claim'])} | {_md(ev['quote'])} | {ev['evidence_category']} | "
              f"{ev['confidence']} | {ACCESS_LABELS.get(ev.get('access_level') or '', 'abstract only')} | "
              f"[{index.ref(ev['source_record_id'])}] | "
              f"**{source_type_badge(index.statuses.get(ev['source_record_id']))}** |")
    else:
        w("No accepted evidence records.")
    w("")

    # E
    w(f"## {SECTION_TITLES['E']}")
    w("")
    conflicting_ev = [e for e in accepted if e.get("evidence_category") == "conflicting"]
    if by_section["conflicting_evidence"]:
        for p in by_section["conflicting_evidence"]:
            w(_paragraph_line(p, evidence_by_id, index))
    elif conflicting_ev:
        for ev in conflicting_ev:
            w(f"- **[conflicting]** {_md(ev['claim'])} — evidence: {_ev_refs([ev['evidence_id']], evidence_by_id, index)}")
    else:
        w("None identified in the validated evidence.")
    w("")

    # F
    w(f"## {SECTION_TITLES['F']}")
    w("")
    counts = source_texts.get("counts") or {}
    ev_counts = evidence.get("counts") or {}
    w("- All evidence comes from abstracts only (no full text); methods and limitations may be incomplete.")
    w(f"- Source texts: {', '.join(f'{k} {v}' for k, v in counts.items()) or 'none'}; truncated abstracts: "
      f"{sum(1 for s in source_texts.get('sources', []) if s.get('truncated'))}.")
    if source_texts.get("eligibility_policy") == "verified_or_partial":
        w("- Partially verified v0.2 records were included by explicit opt-in and are labelled as such.")
    w(f"- Evidence items: {ev_counts.get('accepted', 0)} accepted, {ev_counts.get('rejected', 0)} rejected by "
      "deterministic validation (exact quote, numbers, ids, bibliographic fields).")
    for name, res in (("Research gaps", gaps), ("Hypotheses", hypotheses), ("Narrative paragraphs", narrative)):
        if res is not None:
            w(f"- {name}: {len(res.accepted)} accepted, {len(res.rejected)} rejected; stage status {res.status}"
              + (f" ({res.skip_reason})" if res.skip_reason else "") + ".")
    claim_check_start = len(out)
    for line in claim_check_lines(claim_check_summary(**check_inputs)):
        w(f"- {line}")   # re-rendered below once citation counts are known
    for note in stage_notes:
        w(f"- {_md(note)}")
    for p in by_section["limitations"]:
        w(_paragraph_line(p, evidence_by_id, index))
    w("")

    # G
    w(f"## {SECTION_TITLES['G']}")
    w("")
    if gaps and gaps.accepted:
        for g in gaps.accepted:
            conflicting = _ev_refs(g["conflicting_evidence_ids"], evidence_by_id, index)
            w(f"- **{g['gap_id']}** **[{g['label']}]** {_md(g['gap_statement'])} Why unresolved: "
              f"{_md(g['why_unresolved'])} — supporting evidence: {_ev_refs(g['supporting_evidence_ids'], evidence_by_id, index)}; "
              f"conflicting evidence: {conflicting}; confidence: {g['confidence']}.")
    else:
        w("No validated research gaps.")
    w("")

    # H
    w(f"## {SECTION_TITLES['H']}")
    w("")
    w(f"_{HYPOTHESIS_DISCLAIMER}_")
    w("")
    _hypothesis_section(w, hypotheses, evidence_by_id, index, gaps)
    w("")

    # I
    w(f"## {SECTION_TITLES['I']}")
    w("")
    if by_section["next_steps"]:
        for p in by_section["next_steps"]:
            w(_paragraph_line(p, evidence_by_id, index))
    else:
        w("- Obtain and read the full text of the cited sources (this report used abstracts only).")
        if hypotheses and hypotheses.accepted:
            w("- Design studies that test the candidate hypotheses in H against their predictions and falsification "
              "tests.")
        if gaps and gaps.accepted:
            w("- Search specifically for evidence addressing the research gaps in G.")
    w("")

    # J
    w(f"## {SECTION_TITLES['J']}")
    w("")
    lines, citations, issues = index.render()
    claim_checks = claim_check_summary(
        **check_inputs, citations_resolved=sum(1 for c in citations if c["status"] == "resolved"),
        citations_unresolved=sum(1 for c in citations if c["status"] == "unresolved"))
    for i, line in enumerate(claim_check_lines(claim_checks)):
        out[claim_check_start + i] = f"- {line}"
    if lines:
        type_counts: dict[str, int] = {}
        for c in citations:
            if c["status"] == "resolved":
                badge = source_type_badge(c.get("source_status"))
                type_counts[badge] = type_counts.get(badge, 0) + 1
        w("Each source starts with its source type (from bibliographic metadata; \"peer-reviewed journal "
          "article\" is a metadata label, not a guarantee of peer review). Source types of cited sources: "
          + ", ".join(f"**{k}** {v}" for k, v in sorted(type_counts.items())) + ".")
        w("")
        out.extend(lines)
    else:
        w("No sources cited.")
    w("")
    w("---")
    w("Report validation details: `report_validation.json`.")

    report = "\n".join(out) + "\n"
    issues = list(extra_issues or []) + issues
    sections = [SECTION_TITLES[k] for k in "ABCDEFGHIJ" if f"## {SECTION_TITLES[k]}" in report]
    validation = {
        "prompt_version": SYNTHESIS_PROMPT_VERSION,
        "status": "issues" if issues else ("passed_with_rejections" if narrative and narrative.rejected else "passed"),
        "deterministic_precedence": "deterministic checks run first and override any model judgement; rejected "
                                    "paragraphs are excluded from report.md",
        "sections_present": sections,
        "narrative": {**(narrative.base_json() if narrative else {"status": "not_run"}),
                      "accepted": [{k: p[k] for k in ("paragraph_id", "section", "label", "evidence_ids", "source_ids",
                                                      "text_sha256")} for p in paragraphs],
                      "rejected": narrative.rejected if narrative else []},
        "citations": citations,
        "claim_checks": claim_checks,
        "issues": issues,
        "counts": {"citations_resolved": sum(1 for c in citations if c["status"] == "resolved"),
                   "citations_unresolved": sum(1 for c in citations if c["status"] == "unresolved"),
                   "paragraphs_accepted": len(paragraphs),
                   "paragraphs_rejected": len(narrative.rejected) if narrative else 0},
    }
    return report, validation
