"""Truthful claim-check reporting (code only; no model; nothing fabricated).

What actually runs in this build (all deterministic, see
:mod:`sciforge.stages.validation`, :mod:`sciforge.stages.numbers`,
:mod:`sciforge.stages.synthesis_checks`, :mod:`sciforge.stages.report`):

* ``exact_quote`` — every evidence quote must be an exact, case-sensitive
  substring of the abstract that was sent to the model;
* ``numeric_consistency`` — numbers/units in evidence claims/findings must occur
  in the quote (methods: in the source text); numbers in narrative paragraphs,
  gaps and hypotheses must occur in the referenced evidence;
* ``citation_validation`` — every source/evidence id must be one the pipeline
  supplied, no bibliographic fields or citation-like text may come from the
  model, and every citation is rendered by code from the verified v0.2 records
  (unresolvable ids fail closed as ``[UNRESOLVED CITATION: id]``).

Semantic (model-based) claim checking — asking a model whether a non-numeric
claim is actually entailed by its evidence — is **not implemented**. No
semantic check is ever run or reported as run; ``SCIFORGE_MODEL_ENTAILMENT`` is
reserved and has no effect.

Counts are derived from the recorded validation results only: an item "failed"
a check when it carries at least one reason code of that check; "checked" counts
items that reached per-field validation (malformed / non-object items are not
counted as checked).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

__all__ = ["SEMANTIC_CLAIM_CHECK", "SEMANTIC_STATUS", "claim_check_lines", "claim_check_summary"]

SEMANTIC_STATUS = "not implemented"
SEMANTIC_CLAIM_CHECK = {
    "status": SEMANTIC_STATUS,
    "ran": False,
    "note": ("Semantic (model-based) claim checking is not implemented in this build; no semantic check ran. "
             "Only the deterministic checks listed here were applied."),
}

EXACT_QUOTE_CODES = frozenset({"missing_quote", "quote_too_short", "quote_not_in_source"})
NUMERIC_CODES = frozenset({"number_not_in_quote", "number_not_in_source", "unit_mismatch", "unsupported_claim"})
CITATION_CODES = frozenset({"unknown_record_id", "record_id_not_in_request", "bibliographic_field",
                            "fabricated_bibliographic_field", "unknown_evidence_id", "unknown_source_id",
                            "unknown_gap_id", "unknown_hypothesis_id", "identifier_in_text", "bibliographic_text",
                            "missing_evidence_reference"})
NOT_CHECKED_CODES = frozenset({"malformed_item"})


def _codes(item: Mapping[str, Any]) -> set[str]:
    codes = item.get("reason_codes")
    if not isinstance(codes, list):
        codes = [r.get("code") for r in item.get("reasons") or [] if isinstance(r, Mapping)]
    return {c for c in codes if isinstance(c, str)}


def _group(accepted: int, rejected: Iterable[Mapping[str, Any]], codes: frozenset[str]) -> dict[str, int]:
    rejected = list(rejected)
    checked_rejected = [r for r in rejected if not (_codes(r) & NOT_CHECKED_CODES)]
    failed = sum(1 for r in checked_rejected if _codes(r) & codes)
    checked = accepted + len(checked_rejected)
    return {"checked": checked, "passed": checked - failed, "failed": failed}


def claim_check_summary(*, evidence_accepted: int, evidence_rejected: Iterable[Mapping[str, Any]],
                        synthesis_accepted: int = 0, synthesis_rejected: Iterable[Mapping[str, Any]] = (),
                        citations_resolved: int = 0, citations_unresolved: int = 0) -> dict[str, Any]:
    """Structured record of which claim checks ran and their pass/fail counts (never invents results)."""
    ev_rej = list(evidence_rejected)
    syn_rej = list(synthesis_rejected)
    quote = {**_group(evidence_accepted, ev_rej, EXACT_QUOTE_CODES), "scope": "evidence items"}
    ev_num = _group(evidence_accepted, ev_rej, NUMERIC_CODES)
    syn_num = _group(synthesis_accepted, syn_rej, NUMERIC_CODES)
    numeric = {k: ev_num[k] + syn_num[k] for k in ("checked", "passed", "failed")}
    numeric["scope"] = "evidence items and model-written gaps, hypotheses and narrative paragraphs"
    ev_cit = _group(evidence_accepted, ev_rej, CITATION_CODES)
    syn_cit = _group(synthesis_accepted, syn_rej, CITATION_CODES)
    ids = {k: ev_cit[k] + syn_cit[k] for k in ("checked", "passed", "failed")}
    citation = {**ids, "scope": "source/evidence ids and bibliographic fields in model output",
                "citations_rendered_by_code": citations_resolved, "citations_unresolved": citations_unresolved}
    for group in (quote, numeric, citation):
        group["ran"] = group["checked"] > 0
    return {"semantic_claim_check": dict(SEMANTIC_CLAIM_CHECK),
            "deterministic_checks": {"exact_quote": quote, "numeric_consistency": numeric,
                                     "citation_validation": citation}}


def claim_check_lines(summary: Mapping[str, Any]) -> list[str]:
    """Plain-text lines (Markdown-safe, code-built) describing what ran."""
    det = summary.get("deterministic_checks") or {}
    labels = (("exact_quote", "Exact-quote check"), ("numeric_consistency", "Numeric-consistency check"),
              ("citation_validation", "Citation/id validation"))
    lines = []
    for key, label in labels:
        g = det.get(key) or {}
        if g.get("ran"):
            line = f"{label} (deterministic): ran on {g['checked']} item(s) — {g['passed']} passed, {g['failed']} failed"
        else:
            line = f"{label} (deterministic): not run (no items to check)"
        if key == "citation_validation":
            line += (f"; citations rendered by code from verified records: {g.get('citations_rendered_by_code', 0)}, "
                     f"unresolved: {g.get('citations_unresolved', 0)}")
        lines.append(line + ".")
    lines.append("Semantic (model-based) claim check: not implemented — no semantic claim check ran.")
    return lines
