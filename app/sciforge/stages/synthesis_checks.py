"""Deterministic checks shared by the M3 synthesis stages (gaps, hypotheses, report narrative).

"V0.2 remains the authority for source identity. V0.3 can interpret verified
sources, but it cannot create citations."

Reason codes (all deterministic; a failure always overrides any model judgement,
see :func:`sciforge.stages.validation.combine_support`):

``malformed_item``                 item is not a JSON object
``fabricated_bibliographic_field`` item carries a bibliographic key (title, authors, doi, url, ...)
``unexpected_field``               any other key outside the stage schema
``schema_violation``               wrong type / missing field
``empty_text``                     a required text field is blank
``unknown_evidence_id``            an ``ev_`` id (list or inline text) that is not an accepted evidence record
``unknown_source_id``              a ``rec_`` id (list or inline text) that is not a pipeline source record
``unknown_gap_id``                 a gap id that is not an accepted gap
``unknown_hypothesis_id``          report: an inline hypothesis id that is not an accepted hypothesis
``missing_evidence_reference``     no evidence id where at least one is required
``hypothesis_missing_evidence``    hypothesis without supporting evidence ids
``hypothesis_missing_gap``         hypothesis without research gap ids
``hypothesis_asserted_as_fact``    hypothesis phrased as established fact
``invalid_label``                  label not allowed (hypothesis label must be "hypothesis")
``invalid_confidence``             confidence not high / moderate / low
``invalid_section``                narrative section not allowed
``identifier_in_text``             DOI / PMID / PMCID / URL / arXiv string in model text
``bibliographic_text``             citation-like text (et al., author–year, journal names, "Title:" labels)
``unsupported_claim``              a number/statistic in the text that does not occur in the referenced
                                   evidence (claim + finding), or occurs with another unit
``exceeds_item_limit``             more items than the stage allows
``unresolved_citation``            report: a cited id could not be resolved to a v0.2 record (fail closed)

Rejection records are SAFE diagnostics only: codes, field names, lengths,
SHA-256 of the raw item, normalised number/unit, bibliographic key names and
ids only when they have the pipeline's own opaque form. Never model free text.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from sciforge.stages.identifiers import IDENTIFIER_PATTERNS
from sciforge.stages.numbers import check_numbers
from sciforge.stages.validation import (
    BIBLIOGRAPHIC_KEYS,
    DeterministicResult,
    _bib_key_names,
    _norm_key,
    _safe_key,
    _sha256,
    combine_support,
    raw_item_sha256,
)

CONFIDENCE_LEVELS = ("high", "moderate", "low")
STATEMENT_LABELS = ("established", "conflicting", "inference", "hypothesis")

SYNTHESIS_REASON_CODES = frozenset({
    "malformed_item", "fabricated_bibliographic_field", "unexpected_field", "schema_violation", "empty_text",
    "unknown_evidence_id", "unknown_source_id", "unknown_gap_id", "unknown_hypothesis_id", "missing_evidence_reference",
    "hypothesis_missing_evidence", "hypothesis_missing_gap", "hypothesis_asserted_as_fact", "invalid_label",
    "invalid_confidence", "invalid_section", "identifier_in_text", "bibliographic_text", "unsupported_claim",
    "exceeds_item_limit", "unresolved_citation",
})

EVIDENCE_ID_RE = re.compile(r"^ev_\d{4}$")
SOURCE_ID_RE = re.compile(r"^rec_[0-9a-f]{16}$")
GAP_ID_RE = re.compile(r"^gap_\d{2}$")
HYP_ID_RE = re.compile(r"^hyp_\d{2}$")
_SAFE_ID_RES = (EVIDENCE_ID_RE, SOURCE_ID_RE, GAP_ID_RE, HYP_ID_RE)

_INLINE_EV = re.compile(r"\bev_[A-Za-z0-9_]*")
_INLINE_REC = re.compile(r"\brec_[A-Za-z0-9_]*")
_INLINE_GAP = re.compile(r"\bgap_[A-Za-z0-9_]*")
_INLINE_HYP = re.compile(r"\bhyp_[A-Za-z0-9_]*")

BIBLIOGRAPHIC_TEXT_PATTERNS: dict[str, re.Pattern[str]] = {
    "et_al": re.compile(r"\bet\s+al\b", re.IGNORECASE),
    "author_year": re.compile(
        r"\(\s*[A-Z][A-Za-z'\-]+(?:\s+(?:and|&)\s+[A-Z][A-Za-z'\-]+)?(?:\s+et\s+al\.?)?,?\s+(?:1[89]|20)\d{2}[a-z]?\s*\)"
        r"|\b[A-Z][a-z]+(?:\s+(?:and|&)\s+[A-Z][a-z]+)?\s+\((?:1[89]|20)\d{2}[a-z]?\)"),
    "journal_name": re.compile(
        r"\b(?:Journal of|J\.\s?[A-Z][a-z]+\.|Proc\.?\s+Natl|Proceedings of the|N\s?Engl\s?J\s?Med|NEJM|PLoS\s+[A-Z]|"
        r"JAMA|BMJ|Lancet)\b"),
    "field_label": re.compile(r"\b(?:title|authors?|journal|doi|pmid|pmcid|citation|reference)\s*:", re.IGNORECASE),
}

ASSERTIVE_PATTERNS = [
    re.compile(r"\bit\s+(?:is|has\s+been|was)\s+(?:well[\s-])?(?:established|proven|proved|demonstrated|shown|known|"
               r"confirmed|certain)\b", re.IGNORECASE),
    re.compile(r"\b(?:proves|proved|proven|demonstrates|demonstrated|confirms|confirmed|establishes|established)\s+"
               r"that\b", re.IGNORECASE),
    re.compile(r"\b(?:definitively|undoubtedly|unquestionably|conclusively)\b", re.IGNORECASE),
    re.compile(r"\b(?:clearly\s+shows|is\s+a\s+(?:proven\s+)?fact|is\s+known\s+to)\b", re.IGNORECASE),
]


def safe_id(value: Any) -> Any:
    """Ids in the pipeline's own opaque forms verbatim; anything else as hash + length."""
    if isinstance(value, str):
        if any(r.match(value) for r in _SAFE_ID_RES):
            return value
        return {"redacted": True, "sha256": _sha256(value), "length": len(value)}
    return None if value is None else {"redacted": True, "type": type(value).__name__}


def check_keys(raw: Any, allowed: Iterable[str]) -> list[dict[str, Any]]:
    allowed = tuple(allowed)
    if not isinstance(raw, dict):
        return [{"code": "malformed_item", "detail": f"item is a {type(raw).__name__}, not an object"}]
    reasons: list[dict[str, Any]] = []
    bib = _bib_key_names(raw)
    if bib:
        reasons.append({"code": "fabricated_bibliographic_field",
                        "detail": "model returned bibliographic fields (values not recorded)", "keys": bib})
    extra = [k for k in raw if k not in allowed and _norm_key(k) not in BIBLIOGRAPHIC_KEYS]
    if extra:
        reasons.append({"code": "unexpected_field", "detail": "fields outside the allowed set",
                        "keys": sorted({_safe_key(k) for k in extra}), "count": len(extra)})
    return reasons


def check_text(field: str, text: Any, *, known_evidence: set[str], known_sources: set[str],
               known_gaps: set[str] | None = None, known_hypotheses: set[str] | None = None,
               required: bool = True) -> tuple[list[dict[str, Any]], set[str]]:
    """Identifier / citation-pattern / inline-id checks for one text field.

    Returns (reasons, evidence ids referenced inline).
    """
    reasons: list[dict[str, Any]] = []
    if text is None and not required:
        return reasons, set()
    if not isinstance(text, str):
        return [{"code": "schema_violation", "detail": f"{field}: must be a string", "field": field}], set()
    if required and not text.strip():
        return [{"code": "empty_text", "detail": f"{field} is empty", "field": field}], set()
    for kind, pattern in IDENTIFIER_PATTERNS.items():
        if pattern.search(text):
            reasons.append({"code": "identifier_in_text", "detail": "identifier-like string in model text",
                            "field": field, "type": kind})
    for kind, pattern in BIBLIOGRAPHIC_TEXT_PATTERNS.items():
        if pattern.search(text):
            reasons.append({"code": "bibliographic_text", "detail": "citation-like text in model text",
                            "field": field, "type": kind})
    referenced: set[str] = set()
    for m in _INLINE_EV.finditer(text):
        token = m.group(0)
        if token in known_evidence:
            referenced.add(token)
        else:
            reasons.append({"code": "unknown_evidence_id", "detail": "inline evidence id is not an accepted record",
                            "field": field, "value": safe_id(token)})
    for m in _INLINE_REC.finditer(text):
        token = m.group(0)
        if token not in known_sources:
            reasons.append({"code": "unknown_source_id", "detail": "inline source id is not a pipeline record",
                            "field": field, "value": safe_id(token)})
    if known_gaps is not None:
        for m in _INLINE_GAP.finditer(text):
            if m.group(0) not in known_gaps:
                reasons.append({"code": "unknown_gap_id", "detail": "inline gap id is not an accepted gap",
                                "field": field, "value": safe_id(m.group(0))})
    if known_hypotheses is not None:
        for m in _INLINE_HYP.finditer(text):
            if m.group(0) not in known_hypotheses:
                reasons.append({"code": "unknown_hypothesis_id", "detail": "inline hypothesis id is not accepted",
                                "field": field, "value": safe_id(m.group(0))})
    return reasons, referenced


def check_id_list(field: str, value: Any, known: set[str], *, unknown_code: str, empty_code: str | None,
                  pattern: re.Pattern[str] | None = None) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate a list of ids. Returns (reasons, valid ids in order)."""
    if value is None:
        value = []
    if not isinstance(value, list):
        return [{"code": "schema_violation", "detail": f"{field}: must be a list", "field": field}], []
    reasons: list[dict[str, Any]] = []
    valid: list[str] = []
    for v in value:
        if isinstance(v, str) and v in known:
            if v not in valid:
                valid.append(v)
        else:
            reasons.append({"code": unknown_code, "detail": f"{field} contains an unknown id", "field": field,
                            "value": safe_id(v)})
    if empty_code and not value:
        reasons.append({"code": empty_code, "detail": f"{field} must reference at least one id", "field": field})
    return reasons, valid


def check_enum(field: str, value: Any, allowed: tuple[str, ...], code: str) -> list[dict[str, Any]]:
    if value in allowed:
        return []
    safe = value if isinstance(value, str) and _norm_key(value) in {"very_high", "very_low", "medium", "none",
                                                                   "unknown", "certain", "fact", "established"} else "[redacted]"
    return [{"code": code, "detail": f"{field} must be one of {list(allowed)}", "field": field,
             "value": None if value is None else safe}]


def check_numbers_supported(field: str, text: Any, reference: str) -> list[dict[str, Any]]:
    """Numbers in ``text`` must occur (same unit) in ``reference`` → else ``unsupported_claim``."""
    if not isinstance(text, str):
        return []
    out = []
    for r in check_numbers(field, text, reference, scope="source_text"):
        out.append({"code": "unsupported_claim",
                    "detail": f"{field}: number not found in the referenced evidence"
                    if r["code"] != "unit_mismatch" else f"{field}: number found with a different unit",
                    "field": field, "value": r["value"], "unit": r["unit"],
                    "subtype": "number_not_in_evidence" if r["code"] != "unit_mismatch" else "unit_mismatch"})
    return out


def check_assertive(field: str, text: Any) -> list[dict[str, Any]]:
    if not isinstance(text, str):
        return []
    for p in ASSERTIVE_PATTERNS:
        if p.search(text):
            return [{"code": "hypothesis_asserted_as_fact", "detail": f"{field} is phrased as established fact",
                     "field": field}]
    return []


def evidence_reference_text(evidence_ids: Iterable[str], evidence_by_id: Mapping[str, Mapping[str, Any]]) -> str:
    """Claim + finding of the referenced evidence (the model saw exactly these fields)."""
    parts: list[str] = []
    for eid in evidence_ids:
        ev = evidence_by_id.get(eid)
        if ev:
            parts.extend(str(ev.get(k) or "") for k in ("claim", "finding"))
    return "\n".join(parts)


def safe_rejection(raw: Any, index: int, reasons: list[dict[str, Any]], allowed: Iterable[str],
                   *, id_field: str | None = None) -> dict[str, Any]:
    """Safe, redacted record of a rejected synthesis item (no model free text)."""
    allowed = tuple(allowed)
    diagnostics: dict[str, Any] = {"raw_item_sha256": raw_item_sha256(raw), "item_type": type(raw).__name__}
    if isinstance(raw, dict):
        diagnostics.update({
            "fields_present": [f for f in allowed if f in raw],
            "fields_missing": [f for f in allowed if f not in raw],
            "bibliographic_keys": _bib_key_names(raw),
            "other_field_count": sum(1 for k in raw if k not in allowed),
            "field_lengths": {f: (len(raw[f]) if isinstance(raw[f], (str, list)) else type(raw[f]).__name__)
                              for f in allowed if f in raw},
        })
    det = DeterministicResult(passed=False, reasons=tuple(r["code"] for r in reasons))
    return {
        "item_index": index,
        "model_item_id": safe_id(raw.get(id_field)) if isinstance(raw, dict) and id_field else None,
        "status": "rejected",
        "validation_status": "rejected_deterministic",
        "support": combine_support(det, None),
        "reason_codes": [r["code"] for r in reasons],
        "reasons": reasons,
        "diagnostics": diagnostics,
        "raw_output_stored": False,
    }


def envelope_parser(key: str):
    """Structural check for ``{key: [...]}`` responses (raises ModelSchemaError → one repair)."""
    from sciforge.llm.client import ModelSchemaError

    def parse(data: Any, raw: str) -> list[Any]:
        if not isinstance(data, dict):
            raise ModelSchemaError(f"output failed schema validation: <root>: must be an object with a '{key}' list",
                                   raw_text=raw)
        extra = sorted({_safe_key(k) for k in data if k != key})
        if extra:
            raise ModelSchemaError(f"output failed schema validation: <root>: unexpected top-level fields {extra}",
                                   raw_text=raw)
        items = data.get(key)
        if not isinstance(items, list):
            raise ModelSchemaError(f"output failed schema validation: {key}: must be a list", raw_text=raw)
        return items

    return parse


def model_evidence_view(accepted: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Model-visible evidence (allowlist): evidence_id, source_record_id, claim, finding, category, confidence.

    Identifier-like substrings are redacted (defence in depth; accepted items are code-validated).
    """
    from sciforge.stages.identifiers import redact_identifiers

    out = []
    for ev in accepted:
        out.append({
            "evidence_id": ev["evidence_id"],
            "source_record_id": ev["source_record_id"],
            "claim": redact_identifiers(ev.get("claim") or ""),
            "finding": redact_identifiers(ev["finding"]) if ev.get("finding") else None,
            "evidence_category": ev.get("evidence_category"),
            "confidence": ev.get("confidence"),
        })
    return out
