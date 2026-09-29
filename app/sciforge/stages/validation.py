"""Deterministic validation of model evidence items (pure functions, no I/O).

Per item, ALL applicable reasons are collected (an item is rejected if it has
any); valid items are kept, invalid ones recorded — one bad item never drops
its batch. Rejection reasons:

``malformed_item``                 item is not a JSON object
``bibliographic_field``            item carries a bibliographic key (title, authors, journal, year, doi, pmid, url, ...)
``unexpected_field``               any other key outside the allowed nine
``unknown_record_id``              ``source_record_id`` not in the set of sources supplied to the model
``record_id_not_in_request``       a supplied source, but not one sent in THIS call
``missing_quote``                  quote absent, not a string, or blank
``quote_too_short``                quote shorter than :data:`MIN_QUOTE_CHARS` characters
``quote_not_in_source``            quote is not an exact, case-sensitive substring of the capped source text
``unsupported_evidence_category``  category outside the enum
``unsupported_confidence``         confidence outside the enum
``empty_claim``                    claim blank
``number_not_in_quote``            a number in claim/finding does not occur in the quote (see :mod:`.numbers`)
``number_not_in_source``           a number in methods does not occur in the supplied source text
``unit_mismatch``                  the number occurs, but with a different unit
``schema_violation``               any other type/field error (missing field, wrong type)

Quote rule: EXACT substring (``quote in source_text``) of the exact capped text
that was sent — no case folding, no whitespace or quote-glyph normalisation,
no ellipsis handling. False rejections are preferred over false acceptances.

Rejection records are **safe diagnostics** (:func:`rejection_record`): reason
codes, code-supplied record ids, field names/lengths, SHA-256 of the raw item,
normalised numbers/units. They never contain model free text (claim, quote,
finding, invented titles, ...) or identifier-like strings. Raw rejected output
is kept only with the explicit local debug opt-in (see ``model_pipeline``).

Precedence (:func:`combine_support`): deterministic validation always runs
first and overrides any model-based support assessment — a deterministic
failure is final (``rejected``) whatever a model says; a model label can only
downgrade an item that passed.

Identifier flags (warn, don't reject): DOI / PMID / PMCID / URL / arXiv-like
strings in ``claim``, ``finding``, ``methods``, ``limitations``, ``relevance``
are reported as ``warnings`` on the accepted item. (The quote is exempt: it is
verbatim source text.)
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from sciforge.stages.identifiers import IDENTIFIER_PATTERNS, identifier_warnings, redact_identifiers
from sciforge.stages.numbers import check_numbers
from sciforge.stages.schemas import CONFIDENCE_LEVELS, EVIDENCE_CATEGORIES, ITEM_FIELDS, ModelEvidenceItem

__all__ = [
    "BIBLIOGRAPHIC_KEYS", "DETERMINISTIC_FAILURE_CODES", "IDENTIFIER_PATTERNS", "MIN_QUOTE_CHARS",
    "MODEL_SUPPORT_LABELS", "DeterministicResult", "bibliographic_keys", "combine_support", "identifier_warnings",
    "raw_item_sha256", "redact_model_output_text", "rejection_record", "safe_record_id", "validate_item",
]

MIN_QUOTE_CHARS = 20

BIBLIOGRAPHIC_KEYS = frozenset({
    "title", "paper_title", "article_title", "authors", "author", "first_author", "journal", "journal_title",
    "container_title", "venue", "publisher", "year", "publication_year", "pub_year", "date", "publication_date",
    "doi", "pmid", "pmcid", "pmc", "arxiv", "arxiv_id", "isbn", "issn", "url", "source_url", "link", "links",
    "href", "source", "citation", "citations", "reference", "references", "volume", "issue", "pages",
    "bibliographic", "bibliography",
})
FLAGGED_TEXT_FIELDS = ("claim", "finding", "methods", "limitations", "relevance")

# Unsupported enum values are model text: echo them only if they are one of these harmless near-misses.
SAFE_ENUM_ECHO = frozenset({
    "proven", "supported", "unsupported", "speculative", "speculation", "preliminary", "mixed", "contradictory",
    "controversial", "uncertain", "unknown", "none", "null", "not verified", "not_verified", "strong", "weak",
    "medium", "very high", "very low", "very_high", "very_low", "certain", "inferred", "hypothetical", "fact",
    "consensus", "emerging", "limited",
})
_RECORD_ID_RE = re.compile(r"^rec_[0-9a-f]{16}$")
_SAFE_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,39}$")

DETERMINISTIC_FAILURE_CODES = frozenset({
    "malformed_item", "bibliographic_field", "unexpected_field", "unknown_record_id", "record_id_not_in_request",
    "missing_quote", "quote_too_short", "quote_not_in_source", "unsupported_evidence_category",
    "unsupported_confidence", "empty_claim", "number_not_in_quote", "number_not_in_source", "unit_mismatch",
    "schema_violation", "exceeds_item_limit",
})


# ------------------------------------------------------------------ helpers


def _norm_key(key: Any) -> str:
    return re.sub(r"[\s\-]+", "_", str(key).strip().lower())


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def raw_item_sha256(item: Any) -> str:
    """SHA-256 of the canonical JSON of a raw model item (links diagnostics to the debug file)."""
    return _sha256(json.dumps(item, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str))


def _safe_key(key: Any) -> str:
    norm = _norm_key(key)
    if _SAFE_KEY_RE.match(norm) and not any(p.search(str(key)) for p in IDENTIFIER_PATTERNS.values()):
        return norm
    return "[redacted]"


def _safe_enum(value: Any) -> dict[str, Any]:
    if isinstance(value, str) and value.strip().lower() in SAFE_ENUM_ECHO:
        return {"value": value.strip().lower()}
    if value is None:
        return {"value": None}
    return {"value": "[redacted]", "value_type": type(value).__name__,
            "value_length": len(value) if isinstance(value, str) else None}


def bibliographic_keys(obj: Any) -> list[str]:
    """Bibliographic keys anywhere in ``obj`` (recursive; dotted paths of the raw keys)."""
    found: list[str] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, Mapping):
            for k, v in node.items():
                p = f"{path}.{k}" if path else str(k)
                if _norm_key(k) in BIBLIOGRAPHIC_KEYS:
                    found.append(p)
                walk(v, p)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(obj, "")
    return found


def _bib_key_names(obj: Any) -> list[str]:
    """Normalised bibliographic key names present anywhere (allowlisted names only → safe)."""
    names: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            for k, v in node.items():
                if _norm_key(k) in BIBLIOGRAPHIC_KEYS:
                    names.add(_norm_key(k))
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(obj)
    return sorted(names)


# ------------------------------------------------------------------ item validation


def validate_item(
    item: Any,
    *,
    request_texts: Mapping[str, str],
    supplied_ids: frozenset[str] | set[str],
) -> tuple[ModelEvidenceItem | None, list[dict[str, Any]], list[dict[str, str]]]:
    """Validate one raw model item.

    ``request_texts`` maps the record_ids sent in THIS call to the exact text
    the model saw; ``supplied_ids`` is every record_id supplied in the run.
    Returns (validated item or None, rejection reasons, warnings). Reasons are
    safe diagnostics (no model free text).
    """
    reasons: list[dict[str, Any]] = []

    def reject(code: str, detail: str, **extra: Any) -> None:
        reasons.append({"code": code, "detail": detail, **extra})

    if not isinstance(item, dict):
        reject("malformed_item", f"item is a {type(item).__name__}, not an object")
        return None, reasons, []

    bib_names = _bib_key_names(item)
    if bib_names:
        reject("bibliographic_field", "model attempted to return bibliographic fields (values not recorded)",
               keys=bib_names)
    bib_top = {k for k in item if _norm_key(k) in BIBLIOGRAPHIC_KEYS}
    extra = [k for k in item if k not in ITEM_FIELDS and k not in bib_top]
    if extra:
        reject("unexpected_field", "fields outside the allowed set", keys=sorted({_safe_key(k) for k in extra}),
               count=len(extra))

    rid = item.get("source_record_id")
    text: str | None = None
    if not isinstance(rid, str) or rid not in supplied_ids:
        info: dict[str, Any] = {}
        if isinstance(rid, str):
            info = {"value": rid} if _RECORD_ID_RE.match(rid) else {"value": "[redacted]", "value_sha256": _sha256(rid),
                                                                     "value_length": len(rid)}
        reject("unknown_record_id", "source_record_id is not one of the supplied record ids", **info)
    elif rid not in request_texts:
        reject("record_id_not_in_request", "source_record_id was supplied, but not in this request", value=rid)
    else:
        text = request_texts[rid]

    quote = item.get("quote")
    quote_ok = False
    if not isinstance(quote, str) or not quote.strip():
        reject("missing_quote", "quote is missing or empty")
    elif len(quote) < MIN_QUOTE_CHARS:
        reject("quote_too_short", f"quote has fewer than {MIN_QUOTE_CHARS} characters", quote_length=len(quote))
    elif text is not None and quote not in text:
        reject("quote_not_in_source", "quote is not an exact, case-sensitive substring of the supplied source text",
               quote_length=len(quote))
    else:
        quote_ok = text is not None

    category = item.get("evidence_category")
    if category not in EVIDENCE_CATEGORIES:
        reject("unsupported_evidence_category", f"evidence_category must be one of {list(EVIDENCE_CATEGORIES)}",
               **_safe_enum(category))
    confidence = item.get("confidence")
    if confidence not in CONFIDENCE_LEVELS:
        reject("unsupported_confidence", f"confidence must be one of {list(CONFIDENCE_LEVELS)}",
               **_safe_enum(confidence))
    claim = item.get("claim")
    if isinstance(claim, str) and not claim.strip():
        reject("empty_claim", "claim is empty")

    # numeric consistency (only meaningful against a verified exact quote / known source text)
    if quote_ok:
        for name in ("claim", "finding"):
            if isinstance(item.get(name), str):
                reasons.extend(check_numbers(name, item[name], quote, scope="quote"))  # type: ignore[arg-type]
    if text is not None and isinstance(item.get("methods"), str):
        reasons.extend(check_numbers("methods", item["methods"], text, scope="source_text"))

    candidate = {k: v for k, v in item.items() if k in ITEM_FIELDS}
    validated: ModelEvidenceItem | None = None
    try:
        validated = ModelEvidenceItem.model_validate(candidate)
    except ValidationError as exc:
        covered = {"evidence_category", "confidence"} | ({"quote"} if any(r["code"] == "missing_quote" for r in reasons) else set())
        for err in exc.errors():
            loc = ".".join(str(x) for x in err.get("loc", ()))
            if loc in covered:
                continue
            reject("schema_violation", f"{loc or '<item>'}: {err.get('msg', 'invalid')}")

    if reasons:
        return None, reasons, []
    assert validated is not None
    warnings = identifier_warnings({f: getattr(validated, f) for f in FLAGGED_TEXT_FIELDS})
    return validated, [], warnings


def safe_record_id(value: Any) -> Any:
    """Opaque ``rec_…`` ids verbatim; any other value only as hash + length (or type)."""
    if isinstance(value, str):
        if _RECORD_ID_RE.match(value):
            return value
        return {"redacted": True, "sha256": _sha256(value), "length": len(value)}
    return None if value is None else {"redacted": True, "type": type(value).__name__}


def rejection_record(raw: Any, reasons: list[dict[str, Any]], *, call_record_id: str, item_index: int,
                     supplied_ids: Iterable[str]) -> dict[str, Any]:
    """Safe, redacted record of a rejected item (no model free text, no identifier strings)."""
    supplied = set(supplied_ids)
    diagnostics: dict[str, Any] = {"raw_item_sha256": raw_item_sha256(raw), "item_type": type(raw).__name__}
    rid = raw.get("source_record_id") if isinstance(raw, dict) else None
    if isinstance(raw, dict):
        diagnostics.update({
            "fields_present": [f for f in ITEM_FIELDS if f in raw],
            "fields_missing": [f for f in ITEM_FIELDS if f not in raw],
            "bibliographic_keys": _bib_key_names(raw),
            "other_field_count": sum(1 for k in raw if k not in ITEM_FIELDS),
            "field_lengths": {f: (len(raw[f]) if isinstance(raw[f], str) else type(raw[f]).__name__)
                              for f in ITEM_FIELDS if f in raw},
            "quote_length": len(raw["quote"]) if isinstance(raw.get("quote"), str) else None,
        })
    det = DeterministicResult(passed=False, reasons=tuple(r["code"] for r in reasons))
    return {
        "call_record_id": call_record_id,
        "item_index": item_index,
        "source_record_id": rid if isinstance(rid, str) and rid in supplied else None,
        "source_record_id_supplied": isinstance(rid, str) and rid in supplied,
        "source_record_id_reported": safe_record_id(rid),
        "validation_status": "rejected_deterministic",
        "support": combine_support(det, None),
        "reasons": reasons,
        "diagnostics": diagnostics,
        "raw_output_stored": False,
    }


# ------------------------------------------------------------------ precedence contract

MODEL_SUPPORT_LABELS = ("supported", "partially_supported", "unverifiable", "not_supported")  # strongest → weakest
_MODEL_RANK = {label: i for i, label in enumerate(MODEL_SUPPORT_LABELS)}


@dataclass(frozen=True)
class DeterministicResult:
    """Outcome of the deterministic checks for one item."""

    passed: bool
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.passed and self.reasons:
            raise ValueError("a passed deterministic result cannot carry rejection reasons")
        if not self.passed and not self.reasons:
            raise ValueError("a failed deterministic result needs at least one reason")


def combine_support(deterministic: DeterministicResult, model_label: str | None = None) -> dict[str, Any]:
    """Final support status. Deterministic validation ALWAYS precedes and overrides the model.

    * deterministic failure → ``rejected`` / ``rejected_deterministic`` whatever ``model_label`` says
      (the label is recorded as ignored);
    * passed + no model label → ``accepted`` / ``passed_deterministic`` (model check not run);
    * passed + model label → the model label, which can only lower the status: ``supported`` stays
      ``accepted``; ``partially_supported`` / ``unverifiable`` / ``not_supported`` → ``downgraded``.
      An unrecognised label is treated as ``unverifiable`` (downgrade, never upgrade).
    """
    if not deterministic.passed:
        return {"final_status": "rejected", "final_label": "rejected_deterministic", "decided_by": "deterministic",
                "deterministic": "failed", "deterministic_reasons": list(deterministic.reasons),
                "model_label": model_label if model_label in _MODEL_RANK else (None if model_label is None else "invalid"),
                "model_label_applied": False}
    if model_label is None:
        return {"final_status": "accepted", "final_label": "passed_deterministic", "decided_by": "deterministic",
                "deterministic": "passed", "deterministic_reasons": [], "model_label": None,
                "model_label_applied": False}
    label = model_label if model_label in _MODEL_RANK else "unverifiable"
    return {"final_status": "accepted" if label == "supported" else "downgraded", "final_label": label,
            "decided_by": "model_downgrade" if label != "supported" else "deterministic+model",
            "deterministic": "passed", "deterministic_reasons": [],
            "model_label": model_label if model_label in _MODEL_RANK else "invalid", "model_label_applied": True}


# ------------------------------------------------------------------ audit redaction

_BIB_PAIR_RE = re.compile(
    r'("(?:' + "|".join(sorted(BIBLIOGRAPHIC_KEYS, key=len, reverse=True)) + r')"\s*:\s*)("(?:[^"\\]|\\.)*"|\[[^\]]*\]|[-\w.]+)',
    re.IGNORECASE,
)
BIB_REDACTION = "[redacted:bibliographic]"


def _redact_obj(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: (BIB_REDACTION if _norm_key(k) in BIBLIOGRAPHIC_KEYS else _redact_obj(v)) for k, v in node.items()}
    if isinstance(node, list):
        return [_redact_obj(v) for v in node]
    if isinstance(node, str):
        return redact_identifiers(node)
    return node


def redact_model_output_text(text: str | None) -> str | None:
    """Redact bibliographic-key values and identifier-like strings from raw model output text."""
    if text is None:
        return None
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return redact_identifiers(_BIB_PAIR_RE.sub(lambda m: m.group(1) + json.dumps(BIB_REDACTION), text))
    return json.dumps(_redact_obj(data), ensure_ascii=False)
