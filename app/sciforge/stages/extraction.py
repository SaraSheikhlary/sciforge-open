"""Stage S2 — evidence extraction (model) + deterministic validation.

Batching: ONE call per source (plan §5 S2). Each request contains the
question definition (or the raw question if S1 failed) and exactly one
model-visible source (``record_id``, ``access_level``, ``source_text`` — see
:mod:`sciforge.boundary`). One call per source keeps prompts small, makes a
per-source failure/repair independent of the others, and makes "which source
does this quote come from" unambiguous. With the default budget (10 sources,
30 attempts) that is ≤ 10 calls + 1 question call, leaving room for repairs
and retries.

Structural problems with the whole response (not JSON, not an object, no
``items`` list, extra top-level keys) trigger ONE repair retry. Problems with
individual items do not: each item is validated deterministically
(:mod:`sciforge.stages.validation`), valid ones are kept and invalid ones are
recorded as SAFE diagnostics (reason codes, lengths, hashes; no model free
text — :func:`~sciforge.stages.validation.rejection_record`). The raw rejected
items are kept only in memory (``rejected_raw``) for the pipeline's opt-in local
debug file; they never enter ``evidence.json``. Evidence ids (``ev_0001`` ...) are assigned by code in
acceptance order; ``access_level``, ``abstract_only``, ``source_identity`` and
``source_text_sha256`` are filled by code from the SourceText, never by the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sciforge.boundary import ModelSource, canonical_json, source_payload
from sciforge.llm.client import ModelMessage, ModelSchemaError
from sciforge.prompts import EXTRACTION_INSTRUCTIONS, PROMPT_VERSION
from sciforge.sourcetext import SourceText
from sciforge.stages.common import CallContext, call_structured
from sciforge.stages.schemas import CONFIDENCE_LEVELS, EVIDENCE_CATEGORIES, EVIDENCE_SCHEMA, MAX_ITEMS_PER_SOURCE
from sciforge.stages.validation import (
    MIN_QUOTE_CHARS,
    DeterministicResult,
    _safe_key,
    combine_support,
    raw_item_sha256,
    rejection_record,
    validate_item,
)

STAGE = "extraction"
SCHEMA_NAME = "evidence_batch"


def parse_envelope(data: Any, raw: str) -> list[Any]:
    """Structural check of an extraction response; returns the raw item list."""
    if not isinstance(data, dict):
        raise ModelSchemaError("output failed schema validation: <root>: must be an object with an 'items' list",
                               raw_text=raw)
    extra = sorted({_safe_key(k) for k in data if k != "items"})
    if extra:
        raise ModelSchemaError(f"output failed schema validation: <root>: unexpected top-level fields {extra}",
                               raw_text=raw)
    items = data.get("items")
    if not isinstance(items, list):
        raise ModelSchemaError("output failed schema validation: items: must be a list", raw_text=raw)
    return items


@dataclass
class ExtractionResult:
    accepted: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    stop: str | None = None
    question_definition_used: bool = False
    rejected_raw: list[dict[str, Any]] = field(default_factory=list)       # debug only; never serialised here
    # one plan per call whose audit entries must be redacted (rejected items / failed validation / repair):
    # {"record_id", "audit_indices", "final_response_sha256", "items": [safe per-item summaries]}
    redaction_plans: list[dict[str, Any]] = field(default_factory=list)

    @property
    def redact_audit_indices(self) -> list[int]:
        return sorted({i for plan in self.redaction_plans for i in plan["audit_indices"]})

    def counts(self) -> dict[str, int]:
        reason_counts: dict[str, int] = {}
        for r in self.rejected:
            for reason in r["reasons"]:
                reason_counts[reason["code"]] = reason_counts.get(reason["code"], 0) + 1
        return {
            "sources_sent": len(self.calls),
            "calls_ok": sum(1 for c in self.calls if c["status"] == "ok"),
            "calls_failed": sum(1 for c in self.calls if c["status"] == "failed"),
            "calls_skipped": sum(1 for c in self.calls if c["status"] == "skipped"),
            "items_returned": len(self.accepted) + len(self.rejected),
            "accepted": len(self.accepted),
            "rejected": len(self.rejected),
            **{f"rejected_{k}": v for k, v in sorted(reason_counts.items())},
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "stage": STAGE,
            "prompt_version": PROMPT_VERSION,
            "status": "stopped" if self.stop else ("ok" if not self.errors else "completed_with_errors"),
            "stop_reason": self.stop,
            "abstract_only": True,
            "batching": "one model call per source",
            "question_definition_used": self.question_definition_used,
            "rules": {
                "quote_match": f"exact, case-sensitive substring of the supplied (capped) source text; "
                               f"min {MIN_QUOTE_CHARS} characters; no normalisation",
                "evidence_categories": list(EVIDENCE_CATEGORIES),
                "confidence_levels": list(CONFIDENCE_LEVELS),
                "max_items_per_source": MAX_ITEMS_PER_SOURCE,
                "bibliographic_fields": "rejected (model may only reference the opaque source_record_id)",
                "numbers": "claim/finding numbers must occur in the quote, methods numbers in the source text; "
                           "units must match",
                "precedence": "deterministic validation precedes and overrides any model support label",
                "rejected_items": "safe diagnostics only; raw model output is not stored",
            },
            "counts": self.counts(),
            "accepted": self.accepted,
            "rejected": self.rejected,
            "calls": self.calls,
            "errors": self.errors,
        }


def extraction_messages(question_context: dict[str, Any], sources: list[ModelSource]) -> list[ModelMessage]:
    payload = {"question_definition": question_context, "sources": source_payload(sources)}
    return [ModelMessage("user", canonical_json(payload))]


def run_extraction_stage(ctx: CallContext, question_context: dict[str, Any], sources: list[SourceText],
                         *, question_definition_used: bool) -> ExtractionResult:
    """Run one extraction call per usable source and validate every item."""
    from sciforge.boundary import to_model_source

    result = ExtractionResult(question_definition_used=question_definition_used)
    usable = [s for s in sources if s.usable]
    supplied = frozenset(s.record_id for s in usable)
    by_id = {s.record_id: s for s in usable}
    for source in usable:
        if result.stop:
            result.calls.append({"record_id": source.record_id, "status": "skipped", "reason": result.stop,
                                 "logical_calls": 0, "attempts": 0, "repair_attempted": False})
            continue
        model_source = to_model_source(source)
        call = call_structured(ctx, stage=STAGE, instructions=EXTRACTION_INSTRUCTIONS,
                               messages=extraction_messages(question_context, [model_source]),
                               schema_name=SCHEMA_NAME, json_schema=EVIDENCE_SCHEMA, validate=parse_envelope,
                               record_id=source.record_id)
        result.errors.extend(call.errors if not call.ok else [])
        result.calls.append({"record_id": source.record_id, "status": "ok" if call.ok else "failed",
                             "logical_calls": call.logical_calls, "attempts": call.attempts,
                             "repair_attempted": call.repaired, "error": call.error})
        if call.stop:
            result.stop = call.stop
        if not call.ok:
            result.redaction_plans.append({"record_id": source.record_id, "audit_indices": call.audit_indices,
                                           "final_response_sha256": None, "items": []})
            continue
        request_texts = {model_source.record_id: model_source.source_text}
        rejected_before = len(result.rejected)
        summaries: list[dict[str, Any]] = []

        def reject(raw: Any, index: int, reasons: list[dict[str, Any]]) -> None:
            record = rejection_record(raw, reasons, call_record_id=source.record_id, item_index=index,
                                      supplied_ids=supplied)
            result.rejected.append(record)
            summaries.append({"item_index": index, "status": "rejected",
                              "validation_status": record["validation_status"],
                              "source_record_id": record["source_record_id_reported"],
                              "reason_codes": [r["code"] for r in reasons], "reasons": reasons,
                              "diagnostics": record["diagnostics"]})
            result.rejected_raw.append({"call_record_id": source.record_id, "item_index": index,
                                        "raw_item_sha256": raw_item_sha256(raw), "model_output": raw})

        for index, raw in enumerate(call.value or []):
            if index >= MAX_ITEMS_PER_SOURCE:
                reject(raw, index, [{"code": "exceeds_item_limit",
                                     "detail": f"more than {MAX_ITEMS_PER_SOURCE} items for one source"}])
                continue
            item, reasons, warnings = validate_item(raw, request_texts=request_texts, supplied_ids=supplied)
            if item is None:
                reject(raw, index, reasons)
                continue
            src = by_id[item.source_record_id]
            summaries.append({"item_index": index, "status": "accepted",
                              "evidence_id": f"ev_{len(result.accepted) + 1:04d}",
                              "source_record_id": item.source_record_id, "raw_item_sha256": raw_item_sha256(raw)})
            result.accepted.append({
                "evidence_id": f"ev_{len(result.accepted) + 1:04d}",
                **item.model_dump(),
                "access_level": src.access_level,
                "abstract_only": True,
                "source_identity": src.verification_status,
                "source_text_sha256": src.sha256,
                "source_text_truncated": src.truncated,
                "validation": {"deterministic": "passed", "quote_exact_substring": True,
                               "record_id_supplied": True, "numbers_consistent": True},
                "support": combine_support(DeterministicResult(passed=True), None),
                "warnings": warnings,
            })
        if len(result.rejected) > rejected_before or call.repaired:
            result.redaction_plans.append({"record_id": source.record_id, "audit_indices": call.audit_indices,
                                           "final_response_sha256": call.response_sha256, "items": summaries})
    return result
