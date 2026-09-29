"""Stage S4a — research gaps (model) + deterministic validation.

Input to the model (allowlist): the question definition and validated evidence
records only — ``evidence_id``, ``source_record_id``, ``claim``, ``finding``,
``evidence_category``, ``confidence``. No bibliographic data.

Each gap must: reference ≥ 1 accepted evidence id (``supporting_evidence_ids``);
reference only accepted ids (supporting / conflicting / inline); contain no
bibliographic keys, identifiers or citation-like text; state no number that
is absent from the referenced evidence (``unsupported_claim``); use a valid
confidence. Accepted gaps get code-assigned ids ``gap_01`` ... in order and the
statement label ``inference`` (a gap is the pipeline's reasoning, not a source
statement).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import Field

from sciforge.llm.parsing import StrictModel, strict_json_schema
from sciforge.prompts_synthesis import GAPS_INSTRUCTIONS, SYNTHESIS_PROMPT_VERSION
from sciforge.stages.common import CallContext
from sciforge.stages.synthesis_checks import (
    CONFIDENCE_LEVELS,
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

STAGE = "gaps"
MAX_GAPS = 6
GAP_FIELDS = ("gap_id", "gap_statement", "supporting_evidence_ids", "conflicting_evidence_ids", "why_unresolved",
              "confidence")


class ModelGap(StrictModel):
    gap_id: str
    gap_statement: str
    supporting_evidence_ids: list[str] = Field(min_length=1)
    conflicting_evidence_ids: list[str] = Field(default_factory=list)
    why_unresolved: str
    confidence: Literal["high", "moderate", "low"]


class ModelGapBatch(StrictModel):
    gaps: list[ModelGap] = Field(max_length=MAX_GAPS)


GAPS_SCHEMA = strict_json_schema(ModelGapBatch)


def validate_gap(raw: Any, *, evidence_by_id: Mapping[str, Mapping[str, Any]], known_sources: set[str],
                 next_id: str) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    reasons = check_keys(raw, GAP_FIELDS)
    if reasons and reasons[0]["code"] == "malformed_item":
        return None, reasons
    known_ev = set(evidence_by_id)
    r, supporting = check_id_list("supporting_evidence_ids", raw.get("supporting_evidence_ids"), known_ev,
                                  unknown_code="unknown_evidence_id", empty_code="missing_evidence_reference")
    reasons += r
    r, conflicting = check_id_list("conflicting_evidence_ids", raw.get("conflicting_evidence_ids"), known_ev,
                                   unknown_code="unknown_evidence_id", empty_code=None)
    reasons += r
    inline: set[str] = set()
    for f in ("gap_statement", "why_unresolved"):
        r, refs = check_text(f, raw.get(f), known_evidence=known_ev, known_sources=known_sources)
        reasons += r
        inline |= refs
    if not isinstance(raw.get("gap_id"), str):
        reasons.append({"code": "schema_violation", "detail": "gap_id: must be a string", "field": "gap_id"})
    reference = evidence_reference_text([*supporting, *conflicting, *sorted(inline)], evidence_by_id)
    for f in ("gap_statement", "why_unresolved"):
        reasons += check_numbers_supported(f, raw.get(f), reference)
    reasons += check_enum("confidence", raw.get("confidence"), CONFIDENCE_LEVELS, "invalid_confidence")
    if reasons:
        return None, reasons
    return {
        "id": next_id,
        "gap_id": next_id,
        "model_gap_id": safe_id(raw["gap_id"]),
        "label": "inference",
        "gap_statement": raw["gap_statement"],
        "supporting_evidence_ids": supporting,
        "conflicting_evidence_ids": conflicting,
        "why_unresolved": raw["why_unresolved"],
        "confidence": raw["confidence"],
        "source_record_ids": sorted({evidence_by_id[e]["source_record_id"] for e in [*supporting, *conflicting]}),
        "validation": {"deterministic": "passed"},
        "support": combine_support(DeterministicResult(passed=True), None),
    }, []


def run_gaps_stage(ctx: CallContext, question_context: dict[str, Any], accepted_evidence: list[dict[str, Any]],
                   known_sources: set[str]) -> SynthesisResult:
    evidence_by_id = {e["evidence_id"]: e for e in accepted_evidence}
    payload = {"question_definition": question_context, "evidence": model_evidence_view(accepted_evidence)}
    counter = {"n": 0}

    def validate(raw: Any, index: int):
        record, reasons = validate_gap(raw, evidence_by_id=evidence_by_id, known_sources=known_sources,
                                       next_id=f"gap_{counter['n'] + 1:02d}")
        if record is not None:
            counter["n"] += 1
        return record, reasons

    return run_item_stage(ctx, stage=STAGE, instructions=GAPS_INSTRUCTIONS, payload=payload, schema_name="research_gaps",
                          json_schema=GAPS_SCHEMA, key="gaps", allowed_fields=GAP_FIELDS, id_field="gap_id",
                          validate=validate, max_items=MAX_GAPS)


def gaps_json(result: SynthesisResult) -> dict[str, Any]:
    return {**result.base_json(), "prompt_version": SYNTHESIS_PROMPT_VERSION,
            "model_input": "question definition + validated evidence (evidence_id, source_record_id, claim, finding, "
                           "evidence_category, confidence); no bibliographic data",
            "accepted": result.accepted, "rejected": result.rejected}
