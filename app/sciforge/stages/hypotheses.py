"""Stage S4b — candidate hypotheses (model) + deterministic validation.

Each hypothesis: ``label`` must be exactly ``"hypothesis"`` (``invalid_label``);
the statement must not be phrased as established fact
(``hypothesis_asserted_as_fact``); ≥ 1 known supporting evidence id
(``hypothesis_missing_evidence`` / ``unknown_evidence_id``); ≥ 1 accepted gap id
(``hypothesis_missing_gap`` / ``unknown_gap_id``); same bibliographic, identifier
and number checks as gaps (numbers in statement, rationale, predicted outcome and
assumptions must occur in the referenced evidence). Accepted hypotheses get
code-assigned ids ``hyp_01`` ...; the report renderer always prefixes "Hypothesis:".
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import Field

from sciforge.llm.parsing import StrictModel, strict_json_schema
from sciforge.prompts_synthesis import HYPOTHESES_INSTRUCTIONS, SYNTHESIS_PROMPT_VERSION
from sciforge.stages.common import CallContext
from sciforge.stages.synthesis_checks import (
    CONFIDENCE_LEVELS,
    check_assertive,
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

STAGE = "hypotheses"
MAX_HYPOTHESES = 5
HYPOTHESIS_FIELDS = ("hypothesis_id", "label", "statement", "supporting_evidence_ids", "research_gap_ids",
                     "rationale", "predicted_observable_outcome", "assumptions", "confidence")
TEXT_FIELDS = ("statement", "rationale", "predicted_observable_outcome")


class ModelHypothesis(StrictModel):
    hypothesis_id: str
    label: Literal["hypothesis"]
    statement: str
    supporting_evidence_ids: list[str] = Field(min_length=1)
    research_gap_ids: list[str] = Field(min_length=1)
    rationale: str
    predicted_observable_outcome: str
    assumptions: list[str]
    confidence: Literal["high", "moderate", "low"]


class ModelHypothesisBatch(StrictModel):
    hypotheses: list[ModelHypothesis] = Field(max_length=MAX_HYPOTHESES)


HYPOTHESES_SCHEMA = strict_json_schema(ModelHypothesisBatch)


def validate_hypothesis(raw: Any, *, evidence_by_id: Mapping[str, Mapping[str, Any]], known_sources: set[str],
                        gaps_by_id: Mapping[str, Mapping[str, Any]], next_id: str
                        ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    reasons = check_keys(raw, HYPOTHESIS_FIELDS)
    if reasons and reasons[0]["code"] == "malformed_item":
        return None, reasons
    known_ev, known_gaps = set(evidence_by_id), set(gaps_by_id)
    if raw.get("label") != "hypothesis":
        reasons.append({"code": "invalid_label", "detail": 'label must be exactly "hypothesis"', "field": "label"})
    r, supporting = check_id_list("supporting_evidence_ids", raw.get("supporting_evidence_ids"), known_ev,
                                  unknown_code="unknown_evidence_id", empty_code="hypothesis_missing_evidence")
    reasons += r
    r, gap_ids = check_id_list("research_gap_ids", raw.get("research_gap_ids"), known_gaps,
                               unknown_code="unknown_gap_id", empty_code="hypothesis_missing_gap")
    reasons += r
    inline: set[str] = set()
    texts: dict[str, Any] = {f: raw.get(f) for f in TEXT_FIELDS}
    assumptions = raw.get("assumptions")
    if not isinstance(assumptions, list) or not all(isinstance(a, str) for a in assumptions):
        reasons.append({"code": "schema_violation", "detail": "assumptions: must be a list of strings",
                        "field": "assumptions"})
        assumptions = []
    for i, a in enumerate(assumptions):
        texts[f"assumptions[{i}]"] = a
    for f, value in texts.items():
        r, refs = check_text(f, value, known_evidence=known_ev, known_sources=known_sources, known_gaps=known_gaps)
        reasons += r
        inline |= refs
    reasons += check_assertive("statement", raw.get("statement"))
    if not isinstance(raw.get("hypothesis_id"), str):
        reasons.append({"code": "schema_violation", "detail": "hypothesis_id: must be a string",
                        "field": "hypothesis_id"})
    reference = evidence_reference_text([*supporting, *sorted(inline)], evidence_by_id)
    for f, value in texts.items():
        reasons += check_numbers_supported(f, value, reference)
    reasons += check_enum("confidence", raw.get("confidence"), CONFIDENCE_LEVELS, "invalid_confidence")
    if reasons:
        return None, reasons
    return {
        "id": next_id,
        "hypothesis_id": next_id,
        "model_hypothesis_id": safe_id(raw["hypothesis_id"]),
        "label": "hypothesis",
        "statement": raw["statement"],
        "supporting_evidence_ids": supporting,
        "research_gap_ids": gap_ids,
        "rationale": raw["rationale"],
        "predicted_observable_outcome": raw["predicted_observable_outcome"],
        "assumptions": list(assumptions),
        "confidence": raw["confidence"],
        "source_record_ids": sorted({evidence_by_id[e]["source_record_id"] for e in supporting}),
        "validation": {"deterministic": "passed"},
        "support": combine_support(DeterministicResult(passed=True), None),
    }, []


def gap_view(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ("gap_id", "gap_statement", "supporting_evidence_ids", "conflicting_evidence_ids", "why_unresolved",
            "confidence")
    return [{k: g[k] for k in keys} for g in gaps]


def run_hypotheses_stage(ctx: CallContext, question_context: dict[str, Any], accepted_evidence: list[dict[str, Any]],
                         accepted_gaps: list[dict[str, Any]], known_sources: set[str]) -> SynthesisResult:
    evidence_by_id = {e["evidence_id"]: e for e in accepted_evidence}
    gaps_by_id = {g["gap_id"]: g for g in accepted_gaps}
    payload = {"question_definition": question_context, "evidence": model_evidence_view(accepted_evidence),
               "research_gaps": gap_view(accepted_gaps)}
    counter = {"n": 0}

    def validate(raw: Any, index: int):
        record, reasons = validate_hypothesis(raw, evidence_by_id=evidence_by_id, known_sources=known_sources,
                                              gaps_by_id=gaps_by_id, next_id=f"hyp_{counter['n'] + 1:02d}")
        if record is not None:
            counter["n"] += 1
        return record, reasons

    return run_item_stage(ctx, stage=STAGE, instructions=HYPOTHESES_INSTRUCTIONS, payload=payload,
                          schema_name="candidate_hypotheses", json_schema=HYPOTHESES_SCHEMA, key="hypotheses",
                          allowed_fields=HYPOTHESIS_FIELDS, id_field="hypothesis_id", validate=validate,
                          max_items=MAX_HYPOTHESES)


def hypotheses_json(result: SynthesisResult) -> dict[str, Any]:
    return {**result.base_json(), "prompt_version": SYNTHESIS_PROMPT_VERSION,
            "label_rule": 'every hypothesis is labelled "hypothesis"; rendered with the prefix "Hypothesis:"',
            "accepted": result.accepted, "rejected": result.rejected}
