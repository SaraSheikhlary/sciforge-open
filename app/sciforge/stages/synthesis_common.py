"""Runner shared by the M3 list-of-items stages (one structured call + per-item deterministic validation)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sciforge.boundary import canonical_json
from sciforge.llm.client import ModelMessage
from sciforge.stages.common import CallContext, StructuredResult, call_structured
from sciforge.stages.synthesis_checks import envelope_parser, safe_rejection
from sciforge.stages.validation import raw_item_sha256

# validate(raw, index) -> (accepted record or None, reasons)
ItemValidator = Callable[[Any, int], tuple[dict[str, Any] | None, list[dict[str, Any]]]]


@dataclass
class SynthesisResult:
    stage: str
    status: str = "not_run"                 # ok | failed | skipped | not_run
    skip_reason: str | None = None
    accepted: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    rejected_raw: list[dict[str, Any]] = field(default_factory=list)   # debug only; never serialised
    redaction_plan: dict[str, Any] | None = None
    call: StructuredResult[Any] | None = None

    @property
    def stop(self) -> str | None:
        return self.call.stop if self.call else None

    def counts(self) -> dict[str, int]:
        codes: dict[str, int] = {}
        for r in self.rejected:
            for c in r["reason_codes"]:
                codes[c] = codes.get(c, 0) + 1
        return {"returned": len(self.accepted) + len(self.rejected), "accepted": len(self.accepted),
                "rejected": len(self.rejected), **{f"rejected_{k}": v for k, v in sorted(codes.items())}}

    def base_json(self) -> dict[str, Any]:
        call = self.call
        return {
            "stage": self.stage,
            "status": self.status,
            "skip_reason": self.skip_reason,
            "counts": self.counts(),
            "logical_calls": call.logical_calls if call else 0,
            "attempts": call.attempts if call else 0,
            "repair_attempted": call.repaired if call else False,
            "errors": call.errors if call and not call.ok else [],
            "rejected_items": "safe diagnostics only; raw model output is not stored",
        }


def run_item_stage(ctx: CallContext, *, stage: str, instructions: str, payload: dict[str, Any], schema_name: str,
                   json_schema: dict[str, Any], key: str, allowed_fields: tuple[str, ...], id_field: str | None,
                   validate: ItemValidator, max_items: int) -> SynthesisResult:
    result = SynthesisResult(stage=stage)
    call = call_structured(ctx, stage=stage, instructions=instructions,
                           messages=[ModelMessage("user", canonical_json(payload))], schema_name=schema_name,
                           json_schema=json_schema, validate=envelope_parser(key))
    result.call = call
    summaries: list[dict[str, Any]] = []
    # audit metadata for model_calls.json redaction; M3 calls span many records, so no call_record_id
    inputs = {"synthesis_stage": stage,
              "input_evidence_ids": [e["evidence_id"] for e in payload.get("evidence", [])],
              "input_source_record_ids": sorted({e["source_record_id"] for e in payload.get("evidence", [])})}
    if not call.ok:
        result.status = "failed"
        result.redaction_plan = {**inputs, "audit_indices": call.audit_indices,
                                 "final_response_sha256": None, "items": []}
        return result
    result.status = "ok"
    for index, raw in enumerate(call.value or []):
        if index >= max_items:
            accepted, reasons = None, [{"code": "exceeds_item_limit", "detail": f"more than {max_items} items"}]
        else:
            accepted, reasons = validate(raw, index)
        if accepted is None:
            record = safe_rejection(raw, index, reasons, allowed_fields, id_field=id_field)
            result.rejected.append(record)
            result.rejected_raw.append({"stage": stage, "item_index": index, "raw_item_sha256": raw_item_sha256(raw),
                                        "model_output": raw})
            summaries.append({k: record[k] for k in ("item_index", "status", "validation_status", "reason_codes",
                                                     "reasons", "diagnostics", "model_item_id")})
        else:
            result.accepted.append(accepted)
            summaries.append({"item_index": index, "status": "accepted", "id": accepted.get("id"),
                              "raw_item_sha256": raw_item_sha256(raw)})
    if result.rejected or call.repaired:
        result.redaction_plan = {**inputs, "audit_indices": call.audit_indices,
                                 "final_response_sha256": call.response_sha256, "items": summaries}
    return result
