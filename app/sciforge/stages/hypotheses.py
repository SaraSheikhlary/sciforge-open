"""Stage S4b — v0.4 hypothesis reliability engine: generation → critic → revision, all validated in code.

Pipeline (one batched call each, all through the shared budget / retry / audit):

1. **generation** (``hypotheses``): up to 5 structured hypotheses
   (:data:`~sciforge.hypothesis_validation.HYPOTHESIS_FIELDS`). Every item is validated by
   :func:`~sciforge.hypothesis_validation.validate_hypothesis_fields`. HARD problems (schema, unknown or
   rejected evidence ids, unknown gap, bibliographic fields, identifiers or citation-like text, invalid
   level/confidence) reject the item at once; SOFT flags (claim level above the evidence, unhedged causal
   language, prediction / alternative / falsification problems, protocol detail, self-labelled discovery,
   untraceable quotes, unsupported numbers) make it a candidate that must be revised.
2. **critic** (``hypothesis_critic``): ONE separate call reviewing all candidates (opaque ids, validated
   evidence with verified quotes, the gaps and the candidates; no bibliographic metadata). Structured
   verdicts per check; explanations are sanitised by code. The critic can only add problems: it can never
   accept what the validator flags.
3. **revision** (``hypothesis_revision``): ONE call, only for candidates with deterministic flags or
   substantive critic problems. Revisions must keep the gap and every cited evidence id except ids the
   critic flagged as unsupported, may add none, and pass the SAME deterministic validation again —
   otherwise the hypothesis is rejected (never silently accepted). The critic is not re-run on revisions.

Budget exhaustion / failures: if the critic cannot run, deterministically clean candidates are accepted but
marked "not stress-tested (critic not run)" and flagged candidates are rejected (``revision_not_run``); if
the revision cannot run or fails, every candidate that needed it is rejected.

Accepted hypotheses carry the fixed label :data:`~sciforge.hypothesis_validation.HYPOTHESIS_LABEL` (set by
code), a deterministic ``causality_statement`` below the causal-claim level, a deterministic
``source_quality_summary`` and a confidence capped by the deterministic ceiling (qualitative only).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import Field

from sciforge.boundary import canonical_json
from sciforge.hypothesis_validation import (
    FALSIFICATION_FIELDS,
    HYPOTHESIS_FIELDS,
    HYPOTHESIS_LABEL,
    HYPOTHESIS_NOTICE,
    CLAIM_LEVELS,
    ValidationContext,
    causality_statement,
    confidence_ceiling,
    final_confidence,
    has_bibliographic_text,
    revision_link_reasons,
    source_quality_summary,
    validate_hypothesis_fields,
)
from sciforge.llm.client import ModelMessage
from sciforge.llm.parsing import StrictModel, strict_json_schema
from sciforge.prompts_synthesis import (
    HYPOTHESES_INSTRUCTIONS,
    HYPOTHESIS_CRITIC_INSTRUCTIONS,
    HYPOTHESIS_PROMPT_VERSION,
    HYPOTHESIS_REVISION_INSTRUCTIONS,
)
from sciforge.stages.common import CallContext, StructuredResult, call_structured
from sciforge.stages.synthesis_checks import envelope_parser, model_evidence_view, safe_id, safe_rejection
from sciforge.stages.synthesis_common import SynthesisResult
from sciforge.stages.validation import DeterministicResult, combine_support, raw_item_sha256

STAGE = "hypotheses"
CRITIC_STAGE = "hypothesis_critic"
REVISION_STAGE = "hypothesis_revision"
MAX_HYPOTHESES = 5
MAX_EXPLANATION_CHARS = 400
MAX_PROBLEMS = 5
WITHHELD_EXPLANATION = "[explanation withheld: failed deterministic checks]"
CRITIC_CHECKS = ("evidence_supports_mechanism", "causal_language_exceeds_evidence", "distinct_from_evidence",
                 "prediction_measurable", "prediction_discriminates", "falsification_meaningful",
                 "ignored_contradictions_or_missing_evidence", "confidence_consistent")
VERDICTS = ("pass", "fail", "uncertain")
NO_REVISION = "no revision required"


# ------------------------------------------------------------------ model schemas


class ModelAlternative(StrictModel):
    explanation: str
    basis: Literal["evidence", "inference"]
    evidence_ids: list[str]


class ModelFalsification(StrictModel):
    manipulated_or_compared: str
    measured: str
    weakening_result: str
    supporting_result: str


class ModelHypothesis(StrictModel):
    hypothesis_id: str
    hypothesis: str
    evidence_ids: list[str] = Field(min_length=1)
    research_gap_id: str
    mechanistic_claim_level: Literal["observation", "association", "mechanistic_support", "causal_claim"]
    rationale: str
    prediction: str
    alternative_explanation: ModelAlternative
    falsification_test: ModelFalsification
    assumptions: list[str]
    evidence_limitations: list[str]
    confidence: Literal["low", "moderate", "high"]


class ModelHypothesisBatch(StrictModel):
    hypotheses: list[ModelHypothesis] = Field(max_length=MAX_HYPOTHESES)


class CriticCheck(StrictModel):
    verdict: Literal["pass", "fail", "uncertain"]
    explanation: str


class CriticChecks(StrictModel):
    evidence_supports_mechanism: CriticCheck
    causal_language_exceeds_evidence: CriticCheck
    distinct_from_evidence: CriticCheck
    prediction_measurable: CriticCheck
    prediction_discriminates: CriticCheck
    falsification_meaningful: CriticCheck
    ignored_contradictions_or_missing_evidence: CriticCheck
    confidence_consistent: CriticCheck


class CriticReview(StrictModel):
    hypothesis_id: str
    checks: CriticChecks
    unsupported_evidence_ids: list[str]
    substantive_problems: list[str]


class CriticBatch(StrictModel):
    reviews: list[CriticReview] = Field(max_length=MAX_HYPOTHESES)


HYPOTHESES_SCHEMA = strict_json_schema(ModelHypothesisBatch)
CRITIC_SCHEMA = strict_json_schema(CriticBatch)
assert tuple(ModelHypothesis.model_fields) == HYPOTHESIS_FIELDS
assert tuple(CriticChecks.model_fields) == CRITIC_CHECKS
assert tuple(ModelFalsification.model_fields) == FALSIFICATION_FIELDS
assert tuple(ModelHypothesis.model_fields["mechanistic_claim_level"].annotation.__args__) == CLAIM_LEVELS


# ------------------------------------------------------------------ result


@dataclass
class HypothesisEngineResult(SynthesisResult):
    """Hypotheses stage result incl. the critic and revision sub-stages."""

    critic_call: StructuredResult[Any] | None = None
    revision_call: StructuredResult[Any] | None = None
    critic_status: str = "not_run"          # completed | incomplete | failed | not_run
    critic_skip_reason: str | None = None
    revision_status: str = "not_run"        # not_required | completed | failed | not_run
    revision_skip_reason: str | None = None
    candidates: int = 0
    revised: int = 0
    extra_redaction_plans: list[dict[str, Any]] = field(default_factory=list)

    @property
    def stop(self) -> str | None:
        for call in (self.call, self.critic_call, self.revision_call):
            if call is not None and call.stop:
                return call.stop
        return None

    @property
    def redaction_plans(self) -> list[dict[str, Any]]:
        return [p for p in (self.redaction_plan, *self.extra_redaction_plans) if p]

    def attempts_by_stage(self) -> dict[str, int]:
        return {STAGE: self.call.attempts if self.call else 0,
                CRITIC_STAGE: self.critic_call.attempts if self.critic_call else 0,
                REVISION_STAGE: self.revision_call.attempts if self.revision_call else 0}


@dataclass
class _Candidate:
    hypothesis_id: str
    index: int
    raw: dict[str, Any]
    soft: list[dict[str, Any]]
    review: dict[str, Any] | None = None

    @property
    def needs_revision(self) -> bool:
        return bool(self.soft) or bool(self.review and self.review["has_substantive_problems"])


# ------------------------------------------------------------------ helpers


def gap_view(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ("gap_id", "gap_statement", "supporting_evidence_ids", "conflicting_evidence_ids", "why_unresolved",
            "confidence")
    return [{k: g[k] for k in keys} for g in gaps]


def hypothesis_evidence_view(accepted: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Model-visible evidence for the hypothesis engine: the v0.3 view plus the verified exact quote.

    The quote is a code-verified exact substring of the source abstract (identifier-like substrings
    redacted); no bibliographic metadata is included.
    """
    from sciforge.stages.identifiers import redact_identifiers

    by_id = {e["evidence_id"]: e for e in accepted}
    return [{**v, "verified_quote": redact_identifiers(str(by_id[v["evidence_id"]].get("quote") or ""))}
            for v in model_evidence_view(accepted)]


def _model_view(raw: Mapping[str, Any], hypothesis_id: str) -> dict[str, Any]:
    view = {k: raw.get(k) for k in HYPOTHESIS_FIELDS}
    view["hypothesis_id"] = hypothesis_id
    return view


def _safe_flag(r: Mapping[str, Any]) -> dict[str, Any]:
    return {k: r[k] for k in ("code", "detail", "field") if k in r}


def _sanitize(text: Any, known_ev: set[str], known_sources: set[str], known_gaps: set[str]) -> tuple[str, bool]:
    from sciforge.stages.synthesis_checks import check_text

    if not isinstance(text, str):
        return WITHHELD_EXPLANATION, True
    reasons, _ = check_text("critic", text, known_evidence=known_ev, known_sources=known_sources,
                            known_gaps=known_gaps, required=False)
    if reasons or has_bibliographic_text(text):
        return WITHHELD_EXPLANATION, True
    text = " ".join(text.split())
    return (text[:MAX_EXPLANATION_CHARS - 1] + "…") if len(text) > MAX_EXPLANATION_CHARS else text, False


def validate_review(raw: Any, candidates: Mapping[str, _Candidate], seen: set[str],
                    vctx: ValidationContext) -> tuple[dict[str, Any] | None, str]:
    """Code checks of one critic review. Returns (sanitised review or None, status code)."""
    if not isinstance(raw, dict) or set(raw) != {"hypothesis_id", "checks", "unsupported_evidence_ids",
                                                 "substantive_problems"}:
        return None, "malformed_review"
    hid = raw.get("hypothesis_id")
    if hid not in candidates:
        return None, "unknown_hypothesis_id"
    if hid in seen:
        return None, "duplicate_review"
    checks = raw.get("checks")
    if not isinstance(checks, dict) or set(checks) != set(CRITIC_CHECKS):
        return None, "malformed_review"
    known_ev, known_gaps = set(vctx.evidence_by_id), set(vctx.gaps_by_id)
    sanitized = False
    out_checks: dict[str, dict[str, str]] = {}
    for name in CRITIC_CHECKS:
        c = checks.get(name)
        if not isinstance(c, dict) or c.get("verdict") not in VERDICTS:
            return None, "malformed_review"
        text, withheld = _sanitize(c.get("explanation"), known_ev, vctx.known_sources, known_gaps)
        sanitized |= withheld
        out_checks[name] = {"verdict": c["verdict"], "explanation": text}
    cited = set(candidates[hid].raw.get("evidence_ids") or [])
    unsupported_raw = raw.get("unsupported_evidence_ids")
    unsupported = [e for e in unsupported_raw if isinstance(e, str) and e in cited] \
        if isinstance(unsupported_raw, list) else []
    problems_raw = raw.get("substantive_problems")
    problems: list[str] = []
    if isinstance(problems_raw, list):
        for p in problems_raw[:MAX_PROBLEMS]:
            text, withheld = _sanitize(p, known_ev, vctx.known_sources, known_gaps)
            sanitized |= withheld
            if text.strip():
                problems.append(text)
    has_problems = bool(problems) or any(c["verdict"] == "fail" for c in out_checks.values())
    return {"hypothesis_id": hid, "checks": out_checks, "unsupported_evidence_ids": unsupported,
            "substantive_problems": problems, "has_substantive_problems": has_problems,
            "sanitized": sanitized}, "ok"


def _inputs(stage: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    return {"synthesis_stage": stage, "input_evidence_ids": [e["evidence_id"] for e in evidence],
            "input_source_record_ids": sorted({e["source_record_id"] for e in evidence})}


def _plan(stage: str, evidence: list[dict[str, Any]], call: StructuredResult[Any],
          items: list[dict[str, Any]]) -> dict[str, Any]:
    return {**_inputs(stage, evidence), "audit_indices": call.audit_indices,
            "final_response_sha256": call.response_sha256 if call.ok else None, "items": items if call.ok else []}


def _stress_public(c: _Candidate, critic_status: str, revision_status: str) -> dict[str, Any]:
    review = c.review
    status = critic_status if review is not None or critic_status in ("not_run", "failed") else "incomplete"
    out: dict[str, Any] = {"critic_status": status, "revision_status": revision_status}
    if review is not None:
        out.update({"critic_findings": review["checks"], "substantive_problems": review["substantive_problems"],
                    "unsupported_evidence_ids": review["unsupported_evidence_ids"],
                    "critic_found_substantive_problems": review["has_substantive_problems"],
                    "critic_output_sanitized": review["sanitized"]})
    if status != "completed":
        out["note"] = "NOT stress-tested: the critic did not review this hypothesis"
    elif revision_status == "revised":
        out["note"] = "revised after critic/deterministic findings; the critic was not re-run on the revision"
    else:
        out["note"] = "critic review completed"
    return out


def _stress_safe(c: _Candidate, critic_status: str, revision_status: str) -> dict[str, Any]:
    """Rejected items: verdicts only (no model free text)."""
    out: dict[str, Any] = {"critic_status": critic_status if c.review is not None or critic_status != "completed"
                           else "incomplete", "revision_status": revision_status}
    if c.review is not None:
        out["critic_verdicts"] = {k: v["verdict"] for k, v in c.review["checks"].items()}
        out["critic_found_substantive_problems"] = c.review["has_substantive_problems"]
    return out


def build_record(c: _Candidate, raw: Mapping[str, Any], vctx: ValidationContext, check, *,
                 critic_status: str, revision_status: str, initial_flags: list[str]) -> dict[str, Any]:
    level = raw["mechanistic_claim_level"]
    gap = vctx.gaps_by_id.get(raw["research_gap_id"])
    ceiling, reasons = confidence_ceiling(check.evidence_ids, vctx.evidence_by_id, vctx.source_types, level=level,
                                          gap=gap)
    final = final_confidence(raw["confidence"], ceiling)
    alt = raw["alternative_explanation"]
    return {
        "id": c.hypothesis_id,
        "hypothesis_id": c.hypothesis_id,
        "model_hypothesis_id": safe_id(raw.get("hypothesis_id")),
        "label": HYPOTHESIS_LABEL,
        "notice": HYPOTHESIS_NOTICE,
        "hypothesis": raw["hypothesis"],
        "evidence_ids": list(check.evidence_ids),
        "research_gap_id": raw["research_gap_id"],
        "research_gap_ids": [raw["research_gap_id"]],
        "mechanistic_claim_level": level,
        "supported_claim_level": check.supported_level,
        "supported_claim_level_basis": check.supported_level_reasons,
        "causality_statement": causality_statement(level, check.supported_level),
        "rationale": raw["rationale"],
        "prediction": raw["prediction"],
        "alternative_explanation": {"explanation": alt["explanation"], "basis": alt["basis"],
                                    "evidence_ids": list(check.alternative_evidence_ids)},
        "falsification_test": {k: raw["falsification_test"][k] for k in FALSIFICATION_FIELDS},
        "assumptions": list(raw["assumptions"]),
        "evidence_limitations": list(raw["evidence_limitations"]),
        "source_quality_summary": source_quality_summary(check.evidence_ids, vctx.evidence_by_id, vctx.source_types),
        "confidence": final,
        "confidence_detail": {"proposed_by_model": raw["confidence"], "deterministic_ceiling": ceiling,
                              "final": final, "reasons": reasons, "capped": final != raw["confidence"],
                              "scale": "qualitative (low / moderate / high); no probabilities"},
        "source_record_ids": sorted({vctx.evidence_by_id[e]["source_record_id"] for e in check.evidence_ids}),
        "validation": {"deterministic": "passed", "status": "passed_deterministic_validation",
                       "initial_flags": initial_flags,
                       "precedence": "deterministic validation outranks the model's self-assessment and the critic"},
        "stress_test": _stress_public(c, critic_status, revision_status),
        "support": combine_support(DeterministicResult(passed=True), None),
    }


# ------------------------------------------------------------------ stage


def run_hypotheses_stage(ctx: CallContext, question_context: dict[str, Any], accepted_evidence: list[dict[str, Any]],
                         accepted_gaps: list[dict[str, Any]], known_sources: set[str], *,
                         source_types: Mapping[str, str] | None = None,
                         source_texts: Mapping[str, str] | None = None) -> HypothesisEngineResult:
    result = HypothesisEngineResult(stage=STAGE)
    vctx = ValidationContext(evidence_by_id={e["evidence_id"]: e for e in accepted_evidence},
                             gaps_by_id={g["gap_id"]: g for g in accepted_gaps}, known_sources=set(known_sources),
                             source_types=dict(source_types or {}), source_texts=dict(source_texts or {}))
    evidence_view = hypothesis_evidence_view(accepted_evidence)
    base_payload = {"question_definition": question_context, "evidence": evidence_view,
                    "research_gaps": gap_view(accepted_gaps)}

    call = call_structured(ctx, stage=STAGE, instructions=HYPOTHESES_INSTRUCTIONS,
                           messages=[ModelMessage("user", canonical_json(base_payload))],
                           schema_name="candidate_hypotheses", json_schema=HYPOTHESES_SCHEMA,
                           validate=envelope_parser("hypotheses"))
    result.call = call
    gen_items: list[dict[str, Any]] = []
    if not call.ok:
        result.status = "failed"
        result.redaction_plan = _plan(STAGE, accepted_evidence, call, [])
        return result
    result.status = "ok"

    def reject(raw: Any, index: int, reasons: list[dict[str, Any]], at: str, cand: _Candidate | None = None,
               *, revision_status: str = "not_run") -> None:
        record = safe_rejection(raw, index, reasons, HYPOTHESIS_FIELDS, id_field="hypothesis_id")
        record["rejected_at"] = at
        if cand is not None:
            record["hypothesis_id"] = cand.hypothesis_id
            record["stress_test"] = _stress_safe(cand, result.critic_status, revision_status)
        result.rejected.append(record)
        result.rejected_raw.append({"stage": STAGE if at == "generation" else REVISION_STAGE, "item_index": index,
                                    "raw_item_sha256": raw_item_sha256(raw), "model_output": raw})

    candidates: list[_Candidate] = []
    for index, raw in enumerate(call.value or []):
        if index >= MAX_HYPOTHESES:
            reject(raw, index, [{"code": "exceeds_item_limit", "detail": f"more than {MAX_HYPOTHESES} items"}],
                   "generation")
            gen_items.append({"item_index": index, "status": "rejected", "reason_codes": ["exceeds_item_limit"]})
            continue
        check = validate_hypothesis_fields(raw, vctx)
        if check.hard:
            reject(raw, index, check.reasons, "generation")
            gen_items.append({"item_index": index, "status": "rejected",
                              "reason_codes": [r["code"] for r in check.reasons],
                              "raw_item_sha256": raw_item_sha256(raw)})
            continue
        cand = _Candidate(f"hyp_{len(candidates) + 1:02d}", index, raw, check.soft)
        candidates.append(cand)
        gen_items.append({"item_index": index, "status": "candidate", "id": cand.hypothesis_id,
                          "flag_codes": [r["code"] for r in check.soft], "raw_item_sha256": raw_item_sha256(raw)})
    result.candidates = len(candidates)
    if not candidates:
        result.critic_status = "not_run"
        result.critic_skip_reason = "no_candidates"
        result.revision_status = "not_run"
        if result.rejected or call.repaired:
            result.redaction_plan = _plan(STAGE, accepted_evidence, call, gen_items)
        return result
    by_id = {c.hypothesis_id: c for c in candidates}

    # ---- critic (one batched call)
    critic_payload = {**base_payload, "candidate_hypotheses": [
        {**_model_view(c.raw, c.hypothesis_id), "deterministic_flags": [r["code"] for r in c.soft]}
        for c in candidates]}
    critic = call_structured(ctx, stage=CRITIC_STAGE, instructions=HYPOTHESIS_CRITIC_INSTRUCTIONS,
                             messages=[ModelMessage("user", canonical_json(critic_payload))],
                             schema_name="hypothesis_critic", json_schema=CRITIC_SCHEMA,
                             validate=envelope_parser("reviews"))
    result.critic_call = critic
    critic_items: list[dict[str, Any]] = []
    critic_needs_plan = critic.repaired or not critic.ok
    if critic.ok:
        seen: set[str] = set()
        for index, raw in enumerate(critic.value or []):
            review, code = validate_review(raw, by_id, seen, vctx)
            if review is None:
                critic_items.append({"item_index": index, "status": "ignored", "reason_codes": [code]})
                critic_needs_plan = True
                continue
            seen.add(review["hypothesis_id"])
            by_id[review["hypothesis_id"]].review = review
            critic_needs_plan |= review["sanitized"]
            critic_items.append({"item_index": index, "status": "accepted", "id": review["hypothesis_id"],
                                 "verdicts": {k: v["verdict"] for k, v in review["checks"].items()},
                                 "substantive": review["has_substantive_problems"]})
        result.critic_status = "completed" if all(c.review for c in candidates) else "incomplete"
    else:
        result.critic_status = "not_run" if critic.stop else "failed"
        result.critic_skip_reason = critic.stop or (critic.error or {}).get("error_type")

    # ---- revision (one batched call, only if needed)
    needing = [c for c in candidates if c.needs_revision]
    revised_raw: dict[str, tuple[int, Any]] = {}
    revision: StructuredResult[Any] | None = None
    if not needing:
        result.revision_status = "not_required"
    elif result.stop:
        result.revision_status = "not_run"
        result.revision_skip_reason = result.stop
    else:
        rev_payload = {**base_payload, "hypotheses_to_revise": [
            {"hypothesis": _model_view(c.raw, c.hypothesis_id),
             "deterministic_flags": [_safe_flag(r) for r in c.soft],
             "critic": ({"checks": c.review["checks"], "substantive_problems": c.review["substantive_problems"],
                         "unsupported_evidence_ids": c.review["unsupported_evidence_ids"]} if c.review else None)}
            for c in needing]}
        revision = call_structured(ctx, stage=REVISION_STAGE, instructions=HYPOTHESIS_REVISION_INSTRUCTIONS,
                                   messages=[ModelMessage("user", canonical_json(rev_payload))],
                                   schema_name="hypothesis_revision", json_schema=HYPOTHESES_SCHEMA,
                                   validate=envelope_parser("hypotheses"))
        result.revision_call = revision
        if revision.ok:
            result.revision_status = "completed"
            for index, raw in enumerate(revision.value or []):
                hid = raw.get("hypothesis_id") if isinstance(raw, dict) else None
                if isinstance(hid, str) and hid in by_id and hid not in revised_raw:
                    revised_raw[hid] = (index, raw)
        else:
            result.revision_status = "not_run" if revision.stop else "failed"
            result.revision_skip_reason = revision.stop or (revision.error or {}).get("error_type")

    # ---- final decision per candidate (validator outranks model and critic)
    rev_items: list[dict[str, Any]] = []
    for c in candidates:
        initial = [r["code"] for r in c.soft]
        if not c.needs_revision:
            check = validate_hypothesis_fields(c.raw, vctx)
            result.accepted.append(build_record(c, c.raw, vctx, check, critic_status=result.critic_status,
                                                revision_status=NO_REVISION, initial_flags=initial))
            continue
        if result.revision_status != "completed":
            code = "revision_not_run" if result.revision_status == "not_run" else "revision_failed"
            reasons = [*c.soft, {"code": code, "detail": "revision was required but could not be completed"}]
            if c.review and c.review["has_substantive_problems"] and not c.soft:
                reasons.insert(0, {"code": "critic_substantive_problems",
                                   "detail": "the critic reported substantive problems"})
            reject(c.raw, c.index, reasons, "unresolved", c, revision_status=result.revision_status)
            continue
        if c.hypothesis_id not in revised_raw:
            reject(c.raw, c.index, [*c.soft, {"code": "revision_missing",
                                              "detail": "no revised version was returned"}], "revision", c,
                   revision_status="missing")
            rev_items.append({"id": c.hypothesis_id, "status": "missing"})
            continue
        rindex, rraw = revised_raw[c.hypothesis_id]
        check = validate_hypothesis_fields(rraw, vctx)
        extra: list[dict[str, Any]] = []
        if not check.hard:
            unsupported = c.review["unsupported_evidence_ids"] if c.review else []
            extra += revision_link_reasons(check.evidence_ids, list(c.raw.get("evidence_ids") or []), unsupported)
            if rraw.get("research_gap_id") != c.raw.get("research_gap_id"):
                extra.append({"code": "revision_changed_gap", "detail": "revision changed the research gap",
                              "field": "research_gap_id"})
        reasons = [*check.reasons, *extra]
        if reasons:
            reasons.append({"code": "unresolved_after_revision",
                            "detail": "the revised hypothesis still fails deterministic validation"})
            reject(rraw, rindex, reasons, "revision", c, revision_status="revised_but_rejected")
            rev_items.append({"item_index": rindex, "id": c.hypothesis_id, "status": "rejected",
                              "reason_codes": [r["code"] for r in reasons]})
            continue
        result.revised += 1
        result.accepted.append(build_record(c, rraw, vctx, check, critic_status=result.critic_status,
                                            revision_status="revised", initial_flags=initial))
        rev_items.append({"item_index": rindex, "id": c.hypothesis_id, "status": "accepted",
                          "raw_item_sha256": raw_item_sha256(rraw)})
    result.accepted.sort(key=lambda h: h["hypothesis_id"])
    # The critic and revision REQUESTS carry candidate text. If any candidate was rejected after generation,
    # those stored prompts are redacted too, so rejected text never survives in model_calls.json.
    candidate_rejected = any(r.get("rejected_at") in ("revision", "unresolved") for r in result.rejected)
    if critic_needs_plan or candidate_rejected:
        result.extra_redaction_plans.append({**_plan(CRITIC_STAGE, accepted_evidence, critic, critic_items),
                                             "redact_request_messages": candidate_rejected})
    if revision is not None and (revision.repaired or not revision.ok or candidate_rejected
                                 or any(i["status"] != "accepted" for i in rev_items)
                                 or len(revision.value or []) != len(rev_items)):
        result.extra_redaction_plans.append({**_plan(REVISION_STAGE, accepted_evidence, revision, rev_items),
                                             "redact_request_messages": candidate_rejected})
    if result.rejected or call.repaired or needing:
        result.redaction_plan = _plan(STAGE, accepted_evidence, call, gen_items)
    return result


def hypotheses_json(result: SynthesisResult) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    if isinstance(result, HypothesisEngineResult):
        extra = {
            "engine": "v0.4 hypothesis reliability engine (generation -> critic -> revision)",
            "critic": {"status": result.critic_status, "skip_reason": result.critic_skip_reason,
                       "attempts": result.critic_call.attempts if result.critic_call else 0,
                       "repair_attempted": result.critic_call.repaired if result.critic_call else False,
                       "errors": result.critic_call.errors if result.critic_call and not result.critic_call.ok
                       else []},
            "revision": {"status": result.revision_status, "skip_reason": result.revision_skip_reason,
                         "revised": result.revised,
                         "attempts": result.revision_call.attempts if result.revision_call else 0,
                         "errors": result.revision_call.errors if result.revision_call and not result.revision_call.ok
                         else []},
            "candidates": result.candidates,
            "attempts_by_stage": result.attempts_by_stage(),
        }
    return {**result.base_json(), "prompt_version": HYPOTHESIS_PROMPT_VERSION,
            "label_rule": f'every accepted hypothesis carries the fixed label "{HYPOTHESIS_LABEL}" (set by code)',
            "notice": HYPOTHESIS_NOTICE, **extra,
            "accepted": result.accepted, "rejected": result.rejected}
