"""v0.3 model investigation (Milestone 3): M2 pipeline → gaps → hypotheses → report.

"V0.2 remains the authority for source identity. V0.3 can interpret verified
sources, but it cannot create citations."

:func:`run_model_investigation` wraps :func:`sciforge.model_pipeline.run_model_pipeline`
(S0 source texts → S1 question → S2 extraction + validation, unchanged) and then
runs, sharing the same BudgetTracker, RetryPolicy and M1 ModelCallAudit:

* S4a gaps (``gaps.json``), S4b hypotheses (``hypotheses.json``),
* S5 report narrative + deterministic template (``report.md``, ``report_validation.json``).

Privacy (same rules as M2): rejected gaps / hypotheses / paragraphs are stored as
safe diagnostics only; for every M3 call with rejected items, a repair, or a
failure, ``model_calls.json`` keeps a structured safe summary instead of the
model output (``redaction_mode: safe_diagnostics``). M3 summaries name the stage in
``synthesis_stage`` and the inputs in ``input_evidence_ids`` /
``input_source_record_ids``; they carry no ``call_record_id`` (that field is reserved
for a single source record id, as in M2 extraction entries). Raw rejected output only in ``debug/rejected_raw.json`` with
``SCIFORGE_DEBUG_KEEP_REJECTED_RAW=true``. Not wired into the CLI.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from sciforge import __version__
from sciforge.config import ModelSettings, Settings
from sciforge.llm.audit import ModelCallAudit
from sciforge.llm.budget import BudgetTracker, RetryPolicy
from sciforge.llm.client import ModelClient
from sciforge.logging_utils import iso_utc, redact_text, utc_now
from sciforge.model_pipeline import (
    DEBUG_DIR,
    DEBUG_REJECTED_FILE,
    ModelPipelineResult,
    debug_keep_rejected_raw_from_env,
    run_model_pipeline,
)
from sciforge.models import Record, VerificationResult
from sciforge.pipeline import _write_json, make_run_dir
from sciforge.stages.common import CallContext
from sciforge.stages.gaps import gaps_json, run_gaps_stage
from sciforge.stages.hypotheses import hypotheses_json, run_hypotheses_stage
from sciforge.stages.report import build_report, run_narrative_stage
from sciforge.stages.synthesis_common import SynthesisResult

MODEL_LAYER_MILESTONE = "v0.3-m3"
REDACTION_MODE = "safe_diagnostics"


def _synthesis_response_structure(text: str) -> dict[str, Any]:
    """Structure-only description of an unvalidated M3 response (no content)."""
    from sciforge.stages.validation import _safe_key

    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return {"parse_status": "invalid_json"}
    if not isinstance(data, dict):
        return {"parse_status": "not_an_object", "json_type": type(data).__name__}
    lists = [v for v in data.values() if isinstance(v, list)]
    return {"parse_status": "object", "top_level_keys": sorted({_safe_key(k) for k in data}),
            "item_count": len(lists[0]) if len(lists) == 1 else None}


def _synthesis_summary(entry: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    text = entry.get("response_text") or ""
    summary: dict[str, Any] = {
        "redaction_mode": REDACTION_MODE, "synthesis_stage": plan["synthesis_stage"],
        "input_evidence_ids": plan["input_evidence_ids"], "input_source_record_ids": plan["input_source_record_ids"],
        "response_sha256": entry.get("response_sha256"), "response_length": len(text)}
    if plan["final_response_sha256"] and entry.get("response_sha256") == plan["final_response_sha256"]:
        summary.update({"parse_status": "validated_envelope", "items": plan["items"]})
    else:
        summary.update(_synthesis_response_structure(text))
        summary["note"] = "response failed validation; content not stored"
    return summary


def _redact_synthesis_entries(audit: ModelCallAudit, plans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """M3 counterpart of M2's ``_redact_audit_entries`` (which is left untouched for extraction entries).

    Same safe-diagnostics treatment (response_text → structured summary, echoed assistant messages →
    hash + length, ``model_output_redacted``/``redaction_mode`` flags) but with M3 naming: the stage goes
    in ``synthesis_stage`` and inputs in ``input_evidence_ids`` / ``input_source_record_ids``; there is no
    ``call_record_id``. Returns the originals for the opt-in debug file.
    """
    import hashlib

    originals: list[dict[str, Any]] = []
    for plan in plans:
        for i in plan["audit_indices"]:
            if not 0 <= i < len(audit.entries):
                continue
            entry = audit.entries[i]
            if entry.get("model_output_redacted"):
                continue
            original: dict[str, Any] = {"audit_index": entry.get("index"), "stage": entry.get("stage"),
                                        "synthesis_stage": plan["synthesis_stage"],
                                        "response_sha256": entry.get("response_sha256")}
            if entry.get("response_text") is not None:
                original["response_text"] = entry["response_text"]
                entry["response_text"] = json.dumps(_synthesis_summary(entry, plan), ensure_ascii=False,
                                                    sort_keys=True)
            request = entry.get("request")
            if isinstance(request, dict) and isinstance(request.get("messages"), list):
                echoed = []
                for m in request["messages"]:
                    if isinstance(m, dict) and m.get("role") == "assistant":
                        content = m.get("content") or ""
                        echoed.append(content)
                        m["content"] = json.dumps({
                            "redaction_mode": REDACTION_MODE, "echoed_model_output": "[redacted]",
                            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(), "length": len(content),
                        }, sort_keys=True)
                if echoed:
                    original["assistant_messages"] = echoed
            entry["model_output_redacted"] = True
            entry["redaction_mode"] = REDACTION_MODE
            if len(original) > 4:
                originals.append(original)
    return originals


@dataclass
class InvestigationModelResult:
    run_dir: Path
    base: ModelPipelineResult
    gaps: SynthesisResult
    hypotheses: SynthesisResult
    narrative: SynthesisResult
    report_validation: dict[str, Any]
    budget: dict[str, Any]
    files: dict[str, Path]


def _skipped(stage: str, reason: str) -> SynthesisResult:
    return SynthesisResult(stage=stage, status="skipped", skip_reason=reason)


def run_model_investigation(
    question: str,
    records: Sequence[Record],
    verification: Sequence[VerificationResult],
    *,
    model_client: ModelClient,
    settings: Settings,
    model_settings: ModelSettings | None = None,
    search_summary: Mapping[str, Any] | None = None,
    tracker: BudgetTracker | None = None,
    retry: RetryPolicy | None = None,
    run_dir: str | Path | None = None,
    output_dir: str | Path = "runs",
    debug_keep_rejected_raw: bool | None = None,
    sleep: Callable[[float], None] | None = None,
    now: Callable[[], datetime] = utc_now,
    **m2_kwargs: Any,
) -> InvestigationModelResult:
    """Full mocked-model investigation. ``search_summary`` is the v0.2 ``summary.json`` payload (section B).

    Extra keyword arguments go to :func:`run_model_pipeline` (http_client, fetcher, limiters, ...).
    Never raises for network/model failures; they are recorded in the files.
    """
    if tracker is None:
        if model_settings is None:
            raise ValueError("pass a BudgetTracker or ModelSettings (budget limits are required)")
        tracker = BudgetTracker(model_settings.budget_limits(), model_settings.price_table())
    if retry is None:
        retry = model_settings.retry_policy() if model_settings is not None else RetryPolicy()
    if debug_keep_rejected_raw is None:
        debug_keep_rejected_raw = debug_keep_rejected_raw_from_env()
    secrets = list(settings.secret_values()) + (model_settings.secret_values() if model_settings else [])
    audit = ModelCallAudit(store_prompts=model_settings.store_prompts if model_settings else True, secrets=secrets)
    sleep_fn = sleep or time.sleep
    target = Path(run_dir) if run_dir is not None else make_run_dir(Path(output_dir), now())

    base = run_model_pipeline(question, records, verification, model_client=model_client, settings=settings,
                              model_settings=model_settings, tracker=tracker, retry=retry, audit=audit,
                              run_dir=target, debug_keep_rejected_raw=debug_keep_rejected_raw, sleep=sleep_fn,
                              now=now, **m2_kwargs)
    files = dict(base.files)
    for name in ("gaps", "hypotheses", "report_validation"):
        files[name] = target / f"{name}.json"
    files["report"] = target / "report.md"
    header = {"sciforge_version": __version__, "model_layer": MODEL_LAYER_MILESTONE, "question": question.strip()}

    ctx = CallContext(client=model_client, tracker=tracker, audit=audit, retry=retry, sleep=sleep_fn, now=now)
    q = base.question
    context = q.definition.model_dump() if q.definition is not None else {"research_question": question.strip()}
    accepted_evidence = list(base.evidence.accepted)
    known_sources = {s.record_id for s in base.source_texts.usable}
    stop = q.call.stop or base.evidence.stop

    # ---- S4a gaps
    if stop:
        gaps = _skipped("gaps", stop)
    elif not accepted_evidence:
        gaps = _skipped("gaps", "no_accepted_evidence")
    else:
        gaps = run_gaps_stage(ctx, context, accepted_evidence, known_sources)
        stop = gaps.stop
    _write_json(files["gaps"], {**header, "generated_at": iso_utc(now()), **gaps_json(gaps)}, secrets)

    # ---- S4b hypotheses
    if stop:
        hyps = _skipped("hypotheses", stop)
    elif not gaps.accepted:
        hyps = _skipped("hypotheses", "no_accepted_gaps")
    else:
        hyps = run_hypotheses_stage(ctx, context, accepted_evidence, gaps.accepted, known_sources)
        stop = hyps.stop
    _write_json(files["hypotheses"], {**header, "generated_at": iso_utc(now()), **hypotheses_json(hyps)}, secrets)

    # ---- S5 report narrative (model) + deterministic template (code)
    if stop:
        narrative = _skipped("report", stop)
    elif not accepted_evidence:
        narrative = _skipped("report", "no_accepted_evidence")
    else:
        narrative = run_narrative_stage(ctx, context, accepted_evidence, gaps.accepted, hyps.accepted, known_sources)

    plans = [r.redaction_plan for r in (gaps, hyps, narrative) if r.redaction_plan]
    originals = _redact_synthesis_entries(audit, plans)
    audit.write(files["model_calls"])

    notes: list[str] = []
    if q.definition is None:
        notes.append("Question definition failed; the raw question was used.")
    if stop:
        notes.append(f"Model stages stopped early: {stop}. Later sections are code-only.")
    source_texts = json.loads(files["source_texts"].read_text(encoding="utf-8"))
    evidence = json.loads(files["evidence"].read_text(encoding="utf-8"))
    report, validation = build_report(
        question=question.strip(), question_definition=q.definition.model_dump() if q.definition else None,
        search_summary=search_summary, source_texts=source_texts, evidence=evidence, gaps=gaps, hypotheses=hyps,
        narrative=narrative, records=records, verification=verification, stage_notes=notes)
    files["report"].write_text(redact_text(report, secrets), encoding="utf-8")
    budget = tracker.summary()
    _write_json(files["report_validation"], {**header, "generated_at": iso_utc(now()), **validation,
                                             "budget": budget,
                                             "debug_keep_rejected_raw": bool(debug_keep_rejected_raw)}, secrets)

    raw = [item for r in (gaps, hyps, narrative) for item in r.rejected_raw]
    if debug_keep_rejected_raw and (raw or originals):
        debug_file = target / DEBUG_DIR / DEBUG_REJECTED_FILE
        debug_file.parent.mkdir(exist_ok=True)
        existing = json.loads(debug_file.read_text(encoding="utf-8")) if debug_file.exists() else {
            **header,
            "warning": ("LOCAL DEBUG FILE (SCIFORGE_DEBUG_KEEP_REJECTED_RAW=true). Contains raw, unvalidated model "
                        "output, possibly with invented bibliographic data. Never cite, publish or share it."),
            "rejected_items": [], "unredacted_model_outputs": []}
        existing["synthesis_rejected_items"] = raw
        existing["synthesis_unredacted_model_outputs"] = originals
        _write_json(debug_file, existing, secrets)
        files["debug_rejected_raw"] = debug_file
    return InvestigationModelResult(run_dir=target, base=base, gaps=gaps, hypotheses=hyps, narrative=narrative,
                                    report_validation=validation, budget=budget, files=files)
