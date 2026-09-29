"""v0.3 model pipeline (Milestone 2): source texts → question definition → evidence extraction.

"V0.2 remains the authority for source identity. V0.3 can interpret verified
sources, but it cannot create citations."

Not wired into the CLI yet (no ``--with-model``); call :func:`run_model_pipeline`
with the v0.2 records and verification results. Files are written to the run
directory immediately after each stage (secret-scrubbed via the v0.2 writer):

* ``source_texts.json`` — per-record retrieval status, access level, hash, cap
* ``question.json``     — question definition (or failure + fallback)
* ``evidence.json``     — accepted items, rejected items with reasons, counts
* ``model_calls.json``  — M1 per-attempt audit. For extraction calls whose output
  had rejected items, failed validation, or needed a repair, the stored model
  output (``response_text`` and echoed assistant messages) is replaced by a
  structured safe-diagnostics summary (``redaction_mode: safe_diagnostics``):
  accepted items by evidence_id + hash, rejected items by reason codes, field
  names/lengths, hashes and numeric diagnostics only — no model free text.
* ``debug/rejected_raw.json`` — ONLY with ``SCIFORGE_DEBUG_KEEP_REJECTED_RAW=true``
  (default off): raw rejected items and the unredacted outputs, local debugging only.

No gaps, report, or claim-entailment stage in this milestone. v0.2 records
and verification results are only read, never modified.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from sciforge import __version__
from sciforge.config import ModelSettings, Settings, _parse_bool
from sciforge.http_utils import HttpFetcher, RateLimiter
from sciforge.llm.audit import ModelCallAudit
from sciforge.llm.budget import BudgetTracker, RetryPolicy
from sciforge.llm.client import ModelClient
from sciforge.logging_utils import RunLog, iso_utc, utc_now
from sciforge.models import Record, VerificationResult
from sciforge.pipeline import _write_json, make_run_dir
from sciforge.sourcetext import SourceTextBatch, max_source_chars_from_env, prepare_source_texts
from sciforge.stages.common import CallContext
from sciforge.stages.extraction import ExtractionResult, run_extraction_stage
from sciforge.stages.question import QuestionStageResult, run_question_stage

MODEL_LAYER_MILESTONE = "v0.3-m2"
DEBUG_KEEP_REJECTED_RAW_ENV = "SCIFORGE_DEBUG_KEEP_REJECTED_RAW"
DEBUG_DIR = "debug"
DEBUG_REJECTED_FILE = "rejected_raw.json"


def debug_keep_rejected_raw_from_env(environ: dict[str, str] | None = None) -> bool:
    """``SCIFORGE_DEBUG_KEEP_REJECTED_RAW`` (default false; 1/0/true/false/yes/no/on/off)."""
    import os

    env = os.environ if environ is None else environ
    return _parse_bool(env, DEBUG_KEEP_REJECTED_RAW_ENV, False)


REDACTION_MODE = "safe_diagnostics"


def _response_structure(text: str) -> dict[str, Any]:
    """Structure-only description of an unvalidated response (no content)."""
    import json

    from sciforge.stages.validation import _safe_key

    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return {"parse_status": "invalid_json"}
    if not isinstance(data, dict):
        return {"parse_status": "not_an_object", "json_type": type(data).__name__}
    items = data.get("items")
    return {"parse_status": "object", "top_level_keys": sorted({_safe_key(k) for k in data}),
            "item_count": len(items) if isinstance(items, list) else None}


def _safe_summary(entry: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    text = entry.get("response_text") or ""
    summary: dict[str, Any] = {"redaction_mode": REDACTION_MODE, "call_record_id": plan["record_id"],
                               "response_sha256": entry.get("response_sha256"), "response_length": len(text)}
    if plan["final_response_sha256"] and entry.get("response_sha256") == plan["final_response_sha256"]:
        summary.update({"parse_status": "validated_envelope", "items": plan["items"]})
    else:
        summary.update(_response_structure(text))
        summary["note"] = "response failed validation; content not stored"
    return summary


def _redact_audit_entries(audit: ModelCallAudit, plans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace stored model output of the planned audit entries with safe structured summaries (in place).

    ``response_text`` becomes a JSON safe-diagnostics summary; echoed assistant messages in (repair) requests
    become hash + length. Returns the originals for the opt-in debug file.
    """
    import hashlib
    import json

    originals: list[dict[str, Any]] = []
    for plan in plans:
        for i in plan["audit_indices"]:
            if not 0 <= i < len(audit.entries):
                continue
            entry = audit.entries[i]
            if entry.get("model_output_redacted"):
                continue
            original: dict[str, Any] = {"audit_index": entry.get("index"), "stage": entry.get("stage"),
                                        "call_record_id": plan["record_id"],
                                        "response_sha256": entry.get("response_sha256")}
            if entry.get("response_text") is not None:
                original["response_text"] = entry["response_text"]
                entry["response_text"] = json.dumps(_safe_summary(entry, plan), ensure_ascii=False, sort_keys=True)
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
class ModelPipelineResult:
    run_dir: Path
    source_texts: SourceTextBatch
    question: QuestionStageResult
    evidence: ExtractionResult
    budget: dict[str, Any]
    files: dict[str, Path]


def run_model_pipeline(
    question: str,
    records: Sequence[Record],
    verification: Sequence[VerificationResult],
    *,
    model_client: ModelClient,
    settings: Settings,
    model_settings: ModelSettings | None = None,
    tracker: BudgetTracker | None = None,
    retry: RetryPolicy | None = None,
    audit: ModelCallAudit | None = None,
    run_dir: str | Path | None = None,
    output_dir: str | Path = "runs",
    http_client: httpx.Client | None = None,
    fetcher: HttpFetcher | None = None,
    include_partially_verified: bool | None = None,
    max_source_chars: int | None = None,
    debug_keep_rejected_raw: bool | None = None,
    pubmed_limiter: RateLimiter | None = None,
    crossref_limiter: RateLimiter | None = None,
    sleep: Callable[[float], None] | None = None,
    now: Callable[[], datetime] = utc_now,
) -> ModelPipelineResult:
    """Run S0 (source texts), S1 (question) and S2 (extraction + validation).

    Budget: pass ``tracker`` or ``model_settings`` (limits/prices/retry policy
    are derived from it). Eligibility: ``include_partially_verified`` defaults
    to ``model_settings.include_partially_verified`` (else False).
    ``max_source_chars`` defaults to ``SCIFORGE_MAX_SOURCE_CHARS`` (4000).
    ``debug_keep_rejected_raw`` defaults to ``SCIFORGE_DEBUG_KEEP_REJECTED_RAW``
    (false); when true, raw rejected output goes to ``debug/rejected_raw.json`` only.
    Never raises for network/model failures; they are recorded in the files.
    """
    query = question.strip()
    if not query:
        raise ValueError("research question must not be empty")
    if tracker is None:
        if model_settings is None:
            raise ValueError("pass a BudgetTracker or ModelSettings (budget limits are required)")
        tracker = BudgetTracker(model_settings.budget_limits(), model_settings.price_table())
    if retry is None:
        retry = model_settings.retry_policy() if model_settings is not None else RetryPolicy()
    if include_partially_verified is None:
        include_partially_verified = bool(model_settings and model_settings.include_partially_verified)
    if max_source_chars is None:
        max_source_chars = max_source_chars_from_env()
    if debug_keep_rejected_raw is None:
        debug_keep_rejected_raw = debug_keep_rejected_raw_from_env()  # ConfigError on invalid value
    secrets = list(settings.secret_values()) + (model_settings.secret_values() if model_settings else [])
    if audit is None:
        audit = ModelCallAudit(store_prompts=model_settings.store_prompts if model_settings else True,
                               secrets=secrets)
    sleep_fn = sleep or time.sleep

    target = Path(run_dir) if run_dir is not None else make_run_dir(Path(output_dir), now())
    target.mkdir(parents=True, exist_ok=True)
    files = {name: target / f"{name}.json" for name in ("source_texts", "question", "evidence", "model_calls")}
    header = {"sciforge_version": __version__, "model_layer": MODEL_LAYER_MILESTONE, "question": query}
    debug_note = {"debug_keep_rejected_raw": bool(debug_keep_rejected_raw)}

    # ---- S0: source texts (code only)
    own_client = http_client is None and fetcher is None
    http = http_client or (None if fetcher else httpx.Client(follow_redirects=True, timeout=settings.timeout_seconds))
    try:
        if fetcher is None:
            fetcher = HttpFetcher(http, settings, RunLog(settings.secret_values()), sleep=sleep_fn)  # type: ignore[arg-type]
        batch = prepare_source_texts(records, verification, fetcher=fetcher, tracker=tracker,
                                     include_partially_verified=include_partially_verified,
                                     max_source_chars=max_source_chars, pubmed_limiter=pubmed_limiter,
                                     crossref_limiter=crossref_limiter)
    finally:
        if own_client and http is not None:
            http.close()
    _write_json(files["source_texts"], {**header, "generated_at": iso_utc(now()), **batch.to_json()}, secrets)

    ctx = CallContext(client=model_client, tracker=tracker, audit=audit, retry=retry, sleep=sleep_fn, now=now)

    # ---- S1: question definition (question only)
    q = run_question_stage(ctx, query)
    _write_json(files["question"], {**header, "generated_at": iso_utc(now()), **q.to_json()}, secrets)
    audit.write(files["model_calls"])

    # ---- S2: extraction + deterministic validation
    if q.definition is not None:
        context = q.definition.model_dump()
    else:
        context = {"research_question": query}
    if q.call.stop:
        evidence = ExtractionResult(question_definition_used=False, stop=q.call.stop)
        for s in batch.usable:
            evidence.calls.append({"record_id": s.record_id, "status": "skipped", "reason": q.call.stop,
                                   "logical_calls": 0, "attempts": 0, "repair_attempted": False})
    elif not batch.usable:
        evidence = ExtractionResult(question_definition_used=q.definition is not None)
    else:
        evidence = run_extraction_stage(ctx, context, batch.sources,
                                        question_definition_used=q.definition is not None)
    originals = _redact_audit_entries(audit, evidence.redaction_plans)
    if debug_keep_rejected_raw and (evidence.rejected_raw or originals):
        debug_dir = target / DEBUG_DIR
        debug_dir.mkdir(exist_ok=True)
        files["debug_rejected_raw"] = debug_dir / DEBUG_REJECTED_FILE
        _write_json(files["debug_rejected_raw"], {
            **header,
            "warning": ("LOCAL DEBUG FILE (SCIFORGE_DEBUG_KEEP_REJECTED_RAW=true). Contains raw, unvalidated model "
                        "output, possibly with invented bibliographic data. Never cite, publish or share it."),
            "rejected_items": evidence.rejected_raw,
            "unredacted_model_outputs": originals,
        }, secrets)
    budget = tracker.summary()
    _write_json(files["evidence"], {**header, "generated_at": iso_utc(now()),
                                    "note": "No source text available: extraction not run." if not batch.usable else None,
                                    **evidence.to_json(), "budget": budget, **debug_note,
                                    "model_calls_redacted_entries": len(set(evidence.redact_audit_indices))},
                secrets)
    audit.write(files["model_calls"])
    return ModelPipelineResult(run_dir=target, source_texts=batch, question=q, evidence=evidence, budget=budget,
                               files=files)
