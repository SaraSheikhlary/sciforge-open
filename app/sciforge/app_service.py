"""Application interface for the SciForge web app (Streamlit UI calls ONLY this module).

The UI (``streamlit_app.py`` at the repository root) contains no scientific
logic. It collects inputs, calls :func:`run_web_investigation` and renders the
returned :class:`WebInvestigationResult`. This module adds no business logic of
its own either: it validates inputs, picks the model client, runs the existing
pipelines and reshapes their results for display.

* **Demo Mode** (default): bundled SYNTHETIC v0.2-shaped records
  (:mod:`sciforge.demo_data`) + a scripted :class:`~sciforge.llm.fake.FakeModelClient`,
  run through the real :func:`~sciforge.investigation_pipeline.run_model_investigation`
  (source texts, deterministic validation, gaps, hypotheses, report). Fully
  offline: abstracts are served by an ``httpx.MockTransport``; no API key.
* **Live Mode**: v0.2 :func:`~sciforge.pipeline.run_investigation` (PubMed +
  Crossref, verification) followed by ``run_model_investigation`` with the
  existing :class:`~sciforge.llm.xai.XAIClient` and the existing attempt / token /
  spend budgets from :class:`~sciforge.config.ModelSettings`. Enabled only when
  the deployment gate ``SCIFORGE_LIVE_ENABLED`` is exactly ``true`` (default
  false) AND both ``XAI_API_KEY`` and ``XAI_MODEL`` are present (environment
  variables first, then the optional ``secrets`` mapping, e.g. ``st.secrets``).
  The gate is enforced here as well as in the UI (live runs are refused). Presence is
  checked; values are never returned, rendered or logged. NOT validated against
  the real API yet.

Privacy
-------
* Run artifacts (the pipelines always write JSON/Markdown files) go to a fresh
  directory under the system temp dir (never the repository) that is deleted
  before :func:`run_web_investigation` returns. Questions are not persisted.
* Every string in the result passes :func:`guard_display` — secret redaction
  (:func:`sciforge.logging_utils.redact_text`), filesystem-path masking and
  withholding of any raw rejected model output (needles from
  :func:`sciforge.output_guard.rejected_text_values`). The run directory is also
  checked with :func:`sciforge.output_guard.find_leaks` before deletion.
* Citations come only from :func:`sciforge.stages.report.render_citation` /
  :func:`~sciforge.stages.report.unresolved_marker` over the v0.2 records.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import httpx

from sciforge.config import ConfigError, ModelSettings, Settings
from sciforge.investigation_pipeline import InvestigationModelResult, run_model_investigation
from sciforge.llm.client import ModelClient, ModelConfigError, ModelRequest, ModelResponse
from sciforge.logging_utils import redact_text, utc_now
from sciforge.models import Record, SearchOutcome, VerificationResult
from sciforge.output_guard import find_leaks, rejected_text_values
from sciforge.stages.report import SECTION_TITLES, render_citation, unresolved_marker

__all__ = [
    "MAX_QUESTION_CHARS", "MAX_SOURCES_LIMIT", "MIN_QUESTION_CHARS", "MIN_SOURCES", "MODE_DEMO", "MODE_LIVE",
    "PROGRESS_STAGES", "InvestigationRequest", "LiveAvailability", "ProgressEvent", "WebInvestigationResult",
    "guard_display", "live_availability", "live_gate_enabled", "render_sources", "run_web_investigation", "validate_request",
]

MODE_DEMO = "demo"
MODE_LIVE = "live"
MIN_QUESTION_CHARS = 10
MAX_QUESTION_CHARS = 2000
MIN_SOURCES = 1
MAX_SOURCES_LIMIT = 20
DEFAULT_MAX_SOURCES = 5
MIN_YEAR = 1800
MAX_YEAR = 2100
LIVE_CREDENTIAL_NAMES = ("XAI_API_KEY", "XAI_MODEL")
# Deployment gate: Live Mode is off unless this is exactly "true" (case-insensitive, trimmed). Default false.
LIVE_ENABLED_NAME = "SCIFORGE_LIVE_ENABLED"
# Every name the app may read from st.secrets (env vars take precedence for each).
LIVE_SETTING_NAMES = (LIVE_ENABLED_NAME, *LIVE_CREDENTIAL_NAMES)
WITHHELD = "[withheld by privacy guard]"
PATH_MASK = "[path]"
TEMP_PREFIX = "sciforge-web-"

# (key, UI label) in display order.
PROGRESS_STAGES: tuple[tuple[str, str], ...] = (
    ("define", "Defining question"),
    ("search", "Searching literature"),
    ("verify", "Verifying sources"),
    ("extract", "Extracting evidence"),
    ("check", "Checking evidence"),
    ("gaps", "Identifying research gaps"),
    ("hypotheses", "Generating hypotheses"),
    ("report", "Building report"),
)
_STAGE_FOR_MODEL = {"question": "define", "extraction": "extract", "gaps": "gaps", "hypotheses": "hypotheses",
                    "report": "report"}


# ------------------------------------------------------------------ inputs


@dataclass(frozen=True)
class InvestigationRequest:
    question: str
    from_year: int | None = None
    to_year: int | None = None
    max_sources: int = DEFAULT_MAX_SOURCES
    mode: str = MODE_DEMO


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_request(req: InvestigationRequest) -> list[str]:
    """Human-readable input errors (empty list = valid). Never echoes the question back."""
    errors: list[str] = []
    q = req.question.strip() if isinstance(req.question, str) else ""
    if not q:
        errors.append("Please enter a research question.")
    elif len(q) < MIN_QUESTION_CHARS:
        errors.append(f"The research question is too short (minimum {MIN_QUESTION_CHARS} characters).")
    elif len(q) > MAX_QUESTION_CHARS:
        errors.append(f"The research question is too long (maximum {MAX_QUESTION_CHARS} characters).")
    for name, value in (("Start year", req.from_year), ("End year", req.to_year)):
        if value is not None and (not _is_int(value) or not MIN_YEAR <= value <= MAX_YEAR):
            errors.append(f"{name} must be a whole year between {MIN_YEAR} and {MAX_YEAR}.")
    if _is_int(req.from_year) and _is_int(req.to_year) and req.from_year > req.to_year:
        errors.append("Invalid date range: the start year is later than the end year.")
    if not _is_int(req.max_sources) or not MIN_SOURCES <= req.max_sources <= MAX_SOURCES_LIMIT:
        errors.append(f"Maximum sources must be a whole number between {MIN_SOURCES} and {MAX_SOURCES_LIMIT}.")
    if req.mode not in (MODE_DEMO, MODE_LIVE):
        errors.append("Mode must be Demo or Live.")
    return errors


# ------------------------------------------------------------------ live credentials (presence only)


def _clean(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _credential(name: str, environ: Mapping[str, str], secrets: Mapping[str, Any] | None) -> str | None:
    """Environment variable first, then the secrets mapping (e.g. ``st.secrets``)."""
    value = _clean(environ.get(name))
    if value is None and secrets is not None:
        try:
            value = _clean(secrets.get(name))
        except Exception:  # noqa: BLE001 - a broken secrets source means "not configured"
            value = None
    return value


def live_gate_enabled(environ: Mapping[str, str] | None = None,
                      secrets: Mapping[str, Any] | None = None) -> bool:
    """``SCIFORGE_LIVE_ENABLED``: True only for exactly "true" (case-insensitive, trimmed); default False.

    Environment variable first, then the secrets mapping (a non-empty env value always wins).
    """
    env = os.environ if environ is None else environ
    value = _credential(LIVE_ENABLED_NAME, env, secrets)
    return value is not None and value.lower() == "true"


@dataclass(frozen=True)
class LiveAvailability:
    """Whether Live Mode can be enabled. Holds names and booleans only, never values."""

    available: bool
    missing: tuple[str, ...]
    gate_enabled: bool = False

    @property
    def message(self) -> str:
        if self.available:
            return "Live Mode is enabled for this deployment and credentials are configured (values are never displayed)."
        if not self.gate_enabled:
            return (f"Live Mode is disabled for this deployment ({LIVE_ENABLED_NAME} is not set to \"true\").")
        return ("Live Mode is unavailable: " + " and ".join(self.missing)
                + (" is" if len(self.missing) == 1 else " are") + " not configured.")


def live_availability(environ: Mapping[str, str] | None = None,
                      secrets: Mapping[str, Any] | None = None) -> LiveAvailability:
    """Live Mode requires the deployment gate AND both credentials (presence only)."""
    env = os.environ if environ is None else environ
    gate = live_gate_enabled(env, secrets)
    missing = tuple(n for n in LIVE_CREDENTIAL_NAMES if _credential(n, env, secrets) is None)
    return LiveAvailability(available=gate and not missing, missing=missing, gate_enabled=gate)


def _model_env(environ: Mapping[str, str], secrets: Mapping[str, Any] | None) -> dict[str, str]:
    env = dict(environ)
    for name in LIVE_CREDENTIAL_NAMES:
        value = _credential(name, environ, secrets)
        if value is not None:
            env[name] = value
    return env


def _secret_candidates(environ: Mapping[str, str], secrets: Mapping[str, Any] | None) -> list[str]:
    """Values to scrub from every displayed string (never returned)."""
    out = [_credential("XAI_API_KEY", environ, secrets), _clean(environ.get("NCBI_API_KEY")),
           _clean(environ.get("SCIFORGE_CONTACT_EMAIL"))]
    return [v for v in out if v]


# ------------------------------------------------------------------ progress


@dataclass(frozen=True)
class ProgressEvent:
    stage: str          # one of PROGRESS_STAGES keys
    state: str          # running | done | skipped | failed
    detail: str = ""


ProgressCallback = Callable[[ProgressEvent], None]


class _Progress:
    def __init__(self, callback: ProgressCallback | None) -> None:
        self.callback = callback
        self.states: dict[str, str] = {k: "pending" for k, _ in PROGRESS_STAGES}

    def emit(self, stage: str, state: str, detail: str = "") -> None:
        self.states[stage] = state
        if self.callback is not None:
            try:
                self.callback(ProgressEvent(stage, state, detail))
            except Exception:  # noqa: BLE001 - a UI problem must never break the run
                pass


class _ProgressModelClient:
    """Delegating ModelClient that reports stage transitions (request.stage) to the progress tracker."""

    def __init__(self, inner: ModelClient, progress: _Progress) -> None:
        self._inner = inner
        self._progress = progress
        self._current: str | None = None
        self.name = inner.name
        self.model = inner.model

    def complete(self, request: ModelRequest) -> ModelResponse:
        stage = _STAGE_FOR_MODEL.get(request.stage or "")
        if stage is not None and stage != self._current:
            if self._current is not None:
                self._progress.emit(self._current, "done")
            if self._current == "extract":
                self._progress.emit("check", "done")
            self._current = stage
            self._progress.emit(stage, "running")
        return self._inner.complete(request)


# ------------------------------------------------------------------ privacy guard

_PATH_RE = re.compile(r"(?<![\w.:/\-])(?:/(?:tmp|home|Users|var|private|mnt|workspace|root|opt|srv|etc|usr)"
                      r"(?:/[^\s)\]'\"`>|,;]*)?|[A-Za-z]:\\[^\s)\]'\"`>|,;]*)")


def guard_display(value: Any, *, secrets: Iterable[str] = (), withheld: Iterable[str] = (),
                  paths: Iterable[str] = ()) -> Any:
    """Recursively scrub a display value: secrets, key-like tokens, filesystem paths, raw rejected output."""
    secret_list = [s for s in secrets if s]
    needles = sorted({n for n in withheld if n}, key=len, reverse=True)
    path_list = sorted({p for p in paths if p}, key=len, reverse=True)

    def scrub(text: str) -> str:
        for needle in needles:
            if needle in text:
                text = text.replace(needle, WITHHELD)
        for p in path_list:
            text = text.replace(p, PATH_MASK)
        text = redact_text(text, secret_list)
        return _PATH_RE.sub(PATH_MASK, text)

    def walk(node: Any) -> Any:
        if isinstance(node, str):
            return scrub(node)
        if isinstance(node, dict):
            return {walk(k) if isinstance(k, str) else k: walk(v) for k, v in node.items()}
        if isinstance(node, (list, tuple)):
            return [walk(v) for v in node]
        return node

    return walk(value)


# ------------------------------------------------------------------ result


@dataclass
class WebInvestigationResult:
    ok: bool
    mode: str
    demo: bool
    status: str                                  # ok | degraded | budget_exhausted | model_auth_error | error | invalid_input | live_unavailable
    question: str = ""
    notices: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    question_definition: dict[str, Any] | None = None
    report_markdown: str = ""
    sections: dict[str, str] = field(default_factory=dict)       # section title -> Markdown body (from report.md)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    gaps: list[dict[str, Any]] = field(default_factory=list)
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    progress: dict[str, str] = field(default_factory=dict)

    def displayed_text(self) -> str:
        """Every string the UI can show, concatenated (used by privacy tests)."""
        parts: list[str] = []

        def walk(node: Any) -> None:
            if isinstance(node, str):
                parts.append(node)
            elif isinstance(node, dict):
                for k, v in node.items():
                    walk(k)
                    walk(v)
            elif isinstance(node, (list, tuple)):
                for v in node:
                    walk(v)

        for name in ("question", "notices", "errors", "question_definition", "report_markdown", "sections",
                     "evidence", "conflicts", "gaps", "hypotheses", "sources", "validation", "limitations"):
            walk(getattr(self, name))
        return "\n".join(parts)


def split_report_sections(report: str) -> dict[str, str]:
    """``{"A. Research Question": body, ...}`` from the code-built report.md (bodies unchanged)."""
    wanted = set(SECTION_TITLES.values())
    sections: dict[str, str] = {}
    current: str | None = None
    lines: list[str] = []
    for line in report.splitlines():
        if line.startswith("## ") and line[3:].strip() in wanted:
            if current is not None:
                sections[current] = "\n".join(lines).strip()
            current, lines = line[3:].strip(), []
        elif line.strip() == "---" and current == SECTION_TITLES["J"]:
            sections[current] = "\n".join(lines).strip()
            current, lines = None, []
        elif current is not None:
            lines.append(line)
    if current is not None:
        sections[current] = "\n".join(lines).strip()
    return sections


def render_sources(records: Sequence[Record], verification: Sequence[VerificationResult],
                   citations: Sequence[Mapping[str, Any]], access_levels: Mapping[str, str | None]) -> list[dict[str, Any]]:
    """Source list for the UI, rendered ONLY by the report module's deterministic citation renderer.

    ``citations`` is ``report_validation["citations"]`` (the [S#] numbering assigned by
    the report). Cited records keep their [S#] ref; unresolved ids get the
    ``[UNRESOLVED CITATION: id]`` placeholder; retrieved-but-uncited v0.2 records are
    listed afterwards with refs ``R1``, ``R2``... (same renderer).
    """
    by_id = {r.record_id: r for r in records}
    ver = {v.record_id: v for v in verification}
    out: list[dict[str, Any]] = []
    cited: set[str] = set()
    for c in citations:
        rid = c.get("record_id")
        record = by_id.get(rid) if isinstance(rid, str) else None
        if c.get("status") != "resolved" or record is None:
            out.append({"ref": None, "record_id": rid, "cited": True, "resolved": False,
                        "citation": f"{unresolved_marker(rid)} — no v0.2 record with this id; no citation rendered.",
                        "verification_status": None, "used_as_evidence_source": False})
            continue
        cited.add(record.record_id)
        out.append({"ref": c.get("ref"), "record_id": record.record_id, "cited": True, "resolved": True,
                    "citation": render_citation(str(c.get("ref")), record, ver.get(record.record_id),
                                                access_levels.get(record.record_id)),
                    "verification_status": ver[record.record_id].status if record.record_id in ver else None,
                    "used_as_evidence_source": True})
    n = 0
    for record in records:
        if record.record_id in cited:
            continue
        n += 1
        out.append({"ref": f"R{n}", "record_id": record.record_id, "cited": False, "resolved": True,
                    "citation": render_citation(f"R{n}", record, ver.get(record.record_id),
                                                access_levels.get(record.record_id)),
                    "verification_status": ver[record.record_id].status if record.record_id in ver else None,
                    "used_as_evidence_source": False})
    return out


def _demo_search_summary(question: str, records: list[Record], verification: list[VerificationResult],
                         from_year: int | None, to_year: int | None, max_sources: int) -> dict[str, Any]:
    """v0.2 ``summary.json``-shaped payload for the synthetic dataset (built by the v0.2 summary builder)."""
    from sciforge.logging_utils import RunLog
    from sciforge.pipeline import build_summary

    now = utc_now()
    outcomes = [SearchOutcome(database="crossref", status="ok", records=list(records), total_hits=len(records))]
    params = {"max_results_per_source": max_sources, "from_year": from_year, "to_year": to_year}
    summary = build_summary(question, params, now, now, outcomes, list(records), list(records), 0, verification,
                            RunLog())
    summary["databases_queried"] = ["crossref"]
    summary["query_generation"] = ("none: Demo Mode — no database was searched; the bundled SYNTHETIC demo "
                                   "dataset was used")
    return summary


# ------------------------------------------------------------------ orchestration


def _status_of(res: InvestigationModelResult) -> str:
    stop = res.base.question.call.stop or res.base.evidence.stop or res.gaps.stop or res.hypotheses.stop \
        or res.narrative.stop
    if stop == "budget_exhausted":
        return "budget_exhausted"
    if stop == "auth_error":
        return "model_auth_error"
    if not res.base.evidence.accepted:
        return "degraded"
    return "ok"


def _reason_counts(items: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        codes = item.get("reason_codes") or [r.get("code") for r in item.get("reasons") or [] if isinstance(r, dict)]
        for code in codes:
            if isinstance(code, str):
                counts[code] = counts.get(code, 0) + 1
    return dict(sorted(counts.items()))


def _build_result(req: InvestigationRequest, res: InvestigationModelResult, records: list[Record],
                  verification: list[VerificationResult], search_summary: Mapping[str, Any] | None,
                  demo: bool) -> WebInvestigationResult:
    report = res.files["report"].read_text(encoding="utf-8")
    validation = res.report_validation
    access = {s.record_id: (s.access_level if s.access_level != "not_accessed" else None)
              for s in res.base.source_texts.sources}
    refs = {c["record_id"]: c["ref"] for c in validation.get("citations", []) if c.get("status") == "resolved"}
    evidence = []
    for ev in res.base.evidence.accepted:
        rid = ev["source_record_id"]
        evidence.append({"evidence_id": ev["evidence_id"], "claim": ev["claim"], "quote": ev["quote"],
                         "finding": ev.get("finding"), "methods": ev.get("methods"),
                         "limitations": ev.get("limitations"), "category": ev["evidence_category"],
                         "confidence": ev["confidence"], "access": ev.get("access_level"),
                         "source_ref": refs.get(rid) or unresolved_marker(rid)})
    conflicts = [e for e in evidence if e["category"] == "conflicting"]
    ev_ref = {e["evidence_id"]: e["source_ref"] for e in evidence}
    gaps = [{"gap_id": g["gap_id"], "label": g["label"], "statement": g["gap_statement"],
             "why_unresolved": g["why_unresolved"], "supporting_evidence_ids": g["supporting_evidence_ids"],
             "conflicting_evidence_ids": g["conflicting_evidence_ids"], "confidence": g["confidence"],
             "source_refs": sorted({ev_ref[e] for e in [*g["supporting_evidence_ids"], *g["conflicting_evidence_ids"]]
                                    if e in ev_ref})}
            for g in res.gaps.accepted]
    hypotheses = [{"hypothesis_id": h["hypothesis_id"], "label": "hypothesis", "statement": h["statement"],
                   "rationale": h["rationale"], "predicted_observable_outcome": h["predicted_observable_outcome"],
                   "assumptions": list(h["assumptions"]), "research_gap_ids": h["research_gap_ids"],
                   "supporting_evidence_ids": h["supporting_evidence_ids"], "confidence": h["confidence"]}
                  for h in res.hypotheses.accepted]
    sources = render_sources(records, verification, validation.get("citations", []), access)
    ev_rej = res.base.evidence.rejected
    stages = {name: {"status": r.status, "skip_reason": r.skip_reason, "accepted": len(r.accepted),
                     "rejected": len(r.rejected), "rejection_reasons": _reason_counts(r.rejected)}
              for name, r in (("research_gaps", res.gaps), ("hypotheses", res.hypotheses),
                              ("report_narrative", res.narrative))}
    budget = res.budget
    used = budget.get("used", {})
    validation_view = {
        "report_status": validation.get("status"),
        "sections_present": validation.get("sections_present", []),
        "citations": {"resolved": validation.get("counts", {}).get("citations_resolved", 0),
                      "unresolved": validation.get("counts", {}).get("citations_unresolved", 0)},
        "issues": [{k: v for k, v in i.items() if k in ("code", "record_id", "detail")}
                   for i in validation.get("issues", [])],
        "question_definition": "ok" if res.base.question.definition is not None else "failed (raw question used)",
        "source_texts": res.base.source_texts.counts(),
        "evidence": {"accepted": len(res.base.evidence.accepted), "rejected": len(ev_rej),
                     "rejection_reasons": _reason_counts(ev_rej)},
        "stages": stages,
        "budget": {"attempts_used": used.get("attempts"), "attempts_limit": budget.get("limits", {}).get("max_attempts"),
                   "sources_used": used.get("sources"), "sources_limit": budget.get("limits", {}).get("max_sources"),
                   "input_tokens_used": used.get("input_tokens"), "output_tokens_used": used.get("output_tokens"),
                   "estimated_spend_usd": used.get("spend_usd"),
                   "spend_cap_usd": budget.get("limits", {}).get("max_spend_usd"),
                   "exhausted_by": budget.get("exhausted_by")},
        "model": "scripted demo model (FakeModelClient; no API call)" if demo else "xAI Responses API (XAIClient)",
        "raw_rejected_model_output": "never displayed; rejected items are shown as reason codes only",
    }
    limitations = [
        "Evidence comes from abstracts only (no full text).",
        "Claim support is checked deterministically (exact quotes, numbers, ids); SciForge does not judge study quality.",
        "Citation verification confirms that identifiers resolve and match; it cannot confirm what a paper claims.",
        "Research gaps are the pipeline's inference and hypotheses are untested proposals, not findings.",
    ]
    if demo:
        limitations.insert(0, "DEMO MODE: every source, abstract, finding, gap and hypothesis is SYNTHETIC and "
                              "illustrates the pipeline only. Nothing here is a real finding.")
    else:
        limitations.insert(0, "Live Mode has not yet been validated against the real xAI API; treat output with care.")
    definition = res.base.question.definition.model_dump() if res.base.question.definition else None
    notices = []
    status = _status_of(res)
    if status == "degraded":
        notices.append("No evidence passed validation; the report contains no findings (degraded result).")
    elif status == "budget_exhausted":
        notices.append("The model budget was exhausted; later sections were built by code only.")
    elif status == "model_auth_error":
        notices.append("The model provider rejected the credentials; later sections were built by code only.")
    return WebInvestigationResult(
        ok=True, mode=req.mode, demo=demo, status=status, question=req.question.strip(), notices=notices,
        question_definition=definition, report_markdown=report, sections=split_report_sections(report),
        evidence=evidence, conflicts=conflicts, gaps=gaps, hypotheses=hypotheses, sources=sources,
        validation=validation_view, limitations=limitations)


def _rejected_needles(res: InvestigationModelResult) -> list[str]:
    raw = [*res.base.evidence.rejected_raw, *res.gaps.rejected_raw, *res.hypotheses.rejected_raw,
           *res.narrative.rejected_raw]
    return rejected_text_values(raw)


def run_web_investigation(
    request: InvestigationRequest,
    *,
    progress: ProgressCallback | None = None,
    environ: Mapping[str, str] | None = None,
    secrets: Mapping[str, Any] | None = None,
    live_http_client: httpx.Client | None = None,
    live_model_client_factory: Callable[[ModelSettings], ModelClient] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> WebInvestigationResult:
    """Run one investigation for the web UI. Never raises; problems come back in ``errors``.

    ``live_http_client`` / ``live_model_client_factory`` exist for offline tests
    (MockTransport / FakeModelClient); the app never passes them.
    """
    env = os.environ if environ is None else environ
    tracker = _Progress(progress)
    demo = request.mode == MODE_DEMO
    secret_values = _secret_candidates(env, secrets)
    errors = validate_request(request)
    if errors:
        return WebInvestigationResult(ok=False, mode=request.mode, demo=demo, status="invalid_input", errors=errors,
                                      progress=dict(tracker.states))
    if not demo:
        # Defence in depth: the service refuses live runs itself (deployment gate + credentials), not only the UI.
        availability = live_availability(env, secrets)
        if not availability.available:
            return WebInvestigationResult(ok=False, mode=request.mode, demo=False, status="live_unavailable",
                                          errors=[availability.message], progress=dict(tracker.states))
    tmp = Path(tempfile.mkdtemp(prefix=TEMP_PREFIX))
    # Only specific (multi-component) paths are masked literally; a bare "/" must never be replaced.
    paths = [p for p in {str(tmp), str(tmp.resolve()), str(Path.home()), str(Path.cwd())} if len(Path(p).parts) >= 3]
    try:
        try:
            if demo:
                result, needles, leaks = _run_demo(request, tracker, tmp, secret_values)
            else:
                result, needles, leaks = _run_live(request, tracker, tmp, env, secrets, secret_values,
                                                   live_http_client, live_model_client_factory, sleep)
        except (ModelConfigError, ConfigError) as exc:
            # Config messages name variables only (never values); still scrubbed below.
            return guard_display_result(WebInvestigationResult(
                ok=False, mode=request.mode, demo=demo, status="error",
                errors=[f"Live Mode configuration error: {exc}"], progress=dict(tracker.states)),
                secret_values, [], paths)
        except Exception as exc:  # noqa: BLE001 - never show tracebacks / paths in the UI
            for key, state in tracker.states.items():
                if state == "running":
                    tracker.emit(key, "failed")
            return WebInvestigationResult(ok=False, mode=request.mode, demo=demo, status="error",
                                          errors=[f"The investigation failed unexpectedly ({type(exc).__name__})."],
                                          progress=dict(tracker.states))
        _finish_progress(tracker, result)
        result.progress = dict(tracker.states)
        result.validation["privacy_guard"] = {
            "run_artifacts": "written to a temporary directory outside the repository and deleted after the run",
            "rejected_output_leaks_in_run_files": len(leaks),
            "display_scrubbed": "secrets, key-like tokens, filesystem paths and raw rejected output",
        }
        return guard_display_result(result, secret_values, needles, paths)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _finish_progress(tracker: _Progress, result: WebInvestigationResult) -> None:
    """Close every stage with its final state and a short, code-built detail line."""
    v = result.validation
    ev = v.get("evidence", {})
    stages = v.get("stages", {})
    details = {
        "define": f"question definition {v.get('question_definition', 'n/a')}",
        "extract": f"evidence items proposed: {ev.get('accepted', 0) + ev.get('rejected', 0)}",
        "check": f"{ev.get('accepted', 0)} accepted, {ev.get('rejected', 0)} rejected by deterministic checks",
        "gaps": f"research gaps accepted: {stages.get('research_gaps', {}).get('accepted', 0)}",
        "hypotheses": f"hypotheses accepted: {stages.get('hypotheses', {}).get('accepted', 0)}",
        "report": f"report validation: {v.get('report_status', 'n/a')}",
    }
    for key, _label in PROGRESS_STAGES:
        state = tracker.states[key]
        if key in details or state in ("pending", "running"):
            final = "skipped" if state == "pending" else ("done" if state == "running" else state)
            tracker.emit(key, final, details.get(key, ""))


def guard_display_result(result: WebInvestigationResult, secrets: Sequence[str], needles: Sequence[str],
                         paths: Sequence[str]) -> WebInvestigationResult:
    for name in ("question", "notices", "errors", "question_definition", "report_markdown", "sections", "evidence",
                 "conflicts", "gaps", "hypotheses", "sources", "validation", "limitations"):
        setattr(result, name, guard_display(getattr(result, name), secrets=secrets, withheld=needles, paths=paths))
    return result


def _pipeline_kwargs(tmp: Path, sleep: Callable[[float], None]) -> dict[str, Any]:
    from sciforge.http_utils import RateLimiter

    return {"run_dir": tmp / "model", "sleep": sleep, "debug_keep_rejected_raw": False,
            "pubmed_limiter": RateLimiter(0, sleep=sleep), "crossref_limiter": RateLimiter(0, sleep=sleep)}


def _run_demo(req: InvestigationRequest, tracker: _Progress, tmp: Path,
              secret_values: list[str]) -> tuple[WebInvestigationResult, list[str], list]:
    from sciforge.demo_data import DEMO_LABEL, demo_http_client, demo_model_client, demo_records
    from sciforge.llm.budget import BudgetLimits, BudgetTracker

    question = req.question.strip()
    tracker.emit("search", "running", "Loading the bundled synthetic demo dataset (no database is searched)")
    records, verification = demo_records(req.from_year, req.to_year)
    tracker.emit("search", "done", f"{len(records)} synthetic records")
    tracker.emit("verify", "running", "Demo verification statuses are synthetic")
    summary = _demo_search_summary(question, records, verification, req.from_year, req.to_year, req.max_sources)
    tracker.emit("verify", "done")
    budget = BudgetTracker(BudgetLimits(max_sources=req.max_sources, max_spend_usd=None))
    client = _ProgressModelClient(demo_model_client(), tracker)
    no_sleep: Callable[[float], None] = lambda _s: None  # noqa: E731
    http = demo_http_client()
    try:
        res = run_model_investigation(question, records, verification, model_client=client,
                                      settings=Settings(), tracker=budget, search_summary=summary,
                                      http_client=http, max_source_chars=4000, include_partially_verified=False,
                                      **_pipeline_kwargs(tmp, no_sleep))
    finally:
        http.close()
    needles = _rejected_needles(res)
    leaks = find_leaks(res.run_dir, needles)
    result = _build_result(req, res, records, verification, summary, demo=True)
    result.notices.insert(0, f"{DEMO_LABEL}. Demo Mode always analyses the same bundled synthetic example "
                             "dataset; the entered question is echoed but not searched.")
    return result, needles, leaks


def _run_live(req: InvestigationRequest, tracker: _Progress, tmp: Path, env: Mapping[str, str],
              secrets: Mapping[str, Any] | None, secret_values: list[str], http_client: httpx.Client | None,
              factory: Callable[[ModelSettings], ModelClient] | None,
              sleep: Callable[[float], None] | None) -> tuple[WebInvestigationResult, list[str], list]:
    from sciforge.pipeline import run_investigation

    availability = live_availability(env, secrets)
    if not availability.available:
        raise ModelConfigError(availability.message)
    model_env = _model_env(env, secrets)
    settings = Settings.from_env(model_env)
    model_settings = ModelSettings.from_env(model_env)          # budgets, prices, spend cap (fail closed)
    model_settings = replace(model_settings, max_sources=min(model_settings.max_sources, req.max_sources))
    secret_values.extend(v for v in settings.secret_values() + model_settings.secret_values() if v)
    question = req.question.strip()

    tracker.emit("search", "running", "PubMed and Crossref")
    v02 = run_investigation(question, max_results=req.max_sources, from_year=req.from_year, to_year=req.to_year,
                            output_dir=tmp / "v02", settings=settings, client=http_client,
                            **({"sleep": sleep} if sleep is not None else {}))
    tracker.emit("search", "done", f"{v02.summary.get('total_retrieved', 0)} records retrieved")
    tracker.emit("verify", "done", "DOI/PMID verification (v0.2)")
    if factory is not None:
        inner = factory(model_settings)
    else:
        from sciforge.llm.xai import XAIClient

        inner = XAIClient(model_settings)
    client = _ProgressModelClient(inner, tracker)
    try:
        kwargs = _pipeline_kwargs(tmp, sleep or time.sleep)
        if http_client is None:
            kwargs.pop("pubmed_limiter")
            kwargs.pop("crossref_limiter")
        res = run_model_investigation(question, v02.records, v02.verification, model_client=client,
                                      settings=settings, model_settings=model_settings, search_summary=v02.summary,
                                      http_client=http_client, **kwargs)
    finally:
        close = getattr(inner, "close", None)
        if callable(close) and factory is None:
            close()
    needles = _rejected_needles(res)
    leaks = find_leaks(res.run_dir, needles)
    result = _build_result(req, res, v02.records, v02.verification, v02.summary, demo=False)
    result.notices.insert(0, "Live Mode is experimental and has not been validated against the real xAI API.")
    return result, needles, leaks
