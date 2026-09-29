"""v0.4 deterministic hypothesis validation (the validator outranks the model and the critic).

Every rule here is code: regular expressions, id-set membership, token overlap. Nothing calls a model.
Reason dicts carry codes, field names, pattern types and opaque ids only — never model free text.

Claim levels (``mechanistic_claim_level``), weakest to strongest::

    observation < association < mechanistic_support < causal_claim

Supported level (:func:`supported_claim_level`) — derived ONLY from the verified exact ``quote`` of each
cited accepted evidence item (a substring of the source abstract; claim/finding are model-written and not
used as a signal) plus its ``evidence_category`` and the metadata source type:

* per evidence item:
  - category ``inference`` / ``hypothesis`` → ``observation``;
  - category ``conflicting`` → at most ``association``;
  - category ``established``: interventional cue in the quote (``INTERVENTION_CUES``: blocked, inhibited,
    knockout, randomised, treated with, ...) → causal-eligible; mechanistic cue (``MECHANISTIC_CUES``:
    binds, receptor, pathway, signalling, unfolds, phosphorylation, ...) → ``mechanistic_support``;
    comparison/association cue (``ASSOCIATION_CUES``: increased, compared with, associated, correlated,
    higher, ...) → ``association``; otherwise ``observation``;
* hypothesis ceiling = the strongest per-item level, EXCEPT ``causal_claim``, which additionally needs at
  least two distinct sources labelled peer-reviewed journal article whose cited items are causal-eligible,
  and no cited ``conflicting`` item; otherwise causal-eligible items count as ``mechanistic_support``.

Abstract-only evidence rarely supports causality, so the mapping is deliberately conservative. A model
level above the ceiling is flagged (``claim_level_exceeds_evidence``) and never silently changed.

Heuristics (documented thresholds; they flag for revision, and unresolved flags reject the hypothesis):

* causal language: ``CAUSAL_PATTERN`` (causes, drives, leads to, is responsible for, induces, mediates,
  determines, triggers, results in, underlies, is required for, ...) counts as unhedged unless a hedge
  (``HEDGE_PATTERN``: may, might, could, would, possibly, potentially, whether, if, hypothes*, propose,
  suggest, likely, plausibly, perhaps) occurs earlier in the same sentence; checked in ``hypothesis`` and
  ``rationale`` whenever the level is below ``causal_claim``. The app appends a deterministic
  ``causality_statement`` ("Causality is not established ...") to every hypothesis below ``causal_claim``.
* measurable prediction: must contain a term from ``MEASURABLE_PATTERN`` (measure, level, rate,
  increase, decrease, reduce, higher, lower, fold, percent/%, compared with, relative to, versus,
  correlat*, threshold, expression, concentration, count, frequency, proportion, activity, ...).
* restatement / discrimination: Jaccard similarity of normalised content-word sets (lower-cased
  alphanumeric tokens of ≥ 3 characters, minus ``STOPWORDS``, trailing "s" stripped) ≥
  ``SIMILARITY_THRESHOLD`` (0.8) between prediction and hypothesis (``prediction_restates_hypothesis``),
  prediction and alternative explanation (``prediction_not_discriminating``), or the falsification
  test's weakening vs supporting result (``falsification_not_discriminating``; also when identical).
* wet-lab procedure guard (``PROTOCOL_PATTERNS``): concentrations (nM, µM, mg/mL, ...), volumes (µL,
  mL), centrifugation (rpm, × g), temperatures (°C), incubation durations and numbered step lines in
  the falsification test or prediction → ``procedural_protocol_detail``. Falsification tests must stay
  high level (what is compared, what is measured, which result would weaken / support).
* self-labelled discovery (``DISCOVERY_PATTERN``: discovery, discovered, finding(s), validated, proven,
  proves, confirms, establishes, demonstrates that, breakthrough) in ``hypothesis``, ``rationale`` or
  ``assumptions`` → ``self_labeled_discovery``; plus the v0.3 assertive-phrasing check.
* quotes: any quoted span of ≥ 12 characters in model text must be an exact substring of the verified
  quote or source text of a cited evidence item (``untraceable_quote``).

Confidence (:func:`confidence_ceiling`) is qualitative only (low / moderate / high); the final value is
``min(model proposal, deterministic ceiling)``. Rules (each records a reason):

* abstract-only evidence → ceiling moderate;
* fewer than two distinct cited sources → low;
* no cited source labelled peer-reviewed journal article (unknown / conference / book only) → moderate;
* all cited sources are preprints → low; some preprint-backed evidence → one step lower;
* conflicting evidence cited, or the linked gap lists conflicting evidence → low;
* no cited item in category ``established`` (indirect support only) → low;
* hypothesis stated at ``causal_claim`` → moderate.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from sciforge.stages.identifiers import IDENTIFIER_PATTERNS
from sciforge.stages.synthesis_checks import (
    ASSERTIVE_PATTERNS,
    BIBLIOGRAPHIC_TEXT_PATTERNS,
    check_keys,
    check_numbers_supported,
    check_text,
)

HYPOTHESIS_LABEL = "Unvalidated, AI-generated hypothesis for further investigation"
HYPOTHESIS_NOTICE = HYPOTHESIS_LABEL + " — not a validated discovery or scientific finding."
CLAIM_LEVELS = ("observation", "association", "mechanistic_support", "causal_claim")
_RANK = {lvl: i for i, lvl in enumerate(CLAIM_LEVELS)}
CONFIDENCE_ORDER = ("low", "moderate", "high")
_CONF_RANK = {c: i for i, c in enumerate(CONFIDENCE_ORDER)}
ALTERNATIVE_BASES = ("evidence", "inference")
FALSIFICATION_FIELDS = ("manipulated_or_compared", "measured", "weakening_result", "supporting_result")
SIMILARITY_THRESHOLD = 0.8
MIN_QUOTED_SPAN = 12
JOURNAL_ARTICLE = "peer-reviewed journal article"
PREPRINT = "preprint"

# Model-facing fields (generation and revision use the same object).
HYPOTHESIS_FIELDS = ("hypothesis_id", "hypothesis", "evidence_ids", "research_gap_id", "mechanistic_claim_level",
                     "rationale", "prediction", "alternative_explanation", "falsification_test", "assumptions",
                     "evidence_limitations", "confidence")
ALTERNATIVE_FIELDS = ("explanation", "basis", "evidence_ids")

# Hard problems: the candidate is rejected immediately and never revised.
HARD_CODES = frozenset({
    "malformed_item", "schema_violation", "unexpected_field", "fabricated_bibliographic_field",
    "identifier_in_text", "bibliographic_text", "unknown_evidence_id", "unknown_source_id", "unknown_gap_id",
    "hypothesis_missing_evidence", "hypothesis_missing_gap", "invalid_claim_level", "invalid_confidence",
    "revision_added_evidence", "revision_dropped_supported_evidence", "revision_changed_gap", "revision_missing",
    "revision_duplicate", "exceeds_item_limit",
})
# Revisable flags: sent to the revision stage; still present after revision -> rejected.
SOFT_CODES = frozenset({
    "claim_level_exceeds_evidence", "unhedged_causal_language", "prediction_missing", "prediction_not_measurable",
    "prediction_restates_hypothesis", "prediction_not_discriminating", "alternative_missing",
    "alternative_basis_invalid", "falsification_incomplete", "falsification_not_discriminating",
    "procedural_protocol_detail", "self_labeled_discovery", "hypothesis_asserted_as_fact", "untraceable_quote",
    "unsupported_claim", "empty_text", "limitations_missing",
})
HYPOTHESIS_REASON_CODES = HARD_CODES | SOFT_CODES

INTERVENTION_CUES = re.compile(
    r"\b(?:randomi[sz]ed|knock-?(?:out|down|ed\s+out|ed\s+down)|blockade|blocked|blocking|inhibit(?:ed|ion|or|ors)|"
    r"antagonis\w*|deficien\w*|delet(?:ed|ion)|silenc\w*|overexpress\w*|treated\s+with|treatment\s+with|"
    r"administ\w*|exposed\s+to|intervention|abolished|prevented|rescued)\b", re.IGNORECASE)
MECHANISTIC_CUES = re.compile(
    r"\b(?:bind(?:s|ing)?|bound\s+to|receptors?|pathways?|signal(?:l)?ing|phosphorylat\w*|cleav\w*|unfold\w*|"
    r"conformation\w*|mechanis\w*|mediat\w*|interact\w*|inhibit\w*|via)\b", re.IGNORECASE)
ASSOCIATION_CUES = re.compile(
    r"\b(?:associat\w*|correlat\w*|linked|increas\w*|decreas\w*|reduc\w*|rose|fell|higher|lower|greater|less|"
    r"compared\s+with|compared\s+to|versus|vs\.?|relative\s+to|differ\w*|more|fewer|smaller|larger)\b",
    re.IGNORECASE)

CAUSAL_PATTERN = re.compile(
    r"\b(?:causes?|caused|causing|drives?|driven|driving|leads?\s+to|led\s+to|(?:is|are)\s+responsible\s+for|"
    r"induces?|induced|inducing|mediates?|mediated|mediating|determines?|determined\s+by|triggers?|triggered|"
    r"results?\s+in|resulted\s+in|underlies|underlie|is\s+required\s+for|are\s+required\s+for|produces?)\b",
    re.IGNORECASE)
HEDGE_PATTERN = re.compile(
    r"\b(?:may|might|could|would|possibly|potentially|whether|if|hypothes\w*|propos\w*|suggest\w*|likely|"
    r"plausibl\w*|perhaps|speculat\w*)\b", re.IGNORECASE)
MEASURABLE_PATTERN = re.compile(
    r"(?:%|\b(?:measur\w*|levels?|rates?|increas\w*|decreas\w*|reduc\w*|elevat\w*|lower|higher|greater|less|"
    r"smaller|larger|fold|percent\w*|compared\s+(?:with|to)|relative\s+to|versus|vs\.?|correlat\w*|threshold\w*|"
    r"expression|concentrations?|counts?|frequenc\w*|proportions?|amounts?|abundance|activity|duration|time\s+to|"
    r"magnitude|differ\w*|change[sd]?|score[sd]?|incidence|prevalence|odds|risk|size|number)\b)", re.IGNORECASE)
DISCOVERY_PATTERN = re.compile(
    r"\b(?:discover(?:y|ies|ed)|findings?|validated|proven|proves|confirms?|confirmed|establishes|"
    r"demonstrates?\s+that|breakthrough)\b", re.IGNORECASE)
PROTOCOL_PATTERNS: dict[str, re.Pattern[str]] = {
    "concentration": re.compile(
        r"\b\d+(?:\.\d+)?\s*(?:[nµμu]M|mM|pM|mg/mL|mg/ml|µg/mL|ug/mL|ng/mL|mg/kg|U/mL|IU/mL|% ?(?:w/v|v/v))(?![A-Za-z])"),
    "volume": re.compile(r"\b\d+(?:\.\d+)?\s*(?:µL|μL|uL|mL|ml)\b"),
    "centrifugation": re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:rpm|x\s?g|×\s?g)\b|\bcentrifug\w*", re.IGNORECASE),
    "temperature": re.compile(r"\b\d+(?:\.\d+)?\s*°\s?C\b"),
    "incubation": re.compile(r"\bincubat\w*[^.]{0,40}?\b\d+(?:\.\d+)?\s*(?:min|minutes|h|hours|hrs|s|seconds)\b",
                             re.IGNORECASE),
    "numbered_steps": re.compile(r"(?:^|\n)\s*(?:step\s*)?\d+[.)]\s+\S|\bstep\s+\d+\b", re.IGNORECASE),
}
EXTRA_BIBLIOGRAPHIC_PATTERNS: dict[str, re.Pattern[str]] = {
    "year_in_parentheses": re.compile(r"\((?:1[89]|20)\d{2}[a-z]?\)"),
    "identifier_word": re.compile(r"\b(?:doi|pmid|pubmed\s+id|isbn|issn)\b", re.IGNORECASE),
}
_QUOTED = re.compile(r"\"([^\"]{%d,})\"|“([^”]{%d,})”" % (MIN_QUOTED_SPAN, MIN_QUOTED_SPAN))
STOPWORDS = frozenset("""
the and for with that this from are was were been being into onto than then there their they them these those
which while would could should may might will shall can not but also its it's our your has have had does did
under over between among within without upon via per such more less when where what whose who whom how why
""".split())
_TOKEN = re.compile(r"[a-z0-9]+")


def content_words(text: str) -> set[str]:
    words = set()
    for tok in _TOKEN.findall((text or "").lower()):
        if len(tok) < 3 or tok in STOPWORDS:
            continue
        words.add(tok[:-1] if len(tok) > 3 and tok.endswith("s") else tok)
    return words


def jaccard(a: str, b: str) -> float:
    wa, wb = content_words(a), content_words(b)
    if not wa and not wb:
        return 1.0
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _r(code: str, detail: str, field: str | None = None, **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"code": code, "detail": detail}
    if field is not None:
        out["field"] = field
    out.update(extra)
    return out


# ------------------------------------------------------------------ claim level


def evidence_claim_level(ev: Mapping[str, Any]) -> str:
    """Per-item level from the verified quote + category (``causal_claim`` = causal-eligible)."""
    category = ev.get("evidence_category")
    quote = str(ev.get("quote") or "")
    if category in ("inference", "hypothesis"):
        return "observation"
    assoc = bool(ASSOCIATION_CUES.search(quote))
    if category == "conflicting":
        return "association" if assoc else "observation"
    if category != "established":
        return "observation"
    if INTERVENTION_CUES.search(quote):
        return "causal_claim"
    if MECHANISTIC_CUES.search(quote):
        return "mechanistic_support"
    return "association" if assoc else "observation"


def supported_claim_level(evidence_ids: Iterable[str], evidence_by_id: Mapping[str, Mapping[str, Any]],
                          source_types: Mapping[str, str]) -> tuple[str, list[str]]:
    """Maximum level the cited accepted evidence supports, with the reasons (see module docstring)."""
    items = [evidence_by_id[e] for e in evidence_ids if e in evidence_by_id]
    if not items:
        return "observation", ["no accepted evidence cited"]
    levels = {ev["evidence_id"]: evidence_claim_level(ev) for ev in items}
    best = max(levels.values(), key=lambda lvl: _RANK[lvl])
    reasons = [f"{eid}: {lvl.replace('_', ' ')}" for eid, lvl in levels.items()]
    if best == "causal_claim":
        causal_sources = {ev["source_record_id"] for ev in items if levels[ev["evidence_id"]] == "causal_claim"
                          and source_types.get(ev["source_record_id"]) == JOURNAL_ARTICLE}
        conflicting = any(ev.get("evidence_category") == "conflicting" for ev in items)
        if len(causal_sources) < 2 or conflicting:
            best = "mechanistic_support"
            reasons.append("causal claim needs >= 2 distinct peer-reviewed sources with interventional evidence "
                           "and no conflicting evidence; capped at mechanistic support")
    return best, reasons


def causality_statement(level: str, supported: str) -> str:
    if level == "causal_claim":
        return ("Stated at the causal-claim level; the cited abstracts contain interventional evidence from at least "
                "two sources, but causality remains untested by SciForge.")
    return (f"Causality is not established: the cited evidence supports at most "
            f"{supported.replace('_', ' ')}; this hypothesis is stated at the {level.replace('_', ' ')} level.")


def unhedged_causal_spans(text: str) -> int:
    """Number of causal-language matches without an earlier hedge in the same sentence."""
    count = 0
    for sentence in re.split(r"(?<=[.;!?])\s+|\n+", text or ""):
        for m in CAUSAL_PATTERN.finditer(sentence):
            if not HEDGE_PATTERN.search(sentence[:m.start()]):
                count += 1
    return count


# ------------------------------------------------------------------ text checks


def protocol_detail_types(text: str) -> list[str]:
    return [kind for kind, p in PROTOCOL_PATTERNS.items() if p.search(text or "")]


def bibliographic_extra(field_name: str, text: str) -> list[dict[str, Any]]:
    out = []
    for kind, p in EXTRA_BIBLIOGRAPHIC_PATTERNS.items():
        if p.search(text or ""):
            out.append(_r("bibliographic_text", "citation-like text in model text", field_name, type=kind))
    return out


def untraceable_quotes(text: str, reference_texts: Iterable[str]) -> int:
    refs = [r for r in reference_texts if r]
    n = 0
    for m in _QUOTED.finditer(text or ""):
        span = (m.group(1) or m.group(2) or "").strip()
        if span and not any(span in ref for ref in refs):
            n += 1
    return n


def has_bibliographic_text(text: str) -> bool:
    return any(p.search(text or "") for p in (*IDENTIFIER_PATTERNS.values(), *BIBLIOGRAPHIC_TEXT_PATTERNS.values(),
                                               *EXTRA_BIBLIOGRAPHIC_PATTERNS.values()))


# ------------------------------------------------------------------ confidence / source quality


def confidence_ceiling(evidence_ids: Iterable[str], evidence_by_id: Mapping[str, Mapping[str, Any]],
                       source_types: Mapping[str, str], *, level: str,
                       gap: Mapping[str, Any] | None = None) -> tuple[str, list[str]]:
    items = [evidence_by_id[e] for e in evidence_ids if e in evidence_by_id]
    ceiling = "high"
    reasons: list[str] = []

    def cap(to: str, why: str) -> None:
        nonlocal ceiling
        if _CONF_RANK[to] < _CONF_RANK[ceiling]:
            ceiling = to
        reasons.append(f"{why} (ceiling {to})")

    sources = {ev["source_record_id"] for ev in items}
    types = {rid: source_types.get(rid, "unknown") for rid in sources}
    if any(ev.get("abstract_only", True) for ev in items):
        cap("moderate", "abstract-only evidence")
    if len(sources) < 2:
        cap("low", f"{len(sources)} distinct source cited")
    if types and JOURNAL_ARTICLE not in types.values():
        cap("moderate", "no cited source is labelled peer-reviewed journal article")
    preprints = [rid for rid, t in types.items() if t == PREPRINT]
    if preprints and len(preprints) == len(types):
        cap("low", "all cited sources are preprints (not peer-reviewed)")
    conflicting = any(ev.get("evidence_category") == "conflicting" for ev in items) or \
        bool(gap and gap.get("conflicting_evidence_ids"))
    if conflicting:
        cap("low", "conflicting evidence cited or recorded for the linked gap")
    if items and not any(ev.get("evidence_category") == "established" for ev in items):
        cap("low", "indirect support only (no established evidence item cited)")
    if level == "causal_claim":
        cap("moderate", "causal claim level")
    if preprints and len(preprints) < len(types):
        lowered = CONFIDENCE_ORDER[max(0, _CONF_RANK[ceiling] - 1)]
        reasons.append(f"some cited evidence comes from preprints (ceiling lowered one step to {lowered})")
        ceiling = lowered
    if not reasons:
        reasons.append("no deterministic cap applied")
    return ceiling, reasons


def final_confidence(proposed: str, ceiling: str) -> str:
    return proposed if _CONF_RANK.get(proposed, 0) <= _CONF_RANK[ceiling] else ceiling


def source_quality_summary(evidence_ids: Iterable[str], evidence_by_id: Mapping[str, Mapping[str, Any]],
                           source_types: Mapping[str, str]) -> str:
    """Deterministic summary of the cited sources (preprints always named as not peer-reviewed)."""
    from sciforge.stages.report import source_type_badge

    items = [evidence_by_id[e] for e in evidence_ids if e in evidence_by_id]
    sources = sorted({ev["source_record_id"] for ev in items})
    counts: dict[str, int] = {}
    for rid in sources:
        badge = source_type_badge(source_types.get(rid))
        counts[badge] = counts.get(badge, 0) + 1
    parts = ", ".join(f"{n} {badge}" for badge, n in sorted(counts.items()))
    return (f"{len(items)} evidence item(s) from {len(sources)} distinct source(s): {parts or 'none'}; "
            "all evidence is abstract-only (no full text).")


# ------------------------------------------------------------------ candidate validation


@dataclass
class ValidationContext:
    evidence_by_id: Mapping[str, Mapping[str, Any]]
    gaps_by_id: Mapping[str, Mapping[str, Any]]
    known_sources: set[str]
    source_types: Mapping[str, str] = field(default_factory=dict)
    source_texts: Mapping[str, str] = field(default_factory=dict)   # record_id -> verified capped source text


@dataclass
class HypothesisCheck:
    """Outcome of :func:`validate_hypothesis_fields`."""

    hard: list[dict[str, Any]]
    soft: list[dict[str, Any]]
    evidence_ids: list[str]
    alternative_evidence_ids: list[str]
    supported_level: str = "observation"
    supported_level_reasons: list[str] = field(default_factory=list)

    @property
    def reasons(self) -> list[dict[str, Any]]:
        return [*self.hard, *self.soft]

    @property
    def passed(self) -> bool:
        return not self.hard and not self.soft


def _id_list(field_name: str, value: Any, known: Iterable[str], *, required: bool) -> tuple[list, list[str]]:
    reasons: list[dict[str, Any]] = []
    known = set(known)
    if not isinstance(value, list):
        return [_r("schema_violation", f"{field_name}: must be a list", field_name)], []
    ids: list[str] = []
    for v in value:
        if isinstance(v, str) and v in known:
            if v not in ids:
                ids.append(v)
        else:
            from sciforge.stages.synthesis_checks import safe_id
            reasons.append(_r("unknown_evidence_id", f"{field_name}: not an accepted evidence item (rejected or "
                                                     "unknown evidence cannot be cited)", field_name,
                              value=safe_id(v)))
    if required and not value:
        reasons.append(_r("hypothesis_missing_evidence", f"{field_name} must reference at least one accepted "
                                                         "evidence item", field_name))
    return reasons, ids


def validate_hypothesis_fields(raw: Any, vctx: ValidationContext) -> HypothesisCheck:
    """All deterministic checks for one generated or revised hypothesis object."""
    from sciforge.stages.synthesis_checks import safe_id

    hard: list[dict[str, Any]] = []
    soft: list[dict[str, Any]] = []
    for r in check_keys(raw, HYPOTHESIS_FIELDS):
        hard.append(r)
    if not isinstance(raw, dict):
        return HypothesisCheck(hard, soft, [], [])
    missing = [f for f in HYPOTHESIS_FIELDS if f not in raw]
    if missing:
        hard.append(_r("schema_violation", "required fields missing", fields=missing))
    known_ev = set(vctx.evidence_by_id)
    known_gaps = set(vctx.gaps_by_id)

    r, evidence_ids = _id_list("evidence_ids", raw.get("evidence_ids", []), known_ev, required=True)
    hard += r
    gap_id = raw.get("research_gap_id")
    if not isinstance(gap_id, str) or not gap_id.strip():
        hard.append(_r("hypothesis_missing_gap", "research_gap_id must name one accepted research gap",
                       "research_gap_id"))
    elif gap_id not in known_gaps:
        hard.append(_r("unknown_gap_id", "research_gap_id is not an accepted gap", "research_gap_id",
                       value=safe_id(gap_id)))
    level = raw.get("mechanistic_claim_level")
    if level not in CLAIM_LEVELS:
        hard.append(_r("invalid_claim_level", f"mechanistic_claim_level must be one of {list(CLAIM_LEVELS)}",
                       "mechanistic_claim_level"))
    if raw.get("confidence") not in CONFIDENCE_ORDER:
        hard.append(_r("invalid_confidence", "confidence must be low, moderate or high (qualitative only)",
                       "confidence"))
    if not isinstance(raw.get("hypothesis_id"), str):
        hard.append(_r("schema_violation", "hypothesis_id: must be a string", "hypothesis_id"))

    # nested objects
    alt = raw.get("alternative_explanation")
    alt_ids: list[str] = []
    if not isinstance(alt, dict):
        hard.append(_r("schema_violation", "alternative_explanation: must be an object", "alternative_explanation"))
        alt = {}
    else:
        for rr in check_keys(alt, ALTERNATIVE_FIELDS):
            hard.append({**rr, "field": "alternative_explanation"})
        r, alt_ids = _id_list("alternative_explanation.evidence_ids", alt.get("evidence_ids", []), known_ev,
                              required=False)
        hard += r
        basis = alt.get("basis")
        if basis not in ALTERNATIVE_BASES:
            soft.append(_r("alternative_basis_invalid", 'alternative basis must be "evidence" or "inference"',
                           "alternative_explanation.basis"))
        elif basis == "evidence" and not alt_ids:
            soft.append(_r("alternative_basis_invalid", "an evidence-based alternative must cite accepted evidence "
                                                        "ids", "alternative_explanation.evidence_ids"))
        elif basis == "inference" and alt.get("evidence_ids"):
            soft.append(_r("alternative_basis_invalid", "an alternative labelled inference must not cite evidence "
                                                        "ids (use basis evidence)", "alternative_explanation.basis"))
    fals = raw.get("falsification_test")
    if not isinstance(fals, dict):
        hard.append(_r("schema_violation", "falsification_test: must be an object", "falsification_test"))
        fals = {}
    else:
        for rr in check_keys(fals, FALSIFICATION_FIELDS):
            hard.append({**rr, "field": "falsification_test"})

    # collect text fields
    texts: dict[str, Any] = {"hypothesis": raw.get("hypothesis"), "rationale": raw.get("rationale"),
                             "prediction": raw.get("prediction"),
                             "alternative_explanation.explanation": alt.get("explanation")}
    for f in FALSIFICATION_FIELDS:
        texts[f"falsification_test.{f}"] = fals.get(f)
    for list_field in ("assumptions", "evidence_limitations"):
        value = raw.get(list_field)
        if not isinstance(value, list) or not all(isinstance(a, str) for a in value):
            hard.append(_r("schema_violation", f"{list_field}: must be a list of strings", list_field))
            value = []
        for i, a in enumerate(value):
            texts[f"{list_field}[{i}]"] = a
        if list_field == "evidence_limitations" and not [a for a in value if a.strip()]:
            soft.append(_r("limitations_missing", "at least one evidence limitation is required",
                           "evidence_limitations"))

    empty_codes = {"prediction": "prediction_missing", "alternative_explanation.explanation": "alternative_missing"}
    inline: set[str] = set()
    for f, value in texts.items():
        if not isinstance(value, str):
            if value is not None or not f.startswith("falsification_test."):
                hard.append(_r("schema_violation", f"{f}: must be a string", f))
            continue
        if not value.strip():
            if f in empty_codes:
                soft.append(_r(empty_codes[f], f"{f} is empty", f))
            elif f.startswith("falsification_test."):
                soft.append(_r("falsification_incomplete", f"{f} is empty", f))
            else:
                soft.append(_r("empty_text", f"{f} is empty", f))
            continue
        rs, refs = check_text(f, value, known_evidence=known_ev, known_sources=vctx.known_sources,
                              known_gaps=known_gaps)
        hard += rs
        inline |= refs
        hard += bibliographic_extra(f, value)
    for f in FALSIFICATION_FIELDS:
        if not isinstance(fals.get(f), str) and fals:
            soft.append(_r("falsification_incomplete", f"falsification_test.{f} is missing",
                           f"falsification_test.{f}"))
    if not fals:
        soft.append(_r("falsification_incomplete", "falsification test with four parts is required",
                       "falsification_test"))

    # evidence-dependent checks
    cited = [*evidence_ids, *[e for e in alt_ids if e not in evidence_ids], *sorted(inline - set(evidence_ids))]
    ref_parts: list[str] = []
    quote_refs: list[str] = []
    for eid in cited:
        ev = vctx.evidence_by_id.get(eid) or {}
        ref_parts.extend(str(ev.get(k) or "") for k in ("claim", "finding", "quote"))
        quote_refs.append(str(ev.get("quote") or ""))
        quote_refs.append(vctx.source_texts.get(str(ev.get("source_record_id")), ""))
    reference = "\n".join(ref_parts)
    for f, value in texts.items():
        if isinstance(value, str) and value.strip():
            soft += check_numbers_supported(f, value, reference)
            if untraceable_quotes(value, quote_refs):
                soft.append(_r("untraceable_quote", "quoted text is not an exact substring of the verified source "
                                                    "text of the cited evidence", f))

    supported, supported_reasons = supported_claim_level(evidence_ids, vctx.evidence_by_id, vctx.source_types)
    if level in CLAIM_LEVELS and evidence_ids:
        if _RANK[level] > _RANK[supported]:
            soft.append(_r("claim_level_exceeds_evidence", "mechanistic_claim_level exceeds the level the cited "
                                                           "evidence supports (no silent upgrade)",
                           "mechanistic_claim_level", proposed=level, supported=supported))
        if level != "causal_claim":
            for f in ("hypothesis", "rationale"):
                if isinstance(texts.get(f), str) and unhedged_causal_spans(texts[f]):
                    soft.append(_r("unhedged_causal_language", "unhedged causal language below the causal-claim "
                                                               "level", f))

    hyp, pred = texts.get("hypothesis"), texts.get("prediction")
    alt_text = texts.get("alternative_explanation.explanation")
    if isinstance(hyp, str):
        for p in ASSERTIVE_PATTERNS:
            if p.search(hyp):
                soft.append(_r("hypothesis_asserted_as_fact", "hypothesis is phrased as established fact",
                               "hypothesis"))
                break
    for f, value in texts.items():
        if (f in ("hypothesis", "rationale") or f.startswith("assumptions[")) and isinstance(value, str) \
                and DISCOVERY_PATTERN.search(value):
            soft.append(_r("self_labeled_discovery", "a hypothesis must not be described as a discovery, finding "
                                                     "or validated/proven result", f))
    if isinstance(pred, str) and pred.strip():
        if not MEASURABLE_PATTERN.search(pred):
            soft.append(_r("prediction_not_measurable", "prediction names no measurable quantity or comparison "
                                                        "(heuristic)", "prediction"))
        if isinstance(hyp, str) and jaccard(pred, hyp) >= SIMILARITY_THRESHOLD:
            soft.append(_r("prediction_restates_hypothesis", f"prediction overlaps the hypothesis (Jaccard >= "
                                                             f"{SIMILARITY_THRESHOLD})", "prediction"))
        if isinstance(alt_text, str) and alt_text.strip() and jaccard(pred, alt_text) >= SIMILARITY_THRESHOLD:
            soft.append(_r("prediction_not_discriminating", "prediction does not differ from the alternative "
                                                            "explanation", "prediction"))
    weak, sup = fals.get("weakening_result"), fals.get("supporting_result")
    if isinstance(weak, str) and isinstance(sup, str) and weak.strip() and sup.strip():
        if weak.strip().lower() == sup.strip().lower() or jaccard(weak, sup) >= SIMILARITY_THRESHOLD:
            soft.append(_r("falsification_not_discriminating", "weakening and supporting results must differ",
                           "falsification_test"))
    for f in ("prediction", *[f"falsification_test.{x}" for x in FALSIFICATION_FIELDS]):
        value = texts.get(f)
        if isinstance(value, str):
            kinds = protocol_detail_types(value)
            if kinds:
                soft.append(_r("procedural_protocol_detail", "falsification tests must stay high level (no wet-lab "
                                                             "protocol detail)", f, types=kinds))
    return HypothesisCheck(hard=hard, soft=soft, evidence_ids=evidence_ids, alternative_evidence_ids=alt_ids,
                           supported_level=supported, supported_level_reasons=supported_reasons)


def revision_link_reasons(revised_ids: list[str], original_ids: list[str],
                          critic_unsupported: Iterable[str]) -> list[dict[str, Any]]:
    """Revised evidence ids must be a subset of the original ones and keep every supported link.

    A link may be dropped only if the critic flagged exactly that evidence id as unsupported.
    """
    from sciforge.stages.synthesis_checks import safe_id

    reasons = []
    added = [e for e in revised_ids if e not in original_ids]
    if added:
        reasons.append(_r("revision_added_evidence", "revision introduced evidence ids that the original "
                                                     "hypothesis did not cite", "evidence_ids",
                          values=[safe_id(e) for e in added]))
    allowed_drop = set(critic_unsupported)
    dropped = [e for e in original_ids if e not in revised_ids and e not in allowed_drop]
    if dropped:
        reasons.append(_r("revision_dropped_supported_evidence", "revision dropped a supported evidence link",
                          "evidence_ids", values=dropped))
    return reasons
