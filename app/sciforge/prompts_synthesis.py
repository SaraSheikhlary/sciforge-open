"""M3 stage instructions (gaps, hypotheses, report narrative). Same rules as M2 prompts.

Deliberately avoids naming individual metadata fields (tests assert they never occur in requests).
"""

from __future__ import annotations

from sciforge.prompts import _CORE

SYNTHESIS_PROMPT_VERSION = "v0.3-m3-1"

_EVIDENCE_RULES = """
You receive ONLY validated evidence records (evidence_id, source_record_id, claim, finding, evidence_category,
confidence). Reason only from them. Reference evidence only by evidence_id (ev_####) and sources only by
source_record_id (rec_...). Do not introduce numbers, statistics or facts that are not in the referenced
evidence. Never write citations, names of people or periodicals, or identifiers of any kind."""

GAPS_INSTRUCTIONS = _CORE + _EVIDENCE_RULES + """

Task (research gaps): identify up to 6 unanswered or weakly answered questions. Each gap has exactly:
gap_id ("gap_01", "gap_02", ...), gap_statement, supporting_evidence_ids (at least one evidence_id),
conflicting_evidence_ids (evidence_ids that disagree; may be empty), why_unresolved, confidence (high, moderate, low).
Return {"gaps": [...]}."""

HYPOTHESIS_PROMPT_VERSION = "v0.4-hypothesis-1"

_HYPOTHESIS_OBJECT = """Each hypothesis has exactly these fields:
hypothesis_id ("hyp_01", ...); hypothesis (a testable proposal, never phrased as established fact, a discovery
or a finding); evidence_ids (at least one supplied evidence_id); research_gap_id (exactly one supplied gap_id);
mechanistic_claim_level (observation, association, mechanistic_support or causal_claim; never higher than the
cited evidence supports; below causal_claim do not use unhedged causal verbs such as causes, drives, induces,
mediates); rationale (why it is proposed, from the cited evidence and the gap); prediction (a measurable,
observable outcome that differs from what the alternative explanation predicts; not a restatement of the
hypothesis); alternative_explanation {explanation, basis ("evidence" with evidence_ids, or "inference" with an
empty evidence_ids list), evidence_ids}; falsification_test {manipulated_or_compared, measured,
weakening_result, supporting_result} (high level only: no concentrations, volumes, temperatures, incubation
times or step-by-step protocol; weakening and supporting results must differ); assumptions (list);
evidence_limitations (list, at least one); confidence (low, moderate or high; words only, no numbers)."""

HYPOTHESES_INSTRUCTIONS = _CORE + _EVIDENCE_RULES + """

Task (candidate hypotheses): propose up to 5 testable hypotheses tied to the accepted research gaps. The evidence
records include the verified exact quote of each item. """ + _HYPOTHESIS_OBJECT + """
Return {"hypotheses": [...]}."""

HYPOTHESIS_CRITIC_INSTRUCTIONS = _CORE + _EVIDENCE_RULES + """

Task (hypothesis critic): you are an independent, skeptical reviewer. For every supplied candidate hypothesis
return one review with exactly: hypothesis_id; checks, an object with exactly these keys, each
{verdict: "pass" | "fail" | "uncertain", explanation: one or two short sentences}:
evidence_supports_mechanism, causal_language_exceeds_evidence (fail if the wording claims more causality than the
evidence supports), distinct_from_evidence (fail if it merely restates an evidence item), prediction_measurable,
prediction_discriminates (fail if the prediction does not distinguish it from the alternative explanation),
falsification_meaningful, ignored_contradictions_or_missing_evidence (fail if conflicting or missing evidence is
ignored), confidence_consistent; unsupported_evidence_ids (cited evidence_ids that do not support the
hypothesis; may be empty); substantive_problems (short list; empty when there are none). Judge only from the
supplied evidence. Do not rewrite the hypothesis. Return {"reviews": [...]}."""

HYPOTHESIS_REVISION_INSTRUCTIONS = _CORE + _EVIDENCE_RULES + """

Task (hypothesis revision): revise each supplied hypothesis to fix the listed deterministic flags and critic
problems. Keep the same hypothesis_id and research_gap_id. Keep every cited evidence_id except those listed as
unsupported; add no new evidence ids, facts, numbers or citations. Lower the claim level or hedge the wording
instead of adding support. """ + _HYPOTHESIS_OBJECT + """
Return {"hypotheses": [...]} with one revised object per supplied hypothesis."""

REPORT_INSTRUCTIONS = _CORE + _EVIDENCE_RULES + """

Task (report narrative): write short narrative paragraphs for these report sections only: key_findings,
conflicting_evidence, limitations, next_steps. Each paragraph has exactly: section, label (established,
conflicting, inference, hypothesis), text, evidence_ids (evidence the paragraph relies on; at least one for
key_findings and conflicting_evidence), source_ids (source_record_ids mentioned; may be empty).
Inside text you may mention only evidence_ids and source_record_ids. The citations list is built by code.
Return {"paragraphs": [...]}."""
