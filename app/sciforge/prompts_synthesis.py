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

HYPOTHESES_INSTRUCTIONS = _CORE + _EVIDENCE_RULES + """

Task (candidate hypotheses): propose up to 5 testable hypotheses tied to the accepted research gaps. Each has exactly:
hypothesis_id ("hyp_01", ...), label (always "hypothesis"), statement (phrased as a testable proposal, never as
established fact), supporting_evidence_ids (at least one), research_gap_ids (at least one gap_id from the supplied
gaps), rationale, predicted_observable_outcome, assumptions (list), confidence (high, moderate, low).
Return {"hypotheses": [...]}."""

REPORT_INSTRUCTIONS = _CORE + _EVIDENCE_RULES + """

Task (report narrative): write short narrative paragraphs for these report sections only: key_findings,
conflicting_evidence, limitations, next_steps. Each paragraph has exactly: section, label (established,
conflicting, inference, hypothesis), text, evidence_ids (evidence the paragraph relies on; at least one for
key_findings and conflicting_evidence), source_ids (source_record_ids mentioned; may be empty).
Inside text you may mention only evidence_ids and source_record_ids. The citations list is built by code.
Return {"paragraphs": [...]}."""
