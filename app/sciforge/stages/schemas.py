"""Model-facing evidence schemas (strict; NO bibliographic fields exist).

``evidence_category`` values come from the public evidence-record table in
``agent/research_workflow.md`` (and ``agent/system_prompt.md`` rule 4):
``established``, ``conflicting``, ``inference``, ``hypothesis``.
``confidence`` is ``high`` / ``moderate`` / ``low``: the table's fourth value
(``not verified``) is omitted because the model is never asked to verify
anything bibliographic, and claim support is decided by code, not by the model.
"""

from __future__ import annotations

from typing import Literal, get_args

from pydantic import Field

from sciforge.llm.parsing import StrictModel, strict_json_schema

EvidenceCategory = Literal["established", "conflicting", "inference", "hypothesis"]
Confidence = Literal["high", "moderate", "low"]
EVIDENCE_CATEGORIES: tuple[str, ...] = get_args(EvidenceCategory)
CONFIDENCE_LEVELS: tuple[str, ...] = get_args(Confidence)
MAX_ITEMS_PER_SOURCE = 5

ITEM_FIELDS = ("source_record_id", "claim", "quote", "finding", "methods", "limitations", "relevance",
               "evidence_category", "confidence")


class ModelEvidenceItem(StrictModel):
    """One evidence item as the model must return it (these fields ONLY)."""

    source_record_id: str
    claim: str
    quote: str
    finding: str | None
    methods: str | None
    limitations: str | None
    relevance: str | None
    evidence_category: EvidenceCategory
    confidence: Confidence


class ModelEvidenceBatch(StrictModel):
    """Envelope returned by one extraction call."""

    items: list[ModelEvidenceItem] = Field(max_length=MAX_ITEMS_PER_SOURCE)


EVIDENCE_SCHEMA = strict_json_schema(ModelEvidenceBatch)
assert tuple(ModelEvidenceItem.model_fields) == ITEM_FIELDS
