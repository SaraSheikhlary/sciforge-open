"""Stage S1 — question definition (model).

Input: ONLY the researcher's question (verbatim). No sources are needed to
restate a question, and withholding them keeps this call small and free of
any source data. Output: :class:`QuestionDefinition` (strict schema). One
repair retry on invalid JSON / schema violation; failures are recorded and the
pipeline continues with the raw question.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import Field

from sciforge.boundary import canonical_json
from sciforge.llm.client import ModelMessage
from sciforge.llm.parsing import StrictModel, strict_json_schema, validate_structured
from sciforge.prompts import PROMPT_VERSION, QUESTION_INSTRUCTIONS
from sciforge.stages.common import CallContext, StructuredResult, call_structured
from sciforge.stages.validation import identifier_warnings

STAGE = "question"
SCHEMA_NAME = "question_definition"


class QuestionDefinition(StrictModel):
    """Model-produced restatement of the research question."""

    research_question: str = Field(min_length=1)
    scope: str
    assumptions: list[str]
    key_concepts: list[str]
    ambiguities: list[str]


QUESTION_SCHEMA = strict_json_schema(QuestionDefinition)


@dataclass
class QuestionStageResult:
    question: str
    call: StructuredResult[QuestionDefinition]
    warnings: list[dict[str, str]]

    @property
    def definition(self) -> QuestionDefinition | None:
        return self.call.value

    def to_json(self) -> dict[str, Any]:
        return {
            "stage": STAGE,
            "prompt_version": PROMPT_VERSION,
            "status": "ok" if self.call.ok else "failed",
            "question": self.question,
            "input_to_model": "research question only (no sources)",
            "definition": self.definition.model_dump() if self.definition else None,
            "fallback": None if self.call.ok else "later stages use the raw research question",
            "logical_calls": self.call.logical_calls,
            "attempts": self.call.attempts,
            "repair_attempted": self.call.repaired,
            "warnings": self.warnings,
            "errors": self.call.errors,
        }


def question_messages(question: str) -> list[ModelMessage]:
    return [ModelMessage("user", canonical_json({"research_question_verbatim": question}))]


def _validate(data: Any, raw: str) -> QuestionDefinition:
    return validate_structured(data, QuestionDefinition, raw_text=raw)


def run_question_stage(ctx: CallContext, question: str) -> QuestionStageResult:
    call = call_structured(ctx, stage=STAGE, instructions=QUESTION_INSTRUCTIONS, messages=question_messages(question),
                           schema_name=SCHEMA_NAME, json_schema=QUESTION_SCHEMA, validate=_validate)
    warnings: list[dict[str, str]] = []
    if call.value is not None:
        d = call.value
        fields = {"research_question": d.research_question, "scope": d.scope}
        for name in ("assumptions", "key_concepts", "ambiguities"):
            for i, v in enumerate(getattr(d, name)):
                fields[f"{name}[{i}]"] = v
        warnings = identifier_warnings(fields)
    return QuestionStageResult(question=question, call=call, warnings=warnings)
