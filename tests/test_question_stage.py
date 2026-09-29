"""v0.3 M2 question-definition stage: strict schema, repair retry, recorded failures."""

from __future__ import annotations

import json

from m2_support import question_output
from sciforge.llm.audit import ModelCallAudit
from sciforge.llm.budget import BudgetLimits, BudgetTracker
from sciforge.llm.client import ModelAuthError, ModelHTTPError, ModelTimeout
from sciforge.llm.fake import FakeModelClient
from sciforge.stages.common import CallContext
from sciforge.stages.question import QUESTION_SCHEMA, QuestionDefinition, run_question_stage


def ctx_for(script, **limits):
    client = FakeModelClient(script)
    tracker = BudgetTracker(BudgetLimits(max_spend_usd=None, **limits))
    audit = ModelCallAudit()
    return CallContext(client=client, tracker=tracker, audit=audit, sleep=lambda s: None), client


def test_strict_schema_shape():
    assert QUESTION_SCHEMA["additionalProperties"] is False
    assert QUESTION_SCHEMA["required"] == ["research_question", "scope", "assumptions", "key_concepts", "ambiguities"]


def test_valid_definition_parsed():
    ctx, client = ctx_for([question_output()])
    res = run_question_stage(ctx, "Does shear activate platelets?")
    assert isinstance(res.definition, QuestionDefinition)
    assert res.definition.key_concepts == ["shear stress", "platelet activation", "P-selectin"]
    out = res.to_json()
    assert out["status"] == "ok" and out["attempts"] == 1 and out["repair_attempted"] is False
    (req,) = client.requests
    assert req.schema_name == "question_definition" and req.json_schema == QUESTION_SCHEMA and req.stage == "question"


def test_invalid_json_then_repair_succeeds_with_full_history():
    ctx, client = ctx_for(["{not json", question_output()])
    res = run_question_stage(ctx, "Q?")
    assert res.definition is not None and res.call.repaired is True
    assert res.call.attempts == 2 and ctx.tracker.attempts == 2  # the repair counts as an attempt
    first, repair = client.requests
    assert repair.stage == "question:repair"
    assert repair.messages[: len(first.messages)] == first.messages
    assert repair.messages[-2].role == "assistant" and repair.messages[-2].content == "{not json"
    assert repair.messages[-1].role == "user" and "failed validation" in repair.messages[-1].content
    assert len(ctx.audit.entries) == 2
    assert res.to_json()["errors"][0]["error_type"] == "invalid_output_json"


def test_schema_violation_repair_fails_is_recorded_not_raised():
    bad = question_output(extra_field="x")
    ctx, client = ctx_for([bad, {"research_question": "only this"}])
    res = run_question_stage(ctx, "Q?")
    assert res.definition is None
    out = res.to_json()
    assert out["status"] == "failed" and out["repair_attempted"] is True and out["attempts"] == 2
    assert [e["error_type"] for e in out["errors"]] == ["schema_violation", "schema_violation"]
    assert out["fallback"] == "later stages use the raw research question"
    assert "extra_field" in out["errors"][0]["message"]
    assert "failed validation" in client.requests[1].messages[-1].content


def test_empty_research_question_is_schema_violation():
    ctx, _ = ctx_for([question_output(research_question=""), question_output()])
    res = run_question_stage(ctx, "Q?")
    assert res.definition is not None and res.call.repaired


def test_non_repairable_error_no_repair():
    ctx, client = ctx_for([ModelHTTPError("HTTP 400", http_status=400)])
    res = run_question_stage(ctx, "Q?")
    assert res.definition is None and len(client.requests) == 1
    assert res.call.error["error_type"] == "http_error" and res.call.stop is None


def test_retryable_error_retried_by_budgeted_call():
    ctx, client = ctx_for([ModelTimeout("timed out", retryable=True), question_output()])
    res = run_question_stage(ctx, "Q?")
    assert res.definition is not None and res.call.attempts == 2 and res.call.repaired is False


def test_auth_error_sets_stop():
    ctx, _ = ctx_for([ModelAuthError("HTTP 401", http_status=401)])
    res = run_question_stage(ctx, "Q?")
    assert res.call.stop == "auth_error" and res.definition is None


def test_budget_exhausted_before_call_is_recorded():
    ctx, client = ctx_for([question_output()], max_attempts=1)
    ctx.tracker.attempts = 1  # budget already used
    res = run_question_stage(ctx, "Q?")
    assert client.requests == [] and res.call.stop == "budget_exhausted"
    assert res.call.error["limit"] == "max_attempts"


def test_repair_blocked_by_budget():
    ctx, client = ctx_for(["nope"], max_attempts=1)
    res = run_question_stage(ctx, "Q?")
    assert len(client.requests) == 1 and res.call.stop == "budget_exhausted"
    assert [e["error_type"] for e in res.call.errors] == ["invalid_output_json", "budget_exhausted"]


def test_identifier_in_definition_flagged():
    ctx, _ = ctx_for([question_output(scope="See doi:10.1234/abcd and https://example.org")])
    res = run_question_stage(ctx, "Q?")
    assert {w["type"] for w in res.warnings} == {"doi", "url"}
    assert json.dumps(res.to_json())
