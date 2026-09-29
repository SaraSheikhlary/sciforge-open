"""Stage-aware output-token limits, reasoning effort, 200k input default (offline: Fake/MockTransport only)."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

import m3_support as m3
from conftest import FAKE_XAI_KEY, TEST_XAI_BASE, model_env, responses_payload
from sciforge.config import (
    DEFAULT_MODEL_MAX_INPUT_TOKENS,
    DEFAULT_MODEL_MAX_OUTPUT_TOKENS,
    DEFAULT_REASONING_EFFORT,
    MODEL_ENV_VARS,
    REASONING_EFFORTS,
    ModelSettings,
)
from sciforge.llm.audit import ModelCallAudit
from sciforge.llm.budget import BudgetLimits, BudgetTracker, PriceTable, budgeted_call
from sciforge.llm.client import REASONING_EFFORT_VALUES, ModelConfigError, ModelMessage, ModelRequest, stage_key
from sciforge.llm.xai import XAIClient, build_request_body

RECOMMENDED_OUTPUT = {"SCIFORGE_MODEL_MAX_OUTPUT_TOKENS_QUESTION": "2000",
                      "SCIFORGE_MODEL_MAX_OUTPUT_TOKENS_EVIDENCE": "2000",
                      "SCIFORGE_MODEL_MAX_OUTPUT_TOKENS_GAPS": "4000",
                      "SCIFORGE_MODEL_MAX_OUTPUT_TOKENS_HYPOTHESES": "8000",
                      "SCIFORGE_MODEL_MAX_OUTPUT_TOKENS_REPORT": "4000"}
RECOMMENDED_EFFORT = {"SCIFORGE_MODEL_REASONING_EFFORT_EVIDENCE": "medium",
                      "SCIFORGE_MODEL_REASONING_EFFORT_GAPS": "medium",
                      "SCIFORGE_MODEL_REASONING_EFFORT_HYPOTHESES": "high",
                      "SCIFORGE_MODEL_REASONING_EFFORT_REPORT": "medium"}
STAGES = ("question", "evidence", "gaps", "hypotheses", "report")


def settings_from(**env):
    return ModelSettings.from_env(model_env(**env))


def req(stage="gaps", max_out=8000, effort=None):
    return ModelRequest(messages=(ModelMessage("user", "x" * 300),), max_output_tokens=max_out, stage=stage,
                        reasoning_effort=effort)


# ------------------------------------------------------------------ defaults


def test_default_input_budget_is_200000_and_other_defaults_unchanged():
    s = settings_from()
    assert DEFAULT_MODEL_MAX_INPUT_TOKENS == 200_000 and s.max_input_tokens == 200_000
    assert BudgetLimits().max_input_tokens == 200_000
    assert s.max_output_tokens_per_call == DEFAULT_MODEL_MAX_OUTPUT_TOKENS == 2_000
    assert (s.max_attempts, s.max_sources, s.max_spend_usd) == (15, 10, None)
    assert settings_from(SCIFORGE_MODEL_MAX_INPUT_TOKENS="50000").max_input_tokens == 50_000   # still configurable
    assert s.budget_limits().to_dict() == {"max_attempts": 15, "max_sources": 10, "max_input_tokens": 200_000,
                                           "max_output_tokens_per_call": 2_000, "max_spend_usd": None}


def test_new_variables_are_registered_and_effort_values_consistent():
    for name in (*RECOMMENDED_OUTPUT, *RECOMMENDED_EFFORT, "SCIFORGE_MODEL_REASONING_EFFORT"):
        assert name in MODEL_ENV_VARS
    assert "SCIFORGE_MODEL_REASONING_EFFORT_QUESTION" not in MODEL_ENV_VARS          # question uses the global
    assert REASONING_EFFORTS == REASONING_EFFORT_VALUES == ("low", "medium", "high", "xhigh")
    assert DEFAULT_REASONING_EFFORT == "high"


def test_stage_key_mapping():
    assert stage_key("extraction") == "evidence" and stage_key("extraction:repair") == "evidence"
    assert stage_key("gaps:repair") == "gaps" and stage_key(None) is None and stage_key("question") == "question"


# ------------------------------------------------------------------ output limits


def test_stage_specific_output_limits():
    s = settings_from(**RECOMMENDED_OUTPUT)
    assert {st: s.max_output_tokens_for(st) for st in STAGES} == {
        "question": 2000, "evidence": 2000, "gaps": 4000, "hypotheses": 8000, "report": 4000}
    assert s.max_output_tokens_for("extraction") == 2000 and s.max_output_tokens_for("hypotheses:repair") == 8000
    limits = s.budget_limits()
    assert limits.max_output_tokens_per_call == 2000                                   # global kept
    assert limits.max_output_tokens_for("hypotheses") == 8000 and limits.max_output_tokens_for("gaps:repair") == 4000
    assert limits.to_dict()["max_output_tokens_by_stage"] == {"evidence": 2000, "gaps": 4000, "hypotheses": 8000,
                                                              "question": 2000, "report": 4000}


def test_unset_or_blank_stage_limits_fall_back_to_global():
    s = settings_from(SCIFORGE_MODEL_MAX_OUTPUT_TOKENS="3000", SCIFORGE_MODEL_MAX_OUTPUT_TOKENS_GAPS="  ",
                      SCIFORGE_MODEL_MAX_OUTPUT_TOKENS_HYPOTHESES="8000")
    assert {st: s.max_output_tokens_for(st) for st in STAGES} == {
        "question": 3000, "evidence": 3000, "gaps": 3000, "hypotheses": 8000, "report": 3000}
    assert s.max_output_tokens_for(None) == 3000 and s.max_output_tokens_for("unknown") == 3000
    assert settings_from().stage_max_output_tokens() == {}


@pytest.mark.parametrize("bad", ["0", "15", "-5", "abc", "1.5", "128001", "true"])
def test_invalid_stage_output_limit_is_a_config_error(bad):
    with pytest.raises(ModelConfigError, match="SCIFORGE_MODEL_MAX_OUTPUT_TOKENS_REPORT"):
        settings_from(SCIFORGE_MODEL_MAX_OUTPUT_TOKENS_REPORT=bad)
    with pytest.raises(ModelConfigError):
        ModelSettings(api_key="k", model="m", max_spend_usd=None, max_output_tokens_gaps=0)


def test_budget_grants_reserves_and_charges_the_effective_stage_limit():
    prices = PriceTable(input_per_mtok=Decimal("2"), output_per_mtok=Decimal("10"))
    limits = BudgetLimits(max_output_tokens_per_call=2000, stage_max_output_tokens=(("hypotheses", 8000),),
                          max_spend_usd=15.0)
    t = BudgetTracker(limits, prices)
    r_h = t.reserve(req("hypotheses", 8000))
    assert r_h.max_output_tokens == 8000 and r_h.request.max_output_tokens == 8000
    assert r_h.worst_case_cost_usd == prices.cost(r_h.estimated_input_tokens, 8000)
    acct = t.record(r_h, None, failed=True)                          # no usage -> worst case, same rules as before
    assert acct["output_tokens_charged"] == 8000 and acct["cost_source"] == "estimated_worst_case"
    r_g = t.reserve(req("gaps", 8000))                                # no override -> global cap
    assert r_g.max_output_tokens == 2000 and r_g.request.max_output_tokens == 2000
    t.release(r_g)
    assert t.attempts == 1 and t.failed_attempts == 1


def test_attempt_and_spend_accounting_unchanged_without_overrides():
    """Same numbers as the pre-change accounting: fake usage 100 in / 50 out per attempt, price table applied."""
    from sciforge.llm.fake import FakeModelClient

    prices = PriceTable(input_per_mtok=Decimal("3"), output_per_mtok=Decimal("15"))
    t = BudgetTracker(BudgetLimits(max_attempts=4), prices)
    client = FakeModelClient(['{"a": 1}', '{"a": 2}'])
    for _ in range(2):
        budgeted_call(client, req("report", 5000), t, sleep=lambda s: None)
    used = t.summary()["used"]
    assert used["attempts"] == 2 and used["input_tokens"] == 200 and used["output_tokens"] == 100
    assert Decimal(used["spend_usd"]) == 2 * (Decimal(100) * 3 + Decimal(50) * 15) / Decimal(1_000_000)
    assert [r.max_output_tokens for r in client.requests] == [2000, 2000]          # global cap applied


# ------------------------------------------------------------------ reasoning effort


def test_reasoning_effort_default_and_stage_overrides():
    s = settings_from()
    assert s.reasoning_efforts() == {st: "high" for st in STAGES}
    s = settings_from(**RECOMMENDED_EFFORT)
    assert s.reasoning_efforts() == {"question": "high", "evidence": "medium", "gaps": "medium",
                                     "hypotheses": "high", "report": "medium"}
    s = settings_from(SCIFORGE_MODEL_REASONING_EFFORT="Low", SCIFORGE_MODEL_REASONING_EFFORT_GAPS=" XHIGH ",
                      SCIFORGE_MODEL_REASONING_EFFORT_QUESTION="xhigh")            # question override is ignored
    assert s.reasoning_effort_for("question") == "low" and s.reasoning_effort_for("gaps:repair") == "xhigh"
    assert s.reasoning_effort_for("extraction") == "low"
    assert settings_from(SCIFORGE_MODEL_REASONING_EFFORT="").reasoning_effort == "high"


@pytest.mark.parametrize("name", ["SCIFORGE_MODEL_REASONING_EFFORT", *RECOMMENDED_EFFORT])
@pytest.mark.parametrize("bad", ["extreme", "none", "0", "hi", "medium-high"])
def test_invalid_reasoning_effort_is_a_config_error(name, bad):
    with pytest.raises(ModelConfigError, match=name):
        settings_from(**{name: bad})


def test_invalid_reasoning_effort_rejected_in_code_paths():
    with pytest.raises(ModelConfigError):
        ModelSettings(api_key="k", model="m", max_spend_usd=None, reasoning_effort="max")
    with pytest.raises(ModelConfigError):
        ModelSettings(api_key="k", model="m", max_spend_usd=None, reasoning_effort_report="max")
    with pytest.raises(ValueError):
        req(effort="max")


def test_request_body_carries_reasoning_effort_and_keeps_store_false():
    body = build_request_body("m", req("gaps", 100, "medium"))
    assert body["reasoning"] == {"effort": "medium"} and body["store"] is False
    plain = build_request_body("m", req("gaps", 100))
    assert "reasoning" not in plain and plain["store"] is False
    assert req("gaps", 100, "xhigh").with_max_output_tokens(50).reasoning_effort == "xhigh"


def test_xai_source_still_hardcodes_store_false():
    import sciforge.llm.xai as xai

    assert '"store": False,  # D1: never store on the provider side' in Path(xai.__file__).read_text()


# ------------------------------------------------------------------ full pipeline (FakeModelClient)


def test_pipeline_sends_stage_efforts_and_stage_output_limits(tmp_path, settings):
    ms = ModelSettings(api_key=FAKE_XAI_KEY, model="fake-model", max_spend_usd=None, max_attempts=30,
                       max_output_tokens_per_call=2000, max_output_tokens_gaps=4000,
                       max_output_tokens_hypotheses=8000, max_output_tokens_report=4000,
                       reasoning_effort="high", reasoning_effort_evidence="medium", reasoning_effort_gaps="medium",
                       reasoning_effort_report="low")
    recs, ver = m3.records()
    result, client = m3.run(tmp_path, settings, m3.full_script(recs), tracker=None, model_settings=ms)
    seen = [(stage_key(r.stage), r.reasoning_effort, r.max_output_tokens) for r in client.requests]
    assert seen == [("question", "high", 2000), ("evidence", "medium", 2000), ("evidence", "medium", 2000),
                    ("gaps", "medium", 4000), ("hypotheses", "high", 8000), ("report", "low", 4000)]
    assert client.reasoning_efforts == ["high", "medium", "medium", "medium", "high", "low"]
    calls = json.loads(result.files["model_calls"].read_text())
    assert [(c["stage"], c["reasoning_effort"], c["max_output_tokens"]) for c in calls] == [
        (r.stage, r.reasoning_effort, r.max_output_tokens) for r in client.requests]
    assert result.budget["limits"]["max_output_tokens_by_stage"] == {"gaps": 4000, "hypotheses": 8000,
                                                                     "report": 4000}


def test_pipeline_without_model_settings_sends_no_reasoning_parameter(tmp_path, settings):
    recs, ver = m3.records()
    _, client = m3.run(tmp_path, settings, m3.full_script(recs))
    assert set(client.reasoning_efforts) == {None}
    assert {r.max_output_tokens for r in client.requests} == {2000}


# ------------------------------------------------------------------ no reasoning traces / secrets in run files


HIDDEN = "HIDDEN-REASONING-TRACE-7f3a"
ENCRYPTED = "ENCRYPTED-REASONING-BLOB-91c2"


def _xai_client(outputs):
    queue = list(outputs)
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        text = json.dumps(queue.pop(0))
        reasoning = [{"type": "reasoning", "id": "rs_1", "encrypted_content": ENCRYPTED,
                      "summary": [{"type": "summary_text", "text": HIDDEN}], "content": [{"text": HIDDEN}]}]
        payload = responses_payload(text, extra_output=reasoning,
                                    usage={"input_tokens": 50, "output_tokens": 20, "total_tokens": 90,
                                           "output_tokens_details": {"reasoning_tokens": 20}},
                                    reasoning={"effort": bodies[-1].get("reasoning", {}).get("effort"),
                                               "summary": HIDDEN})
        return httpx.Response(200, json=payload)

    ms = ModelSettings.from_env(model_env(XAI_BASE_URL=TEST_XAI_BASE, SCIFORGE_MAX_SPEND_USD="none",
                                          SCIFORGE_MODEL_MAX_ATTEMPTS="30", **RECOMMENDED_EFFORT))
    client = XAIClient(ms, http=httpx.Client(transport=httpx.MockTransport(handler)), sleep=lambda s: None)
    return ms, client, bodies


def test_reasoning_traces_never_reach_audit_or_response_objects():
    ms, client, bodies = _xai_client([{"answer": "ok"}])
    audit = ModelCallAudit(store_prompts=True, secrets=ms.secret_values())
    t = BudgetTracker(ms.budget_limits(), ms.price_table())
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"],
              "additionalProperties": False}
    request = ModelRequest(messages=(ModelMessage("user", "q"),), max_output_tokens=100, schema_name="a",
                           json_schema=schema, stage="gaps", reasoning_effort="medium")
    response = budgeted_call(client, request, t, audit=audit, sleep=lambda s: None)
    assert bodies[0]["reasoning"] == {"effort": "medium"} and bodies[0]["store"] is False
    dumped = audit.to_json() + json.dumps(response.to_dict())
    assert HIDDEN not in dumped and ENCRYPTED not in dumped and FAKE_XAI_KEY not in dumped
    assert audit.entries[0]["reasoning_effort"] == "medium"
    assert response.usage.reasoning_tokens == 20                        # token counts only


def test_no_secrets_or_reasoning_traces_in_any_run_file(tmp_path, settings):
    recs, ver = m3.records()
    ms, client, bodies = _xai_client(m3.full_script(recs))
    from sciforge.investigation_pipeline import run_model_investigation

    result = run_model_investigation(m3.QUESTION, recs, ver, model_client=client, settings=settings,
                                     model_settings=ms, output_dir=tmp_path, search_summary=m3.SEARCH_SUMMARY,
                                     http_client=m3.mock_client(m3.api()), pubmed_limiter=m3.no_throttle(),
                                     crossref_limiter=m3.no_throttle(), sleep=lambda s: None, now=lambda: m3.FIXED)
    assert [b["reasoning"]["effort"] for b in bodies] == ["high", "medium", "medium", "medium", "high", "medium"]
    assert all(b["store"] is False for b in bodies)
    files = [p for p in Path(result.run_dir).rglob("*") if p.is_file()]
    assert files
    for path in files:
        text = path.read_text(encoding="utf-8")
        for needle in (HIDDEN, ENCRYPTED, FAKE_XAI_KEY, settings.ncbi_api_key, settings.contact_email):
            assert needle not in text, (path.name, needle)
