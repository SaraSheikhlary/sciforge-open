"""v0.4 hypothesis engine end to end: attempt/spend accounting, adversarial fixtures, report wording,
redaction and Demo Mode (offline; FakeModelClient + MockTransport only)."""

from __future__ import annotations

import json
import re
from decimal import Decimal
from pathlib import Path

import pytest

from conftest import FAKE_XAI_KEY, model_env
from m2_support import question_output
from m3_support import (
    alternative,
    auto_critic,
    critic_review,
    evidence_items,
    full_script,
    gap,
    hypothesis,
    load,
    narrative,
    records,
    run,
)
from sciforge.hypothesis_validation import HYPOTHESIS_LABEL, HYPOTHESIS_NOTICE
from sciforge.llm.budget import BudgetLimits, BudgetTracker, PriceTable
from sciforge.config import ModelSettings
from sciforge.stages.report import HYPOTHESIS_DISCLAIMER


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SCIFORGE_DEBUG_KEEP_REJECTED_RAW", raising=False)
    monkeypatch.delenv("SCIFORGE_MAX_SOURCE_CHARS", raising=False)


CAUSAL = "GPIb engagement drives the shear-induced P-selectin increase."
FIXED = "GPIb engagement may contribute to the shear-induced P-selectin increase."


def entries(result):
    calls = load(result, "model_calls")
    return calls["entries"] if isinstance(calls, dict) else calls


def run_files_text(result) -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in Path(result.run_dir).rglob("*") if p.is_file())


def revised(hid="hyp_01", **kw):
    return hypothesis(hypothesis_id=hid, hypothesis=FIXED, **kw)


# ================================================================== pipeline order + attempt accounting


def test_full_pipeline_generation_critic_revision_report(tmp_path, settings):
    recs, ver = records()
    script = full_script(recs, hyps={"hypotheses": [hypothesis(hypothesis=CAUSAL)]},
                         revision={"hypotheses": [revised()]})
    result, client = run(tmp_path, settings, script, recs=recs, ver=ver)
    stages = [r.stage for r in client.requests]
    assert stages == ["question", "extraction", "extraction", "gaps", "hypotheses", "hypothesis_critic",
                      "hypothesis_revision", "report"]
    assert client.remaining == 0 and result.budget["used"]["attempts"] == 8
    assert [e["stage"] for e in entries(result)] == stages
    hyps = load(result, "hypotheses")
    assert hyps["attempts_by_stage"] == {"hypotheses": 1, "hypothesis_critic": 1, "hypothesis_revision": 1}
    assert hyps["critic"]["status"] == "completed" and hyps["revision"]["status"] == "completed"
    h = hyps["accepted"][0]
    assert h["hypothesis"] == FIXED and h["stress_test"]["revision_status"] == "revised"
    assert h["validation"]["initial_flags"] == ["unhedged_causal_language"]
    report = result.files["report"].read_text(encoding="utf-8")
    assert "**Revision status:** revised" in report and FIXED in report and CAUSAL not in report


def test_worst_case_with_repairs_stays_within_default_15_attempt_cap(tmp_path, settings):
    """Every synthesis call needs its one repair: 1 + 2 + 2*5 = 13 attempts (2 sources) <= 15."""
    recs, ver = records()
    script = [question_output(), *evidence_items(recs), "bad {", {"gaps": [gap()]},
              "bad {", {"hypotheses": [hypothesis(hypothesis=CAUSAL)]}, "bad {", auto_critic,
              "bad {", {"hypotheses": [revised()]}, "bad {", narrative()]
    t = BudgetTracker(BudgetLimits(max_sources=10, max_spend_usd=None))          # library default: 15 attempts
    assert t.limits.max_attempts == 15
    result, client = run(tmp_path, settings, script, recs=recs, ver=ver, tracker=t)
    assert client.remaining == 0 and result.budget["used"]["attempts"] == 13 <= 15
    hyps = load(result, "hypotheses")
    assert hyps["attempts_by_stage"] == {"hypotheses": 2, "hypothesis_critic": 2, "hypothesis_revision": 2}
    assert hyps["accepted"][0]["stress_test"]["revision_status"] == "revised"


def test_attempt_cap_hit_before_critic_marks_not_stress_tested(tmp_path, settings):
    recs, ver = records()
    t = BudgetTracker(BudgetLimits(max_sources=10, max_attempts=5, max_spend_usd=None))
    result, client = run(tmp_path, settings, full_script(recs), recs=recs, ver=ver, tracker=t)
    assert [r.stage for r in client.requests][-1] == "hypotheses" and len(client.requests) == 5
    assert result.budget["used"]["attempts"] == 5
    hyps = load(result, "hypotheses")
    assert hyps["critic"] == {**hyps["critic"], "status": "not_run", "skip_reason": "budget_exhausted"}
    h = hyps["accepted"][0]
    assert h["stress_test"]["critic_status"] == "not_run" and "NOT stress-tested" in h["stress_test"]["note"]
    report = result.files["report"].read_text(encoding="utf-8")
    assert "NOT stress-tested" in report and "critic_findings" not in json.dumps(h["stress_test"])
    assert load(result, "report_validation")["narrative"]["skip_reason"] == "budget_exhausted"


def test_attempt_cap_hit_before_revision_rejects_flagged(tmp_path, settings):
    recs, ver = records()
    t = BudgetTracker(BudgetLimits(max_sources=10, max_attempts=6, max_spend_usd=None))
    result, client = run(tmp_path, settings, full_script(recs, hyps={"hypotheses": [hypothesis(hypothesis=CAUSAL)]}),
                         recs=recs, ver=ver, tracker=t)
    assert len(client.requests) == 6
    hyps = load(result, "hypotheses")
    assert hyps["accepted"] == [] and "revision_not_run" in hyps["rejected"][0]["reason_codes"]
    assert CAUSAL not in run_files_text(result)


def test_spend_accounting_includes_critic_and_revision(tmp_path, settings):
    recs, ver = records()
    prices = PriceTable(input_per_mtok=Decimal("3"), output_per_mtok=Decimal("15"))
    t = BudgetTracker(BudgetLimits(max_sources=10, max_attempts=15, max_spend_usd=Decimal("2")), prices)
    script = full_script(recs, hyps={"hypotheses": [hypothesis(hypothesis=CAUSAL)]},
                         revision={"hypotheses": [revised()]})
    result, client = run(tmp_path, settings, script, recs=recs, ver=ver, tracker=t)
    used = result.budget["used"]
    per_attempt = (Decimal(100) * 3 + Decimal(50) * 15) / Decimal(1_000_000)   # FakeModelClient usage
    assert used["attempts"] == 8 and Decimal(used["spend_usd"]) == 8 * per_attempt
    assert Decimal(str(result.budget["limits"]["max_spend_usd"])) == Decimal("2")


# ================================================================== adversarial fixtures


ADV_CAUSAL = hypothesis(hypothesis_id="a1", hypothesis="Shear causes platelet activation through GPIb.",
                        mechanistic_claim_level="causal_claim", confidence="high")
ADV_VAGUE = hypothesis(hypothesis_id="a2", prediction="GPIb will turn out to be important.")
ADV_REJECTED_EV = hypothesis(hypothesis_id="a3", evidence_ids=["ev_0001", "ev_0003"])      # ev_0003 was rejected
ADV_DOI = hypothesis(hypothesis_id="a4", rationale="Shown in doi 10.9999/fake.2020.1 for GPIb.")
ADV_AUTHOR_YEAR = hypothesis(hypothesis_id="a5", rationale="Marchetti-Oyelaran (1993) reported GPIb binding.")
ADV_DISCOVERY = hypothesis(hypothesis_id="a6", hypothesis="This discovery shows GPIb may reduce P-selectin.")
ADVERSARIAL_TEXTS = ["Shear causes platelet activation through GPIb.", "GPIb will turn out to be important.",
                     "10.9999/fake.2020.1", "Marchetti-Oyelaran (1993)", "This discovery shows"]


def adversarial_script(recs, revision):
    return full_script(recs, hyps={"hypotheses": [ADV_CAUSAL, ADV_VAGUE, ADV_REJECTED_EV, ADV_DOI, ADV_AUTHOR_YEAR]},
                       revision=revision)


def test_adversarial_outputs_rejected_or_corrected_deterministically(tmp_path, settings):
    recs, ver = records()
    # revision: fixes the causal one (lower level, hedged); returns the vague prediction unchanged
    revision = {"hypotheses": [
        hypothesis(hypothesis_id="hyp_01", hypothesis="Shear might activate platelets partly through GPIb.",
                   mechanistic_claim_level="mechanistic_support", confidence="high"),
        hypothesis(hypothesis_id="hyp_02", prediction="GPIb will turn out to be important.")]}
    result, client = run(tmp_path, settings, adversarial_script(recs, revision), recs=recs, ver=ver)
    assert client.remaining == 0
    hyps = load(result, "hypotheses")
    critic_payload = json.loads(client.requests[5].messages[-1].content)
    flags = {c["hypothesis_id"]: c["deterministic_flags"] for c in critic_payload["candidate_hypotheses"]}
    assert set(flags) == {"hyp_01", "hyp_02"}                           # hard failures never reach the critic
    assert flags["hyp_01"] == ["claim_level_exceeds_evidence"]         # causal level claimed, evidence: mechanistic
    assert flags["hyp_02"] == ["prediction_not_measurable"]
    gen = {r["item_index"]: r for r in hyps["rejected"] if r["rejected_at"] == "generation"}
    by_model = {"a3": gen[2], "a4": gen[3], "a5": gen[4]}                # generation order a1..a5
    assert "unknown_evidence_id" in by_model["a3"]["reason_codes"]       # citing rejected evidence
    assert set(by_model["a4"]["reason_codes"]) & {"identifier_in_text", "bibliographic_text"}
    assert "bibliographic_text" in by_model["a5"]["reason_codes"]
    assert sorted(gen) == [2, 3, 4]
    unresolved = [r for r in hyps["rejected"] if r["rejected_at"] == "revision"][0]
    assert {"prediction_not_measurable", "unresolved_after_revision"} <= set(unresolved["reason_codes"])
    [ok] = hyps["accepted"]
    assert ok["mechanistic_claim_level"] == "mechanistic_support" and ok["confidence"] == "moderate"
    assert ok["confidence_detail"]["proposed_by_model"] == "high"
    text = run_files_text(result)
    for needle in ADVERSARIAL_TEXTS:
        assert needle not in text, needle


def test_self_labelled_discovery_rejected_when_revision_keeps_it(tmp_path, settings):
    recs, ver = records()
    script = full_script(recs, hyps={"hypotheses": [ADV_DISCOVERY]},
                         revision={"hypotheses": [hypothesis(hypothesis_id="hyp_01",
                                                             hypothesis=ADV_DISCOVERY["hypothesis"])]})
    result, _ = run(tmp_path, settings, script, recs=recs, ver=ver)
    hyps = load(result, "hypotheses")
    assert hyps["accepted"] == [] and "self_labeled_discovery" in hyps["rejected"][0]["reason_codes"]
    assert "This discovery shows" not in run_files_text(result)
    report = result.files["report"].read_text(encoding="utf-8")
    assert "No candidate hypotheses passed the deterministic checks." in report


def test_critic_and_revision_prompts_redacted_when_candidate_rejected(tmp_path, settings):
    recs, ver = records()
    script = full_script(recs, hyps={"hypotheses": [hypothesis(hypothesis=CAUSAL)]},
                         revision={"hypotheses": [hypothesis(hypothesis_id="hyp_01", hypothesis=CAUSAL)]})
    result, _ = run(tmp_path, settings, script, recs=recs, ver=ver)
    es = {e["stage"]: e for e in entries(result)}
    for stage in ("hypothesis_critic", "hypothesis_revision"):
        assert es[stage]["redaction_mode"] == "safe_diagnostics"
        user = [m for m in es[stage]["request"]["messages"] if m["role"] == "user"][0]
        assert json.loads(user["content"])["request_payload"].startswith("[redacted")
    assert CAUSAL not in run_files_text(result)


# ================================================================== report wording


DISCOVERY_CLAIM = re.compile(r"\b(?:is|are|represents?|constitutes?)\s+(?:a\s+|an\s+)?(?:validated|confirmed|proven)"
                             r"\s+(?:discover\w*|finding\w*|mechanism\w*)", re.IGNORECASE)


def test_report_sections_and_per_hypothesis_notice(tmp_path, settings):
    recs, ver = records()
    result, _ = run(tmp_path, settings, full_script(recs), recs=recs, ver=ver)
    report = result.files["report"].read_text(encoding="utf-8")
    section = report.split("## H.")[1].split("## I.")[0]
    assert HYPOTHESIS_DISCLAIMER in section
    assert f"### hyp_01 — {HYPOTHESIS_NOTICE}" in section
    assert HYPOTHESIS_NOTICE == ("Unvalidated, AI-generated hypothesis for further investigation — not a validated "
                                 "discovery or scientific finding.")
    for heading in ("**Label:** " + HYPOTHESIS_LABEL, "**Candidate hypothesis:**", "**Claim level:**",
                    "Causality is not established", "**Why it was proposed:**", "research gap gap_01",
                    "linked evidence: ev_0001", "**Prediction (measurable):**", "**Alternative explanation:**",
                    "basis: inference", "**How it could be falsified:**", "would weaken it:", "would support it:",
                    "**Limitations:**", "**Source quality:**", "**Confidence (qualitative):**", "**Critic review:**",
                    "**Revision status:** no revision required", "**Validation status:** passed deterministic"):
        assert heading in section, heading
    assert "## C." in report and "## G." in report                         # evidence and gaps kept separate
    assert not DISCOVERY_CLAIM.search(report)
    assert "%" not in section.split("**Confidence (qualitative):**")[1].split("\n")[0]


def test_web_hypothesis_view_carries_label_and_stress_test():
    from sciforge import app_service as svc
    from sciforge.demo_data import DEMO_QUESTION

    r = svc.run_web_investigation(svc.InvestigationRequest(question=DEMO_QUESTION), environ={})
    assert r.ok
    for h in r.hypotheses:
        assert h["notice"] == HYPOTHESIS_NOTICE and h["label"] == HYPOTHESIS_LABEL
        assert h["critic_status"] == "completed" and h["confidence"] in ("low", "moderate", "high")
        assert set(h["falsification_test"]) == {"manipulated_or_compared", "measured", "weakening_result",
                                                "supporting_result"}
    st = r.validation["stages"]["hypotheses"]
    assert st["critic_status"] == "completed" and st["revision_status"] == "completed" and st["revised"] == 1
    assert r.validation["budget"]["attempts_used"] == 9 <= 15
    assert r.validation["privacy_guard"]["rejected_output_leaks_in_run_files"] == 0
    section = r.sections["H. Candidate Hypotheses"]
    assert section.count(HYPOTHESIS_NOTICE) == 2
    assert "drives the larger effect sizes" not in r.report_markdown              # the over-claim was revised


def test_demo_hypotheses_show_flags_critic_and_revision():
    from sciforge import demo_data

    assert {"hypothesis_critic", "hypothesis_revision"} <= set(demo_data._RESPONDERS)


# ================================================================== secrets / reasoning


def test_no_secrets_in_run_files_even_if_critic_echoes_key(tmp_path, settings):
    recs, ver = records()
    ms = ModelSettings.from_env(model_env(SCIFORGE_MODEL_MAX_ATTEMPTS="30"))
    critic = {"reviews": [critic_review("hyp_01", explanation=f"token {FAKE_XAI_KEY} looks fine")]}
    result, _ = run(tmp_path, settings, full_script(recs, critic=critic), recs=recs, ver=ver, model_settings=ms,
                    tracker=None)
    text = run_files_text(result)
    for needle in (FAKE_XAI_KEY, settings.ncbi_api_key, settings.contact_email):
        assert needle not in text


def test_no_reasoning_keys_in_hypotheses_json(tmp_path, settings):
    recs, ver = records()
    result, _ = run(tmp_path, settings, full_script(recs, hyps={"hypotheses": [hypothesis(hypothesis=CAUSAL)]},
                                                    revision={"hypotheses": [revised()]}), recs=recs, ver=ver)
    blob = result.files["hypotheses"].read_text(encoding="utf-8")
    for key in ('"reasoning"', '"chain_of_thought"', '"thinking"', '"reasoning_trace"', '"encrypted_content"',
                '"probability"'):
        assert key not in blob
