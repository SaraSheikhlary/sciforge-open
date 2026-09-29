"""ModelSettings (v0.3 model layer configuration, decisions D3/D5/D8)."""

import pytest

from conftest import FAKE_XAI_KEY, FAKE_XAI_MODEL, model_env
from sciforge.config import ConfigError, ModelSettings, Settings
from sciforge.llm.budget import BudgetLimits, BudgetTracker
from sciforge.llm.client import ModelConfigError


def test_defaults_match_d8():
    s = ModelSettings.from_env(model_env(SCIFORGE_MAX_SPEND_USD=None,
                                         SCIFORGE_PRICE_INPUT_PER_MTOK="1", SCIFORGE_PRICE_OUTPUT_PER_MTOK="2"))
    assert s.max_attempts == 15                      # every API attempt counts (retries included)
    assert s.max_sources == 10
    assert s.max_input_tokens == 200_000
    assert s.max_output_tokens_per_call == 2_000
    assert s.max_spend_usd == 15.0
    assert s.base_url == "https://api.x.ai"
    assert s.responses_url == "https://api.x.ai/v1/responses"
    assert s.store_prompts is True
    assert s.entailment is True                      # D4
    assert s.eligibility == "verified" and not s.include_partially_verified  # D2
    assert s.budget_limits() == BudgetLimits()
    assert BudgetLimits() == BudgetLimits(15, 10, 200_000, 2_000, 15.0)
    assert s.retry_policy().max_retries == 2 and s.retry_policy().backoff_seconds == 1.0


def test_xai_model_required_no_default():
    with pytest.raises(ModelConfigError, match="XAI_MODEL"):
        ModelSettings.from_env({"XAI_API_KEY": FAKE_XAI_KEY, "SCIFORGE_MAX_SPEND_USD": "none"})
    with pytest.raises(ModelConfigError, match="XAI_MODEL"):
        ModelSettings.from_env(model_env(XAI_MODEL="   "))


def test_missing_key_error_only_when_model_requested():
    # v0.2 path: Settings.from_env never needs or reads the key / model vars.
    Settings.from_env({})
    Settings.from_env({"XAI_MODEL": "x", "XAI_BASE_URL": "http://bad", "SCIFORGE_MAX_SPEND_USD": "garbage"})
    with pytest.raises(ModelConfigError, match="XAI_API_KEY"):
        ModelSettings.from_env({"XAI_MODEL": FAKE_XAI_MODEL, "SCIFORGE_MAX_SPEND_USD": "none"})


def test_v02_settings_do_not_expose_xai_key():
    s = Settings.from_env({"XAI_API_KEY": FAKE_XAI_KEY})
    assert FAKE_XAI_KEY not in repr(s) and FAKE_XAI_KEY not in s.secret_values()


def test_key_never_in_repr_or_str():
    s = ModelSettings.from_env(model_env())
    assert s.api_key == FAKE_XAI_KEY
    assert FAKE_XAI_KEY not in repr(s)
    assert FAKE_XAI_KEY not in str(s)
    assert FAKE_XAI_KEY not in f"{s}"
    assert s.secret_values() == [FAKE_XAI_KEY]


def test_reads_os_environ_by_default(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", FAKE_XAI_KEY)
    monkeypatch.setenv("XAI_MODEL", "m1")
    monkeypatch.setenv("SCIFORGE_MAX_SPEND_USD", "none")
    assert ModelSettings.from_env().model == "m1"


def test_fail_closed_when_cap_enabled_and_prices_unset():
    with pytest.raises(ModelConfigError, match="SCIFORGE_PRICE_INPUT_PER_MTOK"):
        ModelSettings.from_env(model_env(SCIFORGE_MAX_SPEND_USD=None))       # default $15 cap, no prices
    with pytest.raises(ModelConfigError):
        ModelSettings.from_env(model_env(SCIFORGE_MAX_SPEND_USD="5", SCIFORGE_PRICE_INPUT_PER_MTOK="1"))
    with pytest.raises(ModelConfigError):
        ModelSettings(api_key=FAKE_XAI_KEY, model="m")                    # dataclass default also fails closed


@pytest.mark.parametrize("raw", ["none", "NONE", " None "])
def test_spend_cap_can_be_disabled(raw):
    s = ModelSettings.from_env(model_env(SCIFORGE_MAX_SPEND_USD=raw))
    assert s.max_spend_usd is None and not s.prices_configured
    assert s.budget_limits().max_spend_usd is None


@pytest.mark.parametrize("raw", ["0", "-1", "0.0", "abc", "10001", "off", "disabled"])
def test_spend_cap_invalid_values_rejected(raw):
    with pytest.raises(ModelConfigError, match="SCIFORGE_MAX_SPEND_USD"):
        ModelSettings.from_env(model_env(SCIFORGE_MAX_SPEND_USD=raw, SCIFORGE_PRICE_INPUT_PER_MTOK="1",
                                         SCIFORGE_PRICE_OUTPUT_PER_MTOK="1"))


def test_prices_and_custom_limits_parsed():
    s = ModelSettings.from_env(model_env(
        SCIFORGE_MAX_SPEND_USD="2.5", SCIFORGE_PRICE_INPUT_PER_MTOK="3", SCIFORGE_PRICE_OUTPUT_PER_MTOK="15",
        SCIFORGE_MODEL_MAX_ATTEMPTS="5", SCIFORGE_MODEL_MAX_SOURCES="3", SCIFORGE_MODEL_MAX_INPUT_TOKENS="5000",
        SCIFORGE_MODEL_MAX_OUTPUT_TOKENS="500", SCIFORGE_MODEL_TIMEOUT_SECONDS="30",
        SCIFORGE_MAX_RETRIES="1", SCIFORGE_BACKOFF_SECONDS="0"))
    assert (s.max_spend_usd, s.price_input_per_mtok, s.price_output_per_mtok) == (2.5, 3.0, 15.0)
    assert (s.max_attempts, s.max_sources, s.max_input_tokens, s.max_output_tokens_per_call) == (5, 3, 5000, 500)
    assert (s.timeout_seconds, s.max_retries, s.backoff_seconds) == (30.0, 1, 0.0)
    tracker = BudgetTracker(s.budget_limits(), s.price_table())
    assert tracker.prices.configured


@pytest.mark.parametrize("name,value", [
    ("SCIFORGE_MODEL_MAX_ATTEMPTS", "0"), ("SCIFORGE_MODEL_MAX_ATTEMPTS", "x"), ("SCIFORGE_MODEL_MAX_SOURCES", "101"),
    ("SCIFORGE_MODEL_MAX_INPUT_TOKENS", "10"), ("SCIFORGE_MODEL_MAX_OUTPUT_TOKENS", "1"),
    ("SCIFORGE_PRICE_INPUT_PER_MTOK", "-1"),
    ("SCIFORGE_STORE_PROMPTS", "maybe"), ("SCIFORGE_MODEL_ENTAILMENT", "2"),
    ("SCIFORGE_MODEL_ELIGIBILITY", "partial"),
])
def test_invalid_values_raise_model_config_error(name, value):
    with pytest.raises(ModelConfigError) as info:
        ModelSettings.from_env(model_env(**{name: value}))
    assert not isinstance(info.value, ConfigError)  # distinct from v0.2 ConfigError (exit 2 vs 4)


@pytest.mark.parametrize("raw,expected", [
    (None, "https://api.x.ai"), ("https://api.x.ai/", "https://api.x.ai"),
    ("https://api.x.ai/v1", "https://api.x.ai"), ("https://proxy.example.org/xai", "https://proxy.example.org/xai"),
])
def test_base_url_normalized(raw, expected):
    s = ModelSettings.from_env(model_env(XAI_BASE_URL=raw))
    assert s.base_url == expected and s.responses_url == expected + "/v1/responses"


@pytest.mark.parametrize("raw", ["http://api.x.ai", "ftp://x", "https://user:pw@api.x.ai", "https://api.x.ai?x=1",
                                 "https://"])
def test_base_url_must_be_plain_https(raw):
    with pytest.raises(ModelConfigError, match="XAI_BASE_URL"):
        ModelSettings.from_env(model_env(XAI_BASE_URL=raw))


def test_store_prompts_entailment_and_eligibility_toggles():
    s = ModelSettings.from_env(model_env(SCIFORGE_STORE_PROMPTS="false", SCIFORGE_MODEL_ENTAILMENT="0",
                                         SCIFORGE_MODEL_ELIGIBILITY="verified_or_partial"))
    assert s.store_prompts is False and s.entailment is False and s.include_partially_verified


def test_no_xai_store_setting_exists():
    # D1: store is hard-coded false; there is deliberately no config knob for it.
    assert not any("store" in f and f != "store_prompts" for f in ModelSettings.__dataclass_fields__)
