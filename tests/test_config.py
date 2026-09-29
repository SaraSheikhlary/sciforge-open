"""Settings from environment."""

import pytest

from sciforge.config import ConfigError, Settings


def test_defaults_without_env():
    s = Settings.from_env({})
    assert s.ncbi_api_key is None and s.contact_email is None
    assert s.max_retries == 2 and s.timeout_seconds == 20.0
    assert "mailto" not in s.user_agent
    assert s.user_agent.startswith("SciForge/0.4")
    assert s.pubmed_min_interval >= 1 / 3


def test_values_from_env():
    s = Settings.from_env({"NCBI_API_KEY": " k-123456 ", "SCIFORGE_CONTACT_EMAIL": "me@example.org",
                           "SCIFORGE_TIMEOUT_SECONDS": "5", "SCIFORGE_MAX_RETRIES": "0"})
    assert s.ncbi_api_key == "k-123456"
    assert s.user_agent == "SciForge/0.4.0 (https://github.com/SaraSheikhlary/sciforge-open; mailto:me@example.org)"
    assert s.timeout_seconds == 5.0 and s.max_retries == 0
    assert s.pubmed_min_interval >= 0.1 and s.pubmed_min_interval < 1 / 3


def test_repr_hides_secrets():
    s = Settings(ncbi_api_key="super-secret-key", contact_email="me@example.org")
    assert "super-secret-key" not in repr(s) and "me@example.org" not in repr(s)


@pytest.mark.parametrize("env", [{"SCIFORGE_TIMEOUT_SECONDS": "abc"}, {"SCIFORGE_TIMEOUT_SECONDS": "0"},
                                 {"SCIFORGE_MAX_RETRIES": "99"}, {"SCIFORGE_CONTACT_EMAIL": "not-an-email"}])
def test_invalid_env(env):
    with pytest.raises(ConfigError):
        Settings.from_env(env)
