"""Redaction helpers and timestamp formats."""

import logging
from datetime import datetime, timezone

from sciforge.logging_utils import (
    REDACTED,
    RedactingFilter,
    iso_utc,
    redact_params,
    redact_text,
    redact_url,
    run_stamp,
)

KEY = "abcdef0123456789secret"


def test_redact_params():
    params = {"term": "x", "api_key": KEY, "email": "a@example.org", "mailto": "a@example.org", "API_KEY": KEY, "tool": "sciforge"}
    out = redact_params(params)
    assert out["term"] == "x" and out["tool"] == "sciforge"
    assert out["api_key"] == out["email"] == out["mailto"] == out["API_KEY"] == REDACTED
    assert params["api_key"] == KEY  # original untouched
    assert redact_params(None) == {}


def test_redact_url():
    url = f"https://eutils.ncbi.nlm.nih.gov/esearch.fcgi?db=pubmed&api_key={KEY}&term=a+b&email=a%40b.org"
    out = redact_url(url)
    assert KEY not in out and "a%40b.org" not in out
    assert "db=pubmed" in out and "term=a+b" in out
    assert "api_key=%5BREDACTED%5D" in out or "api_key=[REDACTED]" in out
    assert redact_url("https://api.crossref.org/works/10.1000/x") == "https://api.crossref.org/works/10.1000/x"


def test_redact_text():
    msg = f"GET https://x?api_key={KEY}&db=1 failed; key {KEY} and mailto=me@example.net"
    out = redact_text(msg, [KEY])
    assert KEY not in out and "me@example.net" not in out
    assert "db=1" in out


def test_redacting_filter(caplog):
    logger = logging.getLogger("sciforge.test")
    logger.addFilter(RedactingFilter([KEY]))
    with caplog.at_level(logging.INFO, logger="sciforge.test"):
        logger.info("using key %s", KEY)
    assert KEY not in caplog.text and REDACTED in caplog.text


def test_timestamps():
    dt = datetime(2026, 9, 28, 23, 41, 0, tzinfo=timezone.utc)
    assert iso_utc(dt) == "2026-09-28T23:41:00Z"
    assert run_stamp(dt) == "20260928T234100Z"
