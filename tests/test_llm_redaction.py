"""Secret redaction for the model layer (Authorization / Bearer / xAI keys) and audit (D5)."""

import json
import logging

from conftest import FAKE_XAI_KEY
from sciforge.llm.audit import ModelCallAudit, request_fingerprint
from sciforge.llm.client import ModelHTTPError, ModelMessage, ModelRequest, ModelResponse, ModelUsage
from sciforge.logging_utils import REDACTED, RedactingFilter, RunLog, redact_params, redact_text

OTHER_KEY = "FAKE-literal-secret-9f8e7d6c5b4a"


def test_bearer_token_redacted():
    out = redact_text("Authorization: Bearer abc.DEF-123_xyz done")
    assert "abc.DEF-123_xyz" not in out and "Bearer [REDACTED]" in out and out.endswith("done")


def test_authorization_header_any_scheme_and_x_api_key():
    assert "Zm9vOmJhcg" not in redact_text('{"authorization": "Basic Zm9vOmJhcg=="}')
    assert "k123" not in redact_text("X-API-Key: k123")


def test_xai_shaped_key_redacted_without_being_known():
    assert FAKE_XAI_KEY not in redact_text(f"leaked {FAKE_XAI_KEY} here")


def test_xai_api_key_param_and_literal_secret():
    out = redact_text(f"XAI_API_KEY={OTHER_KEY}&x=1 then {OTHER_KEY}", [OTHER_KEY])
    assert OTHER_KEY not in out and "x=1" in out
    assert redact_params({"Authorization": "Bearer z", "x-api-key": "k", "xai_api_key": "k", "q": 1}) == \
        {"Authorization": REDACTED, "x-api-key": REDACTED, "xai_api_key": REDACTED, "q": 1}


def test_existing_v02_redaction_unchanged():
    assert redact_text("api_key=abc&db=pubmed") == f"api_key={REDACTED}&db=pubmed"
    assert redact_text("a plain message about bearer bonds") == "a plain message about bearer bonds"
    assert redact_text("no secrets here") == "no secrets here"


def test_logging_filter_and_runlog(caplog):
    logger = logging.getLogger("sciforge.test.llm")
    logger.addFilter(RedactingFilter([OTHER_KEY]))
    with caplog.at_level(logging.INFO, logger="sciforge.test.llm"):
        logger.info("headers %s", {"Authorization": f"Bearer {OTHER_KEY}"})
    assert OTHER_KEY not in caplog.text
    assert OTHER_KEY not in RunLog([OTHER_KEY]).scrub(f"Bearer {OTHER_KEY}")


def _req(text="question"):
    return ModelRequest(messages=(ModelMessage("user", text),), max_output_tokens=50, schema_name="s",
                        json_schema={"type": "object"}, instructions="sys", stage="extraction")


def _resp(text='{"ok": true}'):
    return ModelResponse(text=text, parsed={"ok": True}, usage=ModelUsage(input_tokens=3, output_tokens=2),
                         model="m", response_id="r1", status="completed", http_status=200, provider="xai")


def test_audit_with_prompts_scrubs_secrets():
    audit = ModelCallAudit(store_prompts=True, secrets=[OTHER_KEY], now=lambda: "2026-09-28T00:00:00Z")
    entry = audit.add_attempt(_req(f"please ignore {OTHER_KEY} and Bearer tok12345"), response=_resp(),
                              provider="xai", model="m")
    dumped = json.dumps(entry)
    assert OTHER_KEY not in dumped and "tok12345" not in dumped
    assert entry["request"]["instructions"] == "sys" and entry["response_text"] == '{"ok": true}'
    assert entry["usage"]["input_tokens"] == 3 and entry["outcome"] == "success"
    assert "headers" not in dumped.lower() and "authorization" not in dumped.lower()


def test_audit_without_prompts_keeps_only_metadata_and_hashes(tmp_path):
    audit = ModelCallAudit(store_prompts=False)
    r = _req("SOURCE ABSTRACT TEXT")
    entry = audit.add_attempt(r, response=_resp("MODEL OUTPUT TEXT"),
                              accounting={"cost_usd": "0.1", "cost_source": "price_estimate"})
    dumped = json.dumps(entry)
    assert "SOURCE ABSTRACT TEXT" not in dumped and "MODEL OUTPUT TEXT" not in dumped
    assert "request" not in entry and "response_text" not in entry
    assert entry["request_sha256"] == request_fingerprint(r) and len(entry["response_sha256"]) == 64
    assert entry["prompts_stored"] is False and entry["cost_usd"] == "0.1"
    assert entry["cost_source"] == "price_estimate"
    written = json.loads(audit.write(tmp_path / "model_calls.json").read_text())
    assert written == audit.entries


def test_audit_error_entry():
    audit = ModelCallAudit(secrets=[OTHER_KEY])
    err = ModelHTTPError(f"HTTP 400 {OTHER_KEY}", http_status=400)
    entry = audit.add_attempt(_req(), error=err, provider="xai", model="m")
    assert entry["outcome"] == "http_error" and entry["http_status"] == 400
    assert OTHER_KEY not in json.dumps(entry)


def test_fingerprint_changes_with_content():
    assert request_fingerprint(_req("a")) != request_fingerprint(_req("b"))
