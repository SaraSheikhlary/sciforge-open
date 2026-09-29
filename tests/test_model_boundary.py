"""v0.3 M2 model input boundary: allowlist, determinism, bibliographic data never in requests."""

from __future__ import annotations

import json
import re

import pytest
from pydantic import ValidationError

from m2_support import BIB, STRUCTURED_ABSTRACT_TEXT, TS, item, question_output, record
from sciforge.boundary import MODEL_SOURCE_FIELDS, ModelSource, canonical_json, model_sources, source_payload, to_model_source
from sciforge.llm.budget import BudgetLimits, BudgetTracker
from sciforge.llm.fake import FakeModelClient
from sciforge.sourcetext import SourceText, sha256_text
from sciforge.stages.common import CallContext
from sciforge.stages.extraction import run_extraction_stage
from sciforge.stages.question import run_question_stage


def source_text(rec, text=STRUCTURED_ABSTRACT_TEXT, **kw) -> SourceText:
    base = dict(record_id=rec.record_id, verification_status="verified", eligibility="verified", status="ok",
                access_level="pubmed_abstract", origin="pubmed", source_text=text, sha256=sha256_text(text) if text else None,
                original_chars=len(text or ""), text_chars=len(text or ""), max_source_chars=4000, retrieved_at=TS)
    base.update(kw)
    return SourceText(**base)


def request_blob(client: FakeModelClient) -> str:
    """Everything that would be sent to the provider, for every request."""
    parts = []
    for r in client.requests:
        parts.append(json.dumps({"messages": [m.to_dict() for m in r.messages], "instructions": r.instructions,
                                 "schema_name": r.schema_name, "schema": r.json_schema}, ensure_ascii=False))
    return "\n".join(parts)


def test_model_source_has_exactly_the_allowlisted_fields():
    assert tuple(ModelSource.model_fields) == MODEL_SOURCE_FIELDS == ("record_id", "access_level", "source_text")
    with pytest.raises(ValidationError):
        ModelSource(record_id="rec_0123456789abcdef", access_level="pubmed_abstract", source_text="x", title="t")
    with pytest.raises(ValidationError):
        ModelSource(record_id="10.1000/abc", access_level="pubmed_abstract", source_text="x")
    with pytest.raises(ValidationError):
        ModelSource(record_id="rec_0123456789abcdef", access_level="full_text", source_text="x")


def test_to_model_source_uses_only_id_access_level_and_text():
    rec = record()
    ms = to_model_source(source_text(rec))
    assert ms.model_dump() == {"record_id": rec.record_id, "access_level": "pubmed_abstract",
                               "source_text": STRUCTURED_ABSTRACT_TEXT}


def test_unusable_sources_are_not_converted():
    rec = record()
    missing = source_text(rec, text=None, status="no_abstract", access_level="not_accessed", sha256=None)
    with pytest.raises(ValueError):
        to_model_source(missing)
    assert model_sources([missing, source_text(rec)]) == [to_model_source(source_text(rec))]


def test_serialisation_is_deterministic():
    rec = record()
    payload = source_payload([to_model_source(source_text(rec))])
    a = canonical_json({"sources": payload, "b": 1, "a": [2]})
    b = canonical_json({"a": [2], "b": 1, "sources": payload})
    assert a == b and a.startswith('{"a":[2],"b":1,"sources":[{"access_level"')


def test_abstract_text_passed_unaltered_even_with_urls():
    rec = record()
    text = "Trial registered at https://example.org/trial NCT01234567; outcomes improved by 12%."
    assert to_model_source(source_text(rec, text=text)).source_text == text


def test_no_bibliographic_value_or_key_in_any_model_request():
    rec = record()
    assert rec.title and rec.authors and rec.journal and rec.year and rec.doi and rec.pmid and rec.source_url
    st = source_text(rec)
    quote = "Shear exposure increased P-selectin expression by 40% compared with static controls."
    client = FakeModelClient([question_output(), {"items": [item(rec.record_id, quote)]}])
    ctx = CallContext(client=client, tracker=BudgetTracker(BudgetLimits(max_spend_usd=None)), sleep=lambda s: None)
    q = run_question_stage(ctx, "Does high shear stress directly activate human platelets?")
    ev = run_extraction_stage(ctx, q.definition.model_dump(), [st], question_definition_used=True)
    assert len(ev.accepted) == 1 and len(client.requests) == 2

    blob = request_blob(client)
    assert rec.record_id in blob and STRUCTURED_ABSTRACT_TEXT in blob
    for value in (rec.title, rec.journal, rec.doi, rec.pmid, rec.source_url, *rec.authors, BIB["doi"]):
        assert value not in blob, value
    assert not re.search(rf"\b{rec.year}\b", blob)
    for surname in ("Albemarle", "Perpetua"):
        assert surname not in blob
    for key in ("title", "authors", "author", "journal", "year", "doi", "pmid", "pmcid", "url", "source_url"):
        assert not re.search(rf"\b{key}\b", blob, re.IGNORECASE), key
    # the extraction request's sources carry only the three allowlisted keys
    user_payload = json.loads(client.requests[1].messages[0].content)
    assert [set(s) for s in user_payload["sources"]] == [set(MODEL_SOURCE_FIELDS)]
    assert set(user_payload) == {"question_definition", "sources"}


def test_question_request_contains_only_the_question():
    client = FakeModelClient([question_output()])
    ctx = CallContext(client=client, tracker=BudgetTracker(BudgetLimits(max_spend_usd=None)), sleep=lambda s: None)
    run_question_stage(ctx, "Q?")
    (req,) = client.requests
    assert [json.loads(m.content) for m in req.messages] == [{"research_question_verbatim": "Q?"}]
    assert "source_text" not in req.messages[0].content
