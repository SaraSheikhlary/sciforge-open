"""FakeModelClient: the only model client used by automated tests (D7)."""

import pytest

from sciforge.llm.client import (
    ModelClient,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponseParseError,
    ModelTimeout,
    ModelUsage,
)
from sciforge.llm.fake import DEFAULT_FAKE_USAGE, FakeModelClient

SCHEMA = {"type": "object", "properties": {}, "additionalProperties": False}


def req(structured=True, text="hi"):
    kw = {"schema_name": "s", "json_schema": SCHEMA} if structured else {}
    return ModelRequest(messages=(ModelMessage("user", text),), max_output_tokens=100, **kw)


def test_scripted_order_and_recording():
    fake = FakeModelClient([{"n": 1}, '{"n": 2}', "```json\n{\"n\": 3}\n```"])
    assert [fake.complete(req(text=str(i))).parsed for i in range(3)] == [{"n": 1}, {"n": 2}, {"n": 3}]
    assert [r.messages[0].content for r in fake.requests] == ["0", "1", "2"]
    assert fake.remaining == 0


def test_exhaustion_raises_assertion():
    fake = FakeModelClient([])
    with pytest.raises(AssertionError, match="exhausted"):
        fake.complete(req())


def test_raises_scripted_errors():
    fake = FakeModelClient([ModelTimeout("t")])
    with pytest.raises(ModelTimeout):
        fake.complete(req())


def test_callable_items_see_request():
    fake = FakeModelClient([lambda r: {"echo": r.messages[-1].content}])
    assert fake.complete(req(text="ping")).parsed == {"echo": "ping"}


def test_invalid_json_for_structured_request():
    fake = FakeModelClient(["not json"])
    with pytest.raises(ModelResponseParseError) as info:
        fake.complete(req())
    assert info.value.usage == DEFAULT_FAKE_USAGE


def test_unstructured_text_and_response_passthrough():
    resp = ModelResponse(text="x", parsed=None, usage=ModelUsage(), model="m")
    fake = FakeModelClient(["plain", resp])
    assert fake.complete(req(structured=False)).parsed is None
    assert fake.complete(req()) is resp


def test_reports_usage_and_protocol():
    fake = FakeModelClient([{}], usage=ModelUsage(input_tokens=5, output_tokens=2), model="fm").add({})
    r = fake.complete(req())
    assert r.usage.input_tokens == 5 and r.model == "fm" and r.provider == "fake"
    assert isinstance(fake, ModelClient) and fake.remaining == 1


def test_unsupported_item_type():
    with pytest.raises(TypeError):
        FakeModelClient([42]).complete(req())
