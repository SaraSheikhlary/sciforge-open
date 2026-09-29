"""Scripted, offline :class:`~sciforge.llm.client.ModelClient` for tests (D7).

Script items, consumed in order, one per ``complete()`` call:

* ``dict`` / ``list``  → JSON payload (``text = json.dumps(item)``)
* ``str``              → raw model text (parsed as JSON if a schema was requested)
* ``ModelResponse``    → returned as-is
* ``ModelError``       → raised
* callable(request)    → called; its return value is handled like the above

Every request is recorded in ``requests`` for assertions.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from typing import Any, Union

from sciforge.llm.client import ModelError, ModelRequest, ModelResponse, ModelUsage
from sciforge.llm.parsing import parse_json_text

ScriptItem = Union[dict, list, str, ModelResponse, ModelError, Callable[[ModelRequest], Any]]

DEFAULT_FAKE_USAGE = ModelUsage(input_tokens=100, output_tokens=50, total_tokens=150)


class FakeModelClient:
    """Deterministic fake model; never performs I/O."""

    name = "fake"

    def __init__(self, script: Iterable[ScriptItem] = (), *, model: str = "fake-model",
                 usage: ModelUsage | None = DEFAULT_FAKE_USAGE) -> None:
        self.model = model
        self._script: list[ScriptItem] = list(script)
        self._usage = usage if usage is not None else ModelUsage(reported=False)
        self.requests: list[ModelRequest] = []

    def add(self, *items: ScriptItem) -> FakeModelClient:
        self._script.extend(items)
        return self

    @property
    def remaining(self) -> int:
        return len(self._script)

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self._script:
            raise AssertionError("FakeModelClient script exhausted (unexpected extra model call)")
        item: Any = self._script.pop(0)
        if callable(item) and not isinstance(item, (ModelResponse, ModelError)):
            item = item(request)
        if isinstance(item, ModelError):
            raise item
        if isinstance(item, ModelResponse):
            return item
        if isinstance(item, (dict, list)):
            text = json.dumps(item)
        elif isinstance(item, str):
            text = item
        else:
            raise TypeError(f"unsupported FakeModelClient script item: {type(item).__name__}")
        parsed = parse_json_text(text, usage=self._usage) if request.structured else None
        return ModelResponse(text=text, parsed=parsed, usage=self._usage, model=self.model,
                             response_id=f"fake-{len(self.requests)}", status="completed",
                             http_status=200, latency_s=0.0, attempts=1, provider=self.name)
