"""Defensive parsing of xAI Responses API payloads and structured output.

Payload shape (docs.x.ai API reference, verified 2026-09-28)::

    {"id": ..., "model": ..., "status": "completed" | "in_progress" | "incomplete",
     "incomplete_details": {...} | null, "error": {...} | null,
     "output": [{"type": "message", "role": "assistant",
                 "content": [{"type": "output_text", "text": "..."}]},
                {"type": "reasoning", ...}],
     "usage": {"input_tokens", "output_tokens", "total_tokens",
               "input_tokens_details": {"cached_tokens"},
               "output_tokens_details": {"reasoning_tokens"},
               "cost_in_usd_ticks", "cost_in_nano_usd"}}

Anything unexpected becomes a typed :class:`ModelResponseParseError` rather
than an exception from deep inside dict indexing.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError

from sciforge.llm.client import (
    NO_USAGE,
    ModelIncompleteError,
    ModelRefusalError,
    ModelResponseParseError,
    ModelSchemaError,
    ModelUsage,
)

USD_TICKS_EXPONENT = 10  # 10**10 ticks per USD (docs: TICKS_IN_USD_CENT = 100_000_000)
NANO_USD_EXPONENT = 9    # 10**9 nano-USD per USD
_MAX_ERROR_SNIPPET = 200
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL | re.IGNORECASE)

M = TypeVar("M", bound=BaseModel)


class StrictModel(BaseModel):
    """Base for model-facing output schemas: unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid")


# ------------------------------------------------------------------ usage


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _first_int(obj: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = _int_or_none(obj.get(key))
        if value is not None:
            return value
    return None


def _exact_int(value: Any) -> int | None:
    """Non-negative integer (ints, or floats/strings with an integral value)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, (float, str)):
        try:
            dec = Decimal(str(value).strip())
        except (InvalidOperation, ValueError):
            return None
        if dec.is_finite() and dec >= 0 and dec == dec.to_integral_value():
            return int(dec)
    return None


def parse_reported_cost(usage: dict[str, Any]) -> tuple[Decimal | None, str | None]:
    """Exact provider-reported cost in USD as a Decimal, and its source.

    ``cost_in_usd_ticks`` (preferred): 1 USD = 10**10 ticks (docs:
    ``TICKS_IN_USD_CENT = 100_000_000``). ``cost_in_nano_usd``: 1 USD = 10**9.
    Integer arithmetic via ``Decimal.scaleb`` keeps the value exact.
    """
    ticks = _exact_int(usage.get("cost_in_usd_ticks"))
    if ticks is not None:
        return Decimal(ticks).scaleb(-USD_TICKS_EXPONENT), "reported_ticks"
    nano = _exact_int(usage.get("cost_in_nano_usd"))
    if nano is not None:
        return Decimal(nano).scaleb(-NANO_USD_EXPONENT), "reported_nano"
    return None, None


def parse_usage(payload: Any) -> ModelUsage:
    """Extract :class:`ModelUsage` from a Responses payload (never raises)."""
    if not isinstance(payload, dict):
        return NO_USAGE
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return NO_USAGE
    in_details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
    out_details = usage.get("output_tokens_details") or usage.get("completion_tokens_details") or {}
    in_details = in_details if isinstance(in_details, dict) else {}
    out_details = out_details if isinstance(out_details, dict) else {}

    cost, cost_source = parse_reported_cost(usage)
    return ModelUsage(
        input_tokens=_first_int(usage, "input_tokens", "prompt_tokens"),
        output_tokens=_first_int(usage, "output_tokens", "completion_tokens"),
        reasoning_tokens=_int_or_none(out_details.get("reasoning_tokens")),
        cached_input_tokens=_int_or_none(in_details.get("cached_tokens")),
        total_tokens=_int_or_none(usage.get("total_tokens")),
        cost_usd_reported=cost,
        cost_source=cost_source,
        reported=True,
    )


# ------------------------------------------------------------------ text


def _snippet(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text[:_MAX_ERROR_SNIPPET]


def extract_output_text(payload: Any) -> str:
    """Concatenate all ``output_text`` parts of assistant ``message`` items.

    Raises :class:`ModelRefusalError`, :class:`ModelIncompleteError` or
    :class:`ModelResponseParseError` (all carrying the parsed usage).
    """
    if not isinstance(payload, dict):
        raise ModelResponseParseError("response payload is not a JSON object", kind="malformed_payload")
    usage = parse_usage(payload)

    error = payload.get("error")
    if error:
        message = error.get("message") if isinstance(error, dict) else error
        raise ModelResponseParseError(f"provider returned an error object: {_snippet(message)}",
                                      kind="provider_error", usage=usage)

    status = payload.get("status")
    output = payload.get("output")
    texts: list[str] = []
    refusals: list[str] = []
    if output is not None and not isinstance(output, list):
        raise ModelResponseParseError("'output' is not a list", kind="malformed_payload", usage=usage)
    for item in output or []:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type not in (None, "message"):
            continue  # reasoning / tool items are ignored
        if item.get("role") not in (None, "assistant"):
            continue
        content = item.get("content")
        if isinstance(content, str):  # tolerate a bare string content
            texts.append(content)
            continue
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == "output_text" and isinstance(part.get("text"), str):
                texts.append(part["text"])
            elif part_type == "refusal":
                refusal = part.get("refusal", part.get("text", ""))
                refusals.append(refusal if isinstance(refusal, str) else "")

    if status == "incomplete":
        details = payload.get("incomplete_details")
        reason = details.get("reason") if isinstance(details, dict) else None
        reason = reason if isinstance(reason, str) else None
        raise ModelIncompleteError(
            f"response incomplete ({reason or 'unknown reason'})", reason=reason, usage=usage,
            raw_text="".join(texts) or None,
        )
    if refusals:
        raise ModelRefusalError(f"model refused: {_snippet(' '.join(refusals))}", usage=usage)
    if not texts:
        detail = f" (status={status!r})" if status not in (None, "completed") else ""
        raise ModelResponseParseError(f"response contains no output_text{detail}",
                                      kind="missing_output_text", usage=usage)
    if status not in (None, "completed"):
        raise ModelResponseParseError(f"unexpected response status {_snippet(status)!s}",
                                      kind="unexpected_status", usage=usage, raw_text="".join(texts))
    return "".join(texts)


def parse_json_text(text: str, *, usage: ModelUsage | None = None) -> Any:
    """``json.loads`` with one tolerance: a single surrounding code fence."""
    candidate = text.strip()
    match = _FENCE_RE.match(candidate)
    if match:
        candidate = match.group(1).strip()
    try:
        return json.loads(candidate)
    except (ValueError, TypeError) as exc:
        raise ModelResponseParseError(f"model output is not valid JSON ({exc.__class__.__name__})",
                                      kind="invalid_output_json", usage=usage, raw_text=text) from None


def summarize_validation_error(exc: ValidationError, limit: int = 8) -> str:
    """Compact error summary with locations and messages only (never input values,
    which could contain source text)."""
    parts = []
    for err in exc.errors()[:limit]:
        loc = ".".join(str(x) for x in err.get("loc", ())) or "<root>"
        parts.append(f"{loc}: {err.get('msg', 'invalid')}")
    more = len(exc.errors()) - limit
    if more > 0:
        parts.append(f"... and {more} more")
    return "; ".join(parts)


def validate_structured(data: Any, model_cls: type[M], *, raw_text: str | None = None,
                        usage: ModelUsage | None = None) -> M:
    """Validate parsed JSON (or a JSON string) against ``model_cls``."""
    if isinstance(data, str):
        data = parse_json_text(data, usage=usage)
    try:
        return model_cls.model_validate(data)
    except ValidationError as exc:
        raise ModelSchemaError(f"output failed schema validation: {summarize_validation_error(exc)}",
                               usage=usage, raw_text=raw_text) from None


def strict_json_schema(model_cls: type[BaseModel]) -> dict[str, Any]:
    """JSON Schema for ``model_cls`` in the strict structured-output subset.

    Every object gets ``additionalProperties: false`` and lists all of its
    properties in ``required`` (optional fields stay nullable through their
    type); ``default`` and ``title`` keywords are dropped.
    """
    schema = model_cls.model_json_schema()
    name_maps = ("properties", "$defs", "definitions", "patternProperties")

    def fix(node: Any) -> Any:
        if isinstance(node, list):
            return [fix(x) for x in node]
        if not isinstance(node, dict):
            return node
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key in ("default", "title"):
                continue
            if key in name_maps and isinstance(value, dict):
                out[key] = {name: fix(sub) for name, sub in value.items()}  # keys are field names
            else:
                out[key] = fix(value)
        if out.get("type") == "object" or "properties" in out:
            out["additionalProperties"] = False
            out["required"] = list(out.get("properties", {}).keys())
        return out

    return fix(schema)
