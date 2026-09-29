"""v0.3 model layer (optional; not used by the v0.2 pipeline).

"V0.2 remains the authority for source identity. V0.3 can interpret verified
sources, but it cannot create citations."

Importing this package performs no I/O and reads no environment variables.
``XAIClient`` is loaded lazily to keep ``sciforge.config`` free of import cycles.
"""

from sciforge.llm.audit import ModelCallAudit
from sciforge.llm.budget import BudgetLimits, BudgetTracker, PriceTable, Reservation, RetryPolicy, budgeted_call
from sciforge.llm.client import (
    BudgetExhausted,
    ModelAuthError,
    ModelClient,
    ModelConfigError,
    ModelConnectionError,
    ModelError,
    ModelHTTPError,
    ModelIncompleteError,
    ModelMessage,
    ModelRateLimited,
    ModelRefusalError,
    ModelRequest,
    ModelResponse,
    ModelResponseParseError,
    ModelSchemaError,
    ModelTimeout,
    ModelUsage,
)
from sciforge.llm.fake import FakeModelClient
from sciforge.llm.parsing import StrictModel, strict_json_schema, validate_structured

__all__ = [
    "BudgetExhausted", "BudgetLimits", "BudgetTracker", "FakeModelClient", "ModelAuthError", "ModelCallAudit",
    "ModelClient", "ModelConfigError", "ModelConnectionError", "ModelError", "ModelHTTPError",
    "ModelIncompleteError", "ModelMessage", "ModelRateLimited", "ModelRefusalError", "ModelRequest",
    "ModelResponse", "ModelResponseParseError", "ModelSchemaError", "ModelTimeout", "ModelUsage", "PriceTable",
    "Reservation", "RetryPolicy", "StrictModel", "XAIClient", "budgeted_call", "strict_json_schema", "validate_structured",
]


def __getattr__(name: str):
    if name == "XAIClient":
        from sciforge.llm.xai import XAIClient

        return XAIClient
    raise AttributeError(f"module 'sciforge.llm' has no attribute {name!r}")
