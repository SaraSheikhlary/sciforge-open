"""v0.3 model input boundary: the ONLY place where source data becomes model-visible.

"V0.2 remains the authority for source identity. V0.3 can interpret verified
sources, but it cannot create citations."

The transformation is an explicit **allowlist**: a :class:`ModelSource` has
exactly three fields — ``record_id`` (opaque v0.2 id, ``rec_`` + 16 hex),
``access_level`` (``pubmed_abstract`` / ``crossref_abstract``) and
``source_text`` (the capped abstract). It is built field by field from a
:class:`~sciforge.sourcetext.SourceText`; nothing is copied generically, so a
new field on a record or source text can never leak into a prompt.

Abstract text is passed **unaltered**: quotes must be exact substrings of what
the model saw, so URLs, trial registry numbers or identifiers that an abstract
itself contains are not redacted. The guarantee enforced (and tested) is that
record-level bibliographic fields (title, authors, journal, publication date,
DOI, PMID, links) never enter the request. Serialisation is deterministic
(sorted keys, compact separators, UTF-8).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any, Literal

from pydantic import Field

from sciforge.llm.parsing import StrictModel
from sciforge.sourcetext import SourceText

MODEL_SOURCE_FIELDS = ("record_id", "access_level", "source_text")
RECORD_ID_PATTERN = r"^rec_[0-9a-f]{16}$"


class ModelSource(StrictModel):
    """Model-visible view of one source (allowlist; extra fields forbidden)."""

    record_id: str = Field(pattern=RECORD_ID_PATTERN)
    access_level: Literal["pubmed_abstract", "crossref_abstract"]
    source_text: str = Field(min_length=1)


def to_model_source(source: SourceText) -> ModelSource:
    """Build the model-visible object for a usable source text (allowlist)."""
    if not source.usable:
        raise ValueError(f"source {source.record_id} has no usable text (status {source.status})")
    return ModelSource(record_id=source.record_id, access_level=source.access_level,  # type: ignore[arg-type]
                       source_text=source.source_text)  # type: ignore[arg-type]


def model_sources(sources: Iterable[SourceText]) -> list[ModelSource]:
    """Model-visible objects for every usable source, in the given order."""
    return [to_model_source(s) for s in sources if s.usable]


def canonical_json(payload: Any) -> str:
    """Deterministic JSON used for every model-visible payload."""
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def source_payload(sources: Iterable[ModelSource]) -> list[dict[str, str]]:
    """Plain dicts restricted to :data:`MODEL_SOURCE_FIELDS` (defence in depth)."""
    out = []
    for s in sources:
        dumped = s.model_dump()
        out.append({k: dumped[k] for k in MODEL_SOURCE_FIELDS})
    return out
