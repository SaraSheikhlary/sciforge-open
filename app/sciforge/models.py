"""Pydantic data models shared across SciForge.

Missing bibliographic fields are always ``None`` (or an empty list for
authors); nothing is inferred or filled in.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from sciforge.normalize import first_author_surname, normalize_doi, normalize_pmid, normalize_title

SourceDatabase = Literal["pubmed", "crossref"]
VerificationStatus = Literal["verified", "partially_verified", "not_verified"]
LookupOutcome = Literal["resolved", "not_found", "lookup_failed"]
ComparisonStatus = Literal["match", "mismatch", "not_compared"]


def make_record_id(
    source_database: str,
    doi: str | None,
    pmid: str | None,
    title: str | None,
    year: int | None,
    authors: list[str] | None = None,
    source_url: str | None = None,
) -> str:
    """Deterministic record ID derived from the source and its identifiers.

    Uses DOI, else PMID, else normalized title + year + first-author surname +
    source URL. The same input always gives the same ID across runs.
    """
    if doi:
        key = f"{source_database}|doi:{doi}"
    elif pmid:
        key = f"{source_database}|pmid:{pmid}"
    else:
        key = (
            f"{source_database}|title:{normalize_title(title) or ''}|year:{year}"
            f"|fa:{first_author_surname(authors) or ''}|url:{source_url or ''}"
        )
    return "rec_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


class Provenance(BaseModel):
    """Where (part of) a record came from."""

    source_database: SourceDatabase
    source_url: str | None = None
    source_record_id: str
    retrieval_timestamp: str
    matched_on: str | None = Field(
        default=None,
        description="None for the record that seeded the merge; otherwise the dedup rule that matched.",
    )


class Record(BaseModel):
    """A normalized bibliographic record."""

    record_id: str = ""
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    pmid: str | None = None
    journal: str | None = None
    source_database: SourceDatabase
    source_url: str | None = None
    retrieval_timestamp: str
    provenance: list[Provenance] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)

    @field_validator("doi", mode="before")
    @classmethod
    def _normalize_doi(cls, value: Any) -> str | None:
        return normalize_doi(value) if value is not None else None

    @field_validator("pmid", mode="before")
    @classmethod
    def _normalize_pmid(cls, value: Any) -> str | None:
        return normalize_pmid(value) if value is not None else None

    @field_validator("year", mode="before")
    @classmethod
    def _check_year(cls, value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("year must be an int or None")
        return value

    @field_validator("title", "journal", "source_url", mode="before")
    @classmethod
    def _blank_to_none(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return value

    @model_validator(mode="after")
    def _fill_derived(self) -> Record:
        if not self.record_id:
            self.record_id = make_record_id(
                self.source_database, self.doi, self.pmid, self.title, self.year, self.authors, self.source_url
            )
        if not self.provenance:
            self.provenance = [
                Provenance(
                    source_database=self.source_database,
                    source_url=self.source_url,
                    source_record_id=self.record_id,
                    retrieval_timestamp=self.retrieval_timestamp,
                )
            ]
        return self


class ErrorEntry(BaseModel):
    """A failure recorded during a run (never raised to the user)."""

    database: str
    stage: str
    query: str | None = None
    timestamp: str
    error_type: str
    http_status: int | None = None
    message: str
    url: str | None = None


class RequestLogEntry(BaseModel):
    """One HTTP request (including its retries) made during a run."""

    database: str
    stage: str
    query: str | None = None
    url: str
    params: dict[str, Any] = Field(default_factory=dict)
    timestamp: str
    status: Literal["ok", "not_found", "error"] = "ok"
    http_status: int | None = None
    attempts: int = 1
    result_count: int | None = None
    error_type: str | None = None
    message: str | None = None


class FieldComparison(BaseModel):
    """Comparison of one field between a record and the authoritative lookup."""

    field: str
    status: ComparisonStatus
    record_value: Any = None
    reference_value: Any = None
    similarity: float | None = None
    affects_status: bool = True
    note: str | None = None


class IdentifierCheck(BaseModel):
    """The result of resolving one identifier (DOI or PMID)."""

    identifier_type: Literal["doi", "pmid"]
    identifier: str
    database: SourceDatabase
    outcome: LookupOutcome
    status: VerificationStatus
    http_status: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    reasons: list[str] = Field(default_factory=list)
    comparisons: list[FieldComparison] = Field(default_factory=list)


class VerificationResult(BaseModel):
    """Overall verification outcome for one deduplicated record."""

    record_id: str
    status: VerificationStatus
    reasons: list[str] = Field(default_factory=list)
    checks: list[IdentifierCheck] = Field(default_factory=list)
    verified_at: str


@dataclass
class LookupResult:
    """Internal result of resolving one identifier against a database."""

    database: SourceDatabase
    identifier_type: Literal["doi", "pmid"]
    identifier: str
    outcome: LookupOutcome
    http_status: int | None = None
    record: Record | None = None
    error_type: str | None = None
    error_message: str | None = None


@dataclass
class SearchOutcome:
    """Internal result of searching one database."""

    database: SourceDatabase
    status: Literal["ok", "partial", "failed"]
    records: list[Record] = field(default_factory=list)
    total_hits: int | None = None
