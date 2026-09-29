"""Identifier-like patterns (DOI / PMID / PMCID / URL / arXiv) used for flagging and redaction."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

IDENTIFIER_PATTERNS: dict[str, re.Pattern[str]] = {
    "doi": re.compile(r"\b(?:doi:\s*)?10\.\d{4,9}/[^\s\"'<>]+", re.IGNORECASE),
    "pmid": re.compile(r"\bPMID:?\s*\d{1,9}\b", re.IGNORECASE),
    "pmcid": re.compile(r"\bPMC\d{3,9}\b"),
    "url": re.compile(r"\bhttps?://\S+|\bwww\.\S+", re.IGNORECASE),
    "arxiv": re.compile(r"\barXiv:\s*\d{4}\.\d{4,5}(?:v\d+)?\b|\barxiv\.org/\S+", re.IGNORECASE),
}


def identifier_warnings(fields: Mapping[str, Any]) -> list[dict[str, str]]:
    """``[{"field", "type", "match"}]`` for identifier-like strings in free text."""
    out: list[dict[str, str]] = []
    for name, value in fields.items():
        if not isinstance(value, str):
            continue
        for kind, pattern in IDENTIFIER_PATTERNS.items():
            for m in pattern.finditer(value):
                out.append({"field": name, "type": kind, "match": m.group(0)[:100]})
    return out


def redact_identifiers(text: str, replacement: str = "[redacted:identifier]") -> str:
    """Replace every identifier-like substring with ``replacement``."""
    for pattern in IDENTIFIER_PATTERNS.values():
        text = pattern.sub(replacement, text)
    return text


def strip_identifiers(text: str) -> str:
    """Replace identifier-like substrings with a space (used before number extraction)."""
    return redact_identifiers(text, " ")
