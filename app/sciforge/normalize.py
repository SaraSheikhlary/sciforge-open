"""Pure normalization helpers for identifiers, titles, and author names.

All functions return ``None`` instead of guessing when the input cannot be
normalized confidently.
"""

from __future__ import annotations

import html
import re
import unicodedata
from collections.abc import Sequence
from typing import Any
from urllib.parse import unquote

# Prefixes stripped from DOIs (repeatedly, so "doi: https://doi.org/10..." works).
_DOI_PREFIX_RE = re.compile(
    r"^(?:(?:https?://)?(?:www\.|dx\.)?doi\.org/|doi\s*:\s*|doi\s+)",
    re.IGNORECASE,
)
# "10." + numeric registrant code (optionally dotted) + "/" + non-empty suffix.
_DOI_VALID_RE = re.compile(r"^10\.\d+(?:\.\d+)*/\S+$")
_DOI_TRAILING = ".,;:'\"" + " \t\r\n"

_PMID_PREFIX_RE = re.compile(r"^pmid\s*[:#]?\s*", re.IGNORECASE)
_PMID_VALID_RE = re.compile(r"^\d{1,9}$")

_TAG_RE = re.compile(r"<[^>]+>")
_NON_WORD_RE = re.compile(r"[\W_]+", re.UNICODE)
_INITIALS_RE = re.compile(r"^[A-Z]{1,4}$")


def normalize_doi(value: Any) -> str | None:
    """Normalize a DOI string, or return None if it is not a valid DOI.

    Steps: URL-decode, strip whitespace, strip resolver / ``doi:`` prefixes,
    strip trailing punctuation, lowercase, then require ``10.<digits>/<suffix>``
    with no internal whitespace.
    """
    if not isinstance(value, str):
        return None
    text = unquote(value).strip()
    previous = None
    while previous != text:
        previous = text
        text = _DOI_PREFIX_RE.sub("", text).strip()
    text = text.rstrip(_DOI_TRAILING).lower()
    if not _DOI_VALID_RE.match(text):
        return None
    return text


def normalize_pmid(value: Any) -> str | None:
    """Normalize a PubMed ID to a canonical digit string, or return None.

    Accepts positive ints and strings such as ``"123"``, ``" 123 "``,
    ``"PMID: 123"``. Leading zeros are removed. Non-numeric input, zero,
    negative numbers, booleans, and values longer than 9 digits are rejected.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value) if 0 < value < 10**9 else None
    if not isinstance(value, str):
        return None
    text = _PMID_PREFIX_RE.sub("", value.strip()).strip()
    if not _PMID_VALID_RE.match(text):
        return None
    number = int(text)
    return str(number) if number > 0 else None


def _fold(text: str) -> str:
    """Unescape HTML, drop tags, NFKD-decompose, remove accents, casefold."""
    text = _TAG_RE.sub(" ", html.unescape(text))
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return stripped.casefold()


def normalize_title(value: Any) -> str | None:
    """Normalize a title for comparison: casefold, strip punctuation and markup,
    remove accents, collapse whitespace. Returns None for empty input."""
    if not isinstance(value, str):
        return None
    text = _NON_WORD_RE.sub(" ", _fold(value)).strip()
    return text or None


def normalize_text_field(value: Any) -> str | None:
    """Same as :func:`normalize_title`; used for journal names."""
    return normalize_title(value)


def normalize_surname(value: Any) -> str | None:
    """Normalize a surname: folded, with all non-alphanumerics removed."""
    if not isinstance(value, str):
        return None
    text = _NON_WORD_RE.sub("", _fold(value))
    return text or None


def extract_surname(name: Any) -> str | None:
    """Extract and normalize the surname from a display name.

    Supported formats:
    * ``"Family, Given"`` (Crossref-derived names in SciForge) -> ``Family``
    * ``"Family AB"`` (PubMed esummary style, trailing initials) -> ``Family``
    * ``"Given Family"`` -> last token
    """
    if not isinstance(name, str) or not name.strip():
        return None
    name = name.strip()
    if "," in name:
        return normalize_surname(name.split(",", 1)[0])
    tokens = name.split()
    if len(tokens) > 1 and _INITIALS_RE.match(tokens[-1]):
        return normalize_surname(" ".join(tokens[:-1]))
    return normalize_surname(tokens[-1])


def first_author_surname(authors: Sequence[str] | None) -> str | None:
    """Normalized surname of the first author, or None if unavailable."""
    if not authors:
        return None
    return extract_surname(authors[0])
