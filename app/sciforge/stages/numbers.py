"""Deterministic numeric-consistency check for evidence items.

Rules (conservative; false rejections preferred over false acceptances):

* **What is checked.** Every numeric value in ``claim`` and ``finding`` must
  appear in the item's exact ``quote`` (reason ``number_not_in_quote``).
  Numbers in ``methods`` must appear somewhere in the supplied (capped) source
  text (reason ``number_not_in_source``), because methods are often described
  in a different sentence than the quoted result. ``limitations`` and
  ``relevance`` are not checked. Numbers that only occur in the quote are fine.
* **Units.** When a value in the checked field carries a unit, the reference
  must contain the same value with the same (canonical) unit, else
  ``unit_mismatch``. A unitless value in the field matches the value with any
  unit. Canonicalisation only merges spelling variants: ``µ``/``μ``/``u``
  prefixes (``µM``, ``μM``, ``uM``), ``%``/``percent``/``per cent``, ``mL``/``ml``,
  ``h``/``hr``/``hours``, ``min``/``minutes``, ``s``/``sec``, ``d``/``days``,
  ``cm2``/``cm²``/``cm^2``, ``°C``/``℃``. SI prefixes stay case-sensitive
  (``mM`` ≠ ``MM``); no unit conversion (``0.4`` ≠ ``40%``; ``1 mM`` ≠ ``1000 µM``).
* **Values.** Compared numerically (``Decimal``) after normalisation: thousands
  separators (``1,000``), decimals (``0.5``, ``0·5``), signs (``-``, ``+`` and the
  Unicode minus ``−``), scientific notation (``1e-3``, ``1.5×10^6``,
  ``1.5 x 10-6``, ``1.5×10⁶``, bare ``10^6``). ``-5`` and ``5`` differ (a claim
  that turns "decreased by 5%" into "-5%" is rejected). Ranges (``10–20``,
  ``10-20``, ``10 to 20``) yield both endpoints; a unit written once after the
  range applies to both (``10–20 µM``).
* **Not counted (non-factual numerals).** Digits embedded in alphanumeric
  tokens: a letter directly before the digits (``CD62``, ``H2O``), a
  letter-hyphen prefix (``IL-6``, ``P-selectin``-style ``TNF-α``), or letters
  directly after that are not a known unit (``3D``, ``5HT``). Identifier-like
  strings (DOI/PMID/PMCID/URL/arXiv) are removed before extraction (they are
  flagged separately). Standalone numerals ARE counted (``type 2 diabetes``,
  ``day 3``) — they must then appear in the quote too.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from sciforge.stages.identifiers import strip_identifiers

_SUPERSCRIPTS = "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺"
_SUPER_TABLE = str.maketrans(_SUPERSCRIPTS, "0123456789-+")

# (pattern, canonical). Case-sensitive unless marked with (?i:...).
_UNIT_VARIANTS: list[tuple[str, str]] = [
    (r"%", "%"), (r"(?i:per\s?cent)", "%"),
    (r"[µμu]M", "µM"), (r"nM", "nM"), (r"pM", "pM"), (r"mM", "mM"), (r"fM", "fM"),
    (r"[µμu]mol", "µmol"), (r"mmol", "mmol"), (r"nmol", "nmol"), (r"mol", "mol"),
    (r"pg", "pg"), (r"ng", "ng"), (r"(?:[µμu]g|mcg)", "µg"), (r"mg", "mg"), (r"kg", "kg"), (r"g", "g"),
    (r"[µμu][lL]", "µL"), (r"m[lL]", "mL"), (r"d[lL]", "dL"), (r"L", "L"),
    (r"nm", "nm"), (r"[µμu]m", "µm"), (r"mm", "mm"), (r"cm(?:\^?2)", "cm2"), (r"cm", "cm"), (r"km", "km"),
    (r"mmHg", "mmHg"), (r"kPa", "kPa"), (r"mPa", "mPa"), (r"Pa", "Pa"), (r"dyn", "dyn"),
    (r"(?:°C|℃)", "°C"), (r"kHz", "kHz"), (r"Hz", "Hz"), (r"kDa", "kDa"), (r"Da", "Da"),
    (r"(?i:hours?|hrs?)", "h"), (r"h", "h"), (r"(?i:minutes?|mins?)", "min"), (r"min", "min"),
    (r"(?i:seconds?|secs?)", "s"), (r"ms", "ms"), (r"s", "s"), (r"(?i:days?)", "d"), (r"d", "d"),
    (r"(?i:weeks?|wks?)", "wk"), (r"(?i:months?)", "mo"), (r"(?i:years?|yrs?)", "y"),
    (r"(?i:fold)", "fold"), (r"bpm", "bpm"), (r"rpm", "rpm"), (r"IU", "IU"), (r"U", "U"), (r"(?i:cells)", "cells"),
]
_UNIT_ALT = "|".join(f"(?:{p})" for p, _ in _UNIT_VARIANTS)
_UNIT_TOKEN = re.compile(rf"(?:{_UNIT_ALT})(?![A-Za-z0-9])")
_UNIT_RE = re.compile(rf"(?P<u1>{_UNIT_ALT})(?:/(?P<u2>{_UNIT_ALT}))?(?![A-Za-z0-9])")

_NUMBER_RE = re.compile(
    r"""
    (?<![A-Za-z0-9_.,^/])(?<![A-Za-z]-)          # not embedded in a token (CD62, IL-6, 10^6's exponent)
    (?P<sign>[-+])?
    (?P<int>\d{1,3}(?:,\d{3})+(?![\d])|\d+)
    (?P<frac>\.\d+)?
    (?:
        (?P<e>[eE][-+]?\d+)
      | \s*[×xX*]\s*10\s*(?:\^|\*\*)?\s*(?P<x10>[-+]?\d+)
      | (?P<pow>\^[-+]?\d+)
    )?
    """,
    re.VERBOSE,
)
_RANGE_SEP = re.compile(r"^\s*(?:-|–|—|to)\s*$")


@dataclass(frozen=True)
class NumberMention:
    value: Decimal
    unit: str | None
    text: str
    start: int
    end: int

    @property
    def display(self) -> str:
        return format(self.value.normalize(), "f")


def _canonical_unit(token: str) -> str:
    for pattern, canonical in _UNIT_VARIANTS:
        if re.fullmatch(pattern, token):
            return canonical
    return token


def _prepare(text: str) -> str:
    text = re.sub(f"[{_SUPERSCRIPTS}]+", lambda m: "^" + m.group(0).translate(_SUPER_TABLE), text)
    text = text.replace("\u2212", "-").replace("\u00a0", " ").replace("\u202f", " ").replace("\u2009", " ")
    text = re.sub(r"(?<=\d)[·⋅](?=\d)", ".", text)
    return strip_identifiers(text)


def extract_numbers(text: str | None) -> list[NumberMention]:
    """All counted numeric values (with canonical units) in ``text``."""
    if not text:
        return []
    prepared = _prepare(text)
    out: list[NumberMention] = []
    for m in _NUMBER_RE.finditer(prepared):
        end = m.end()
        unit: str | None = None
        rest = prepared[end:]
        um = _UNIT_RE.match(rest) or _UNIT_RE.match(rest.lstrip(" ")) if rest else None
        if um is not None:
            gap = len(rest) - len(rest.lstrip(" ")) if not _UNIT_RE.match(rest) else 0
            if gap <= 1:
                u1 = _canonical_unit(um.group("u1"))
                unit = u1 + (f"/{_canonical_unit(um.group('u2'))}" if um.group("u2") else "")
                end += gap + um.end()
            else:
                um = None
        if um is None and rest[:1].isalpha():
            continue  # digits glued to a non-unit word (3D, 5HT)
        try:
            value = Decimal(m.group("int").replace(",", "") + (m.group("frac") or ""))
            if m.group("e"):
                value = value.scaleb(int(m.group("e")[1:]))
            elif m.group("x10"):
                value = value.scaleb(int(m.group("x10")))
            elif m.group("pow"):
                value = value ** int(m.group("pow")[1:])
        except (InvalidOperation, ValueError, OverflowError):
            continue
        if m.group("sign") == "-":
            value = -value
        out.append(NumberMention(value, unit, prepared[m.start():end], m.start(), end))

    # ranges: "10-20", "10 - 20", "10–20 µM", "10 to 20%"
    for i in range(len(out) - 1):
        a, b = out[i], out[i + 1]
        between = prepared[a.end:b.start]
        if b.text.startswith("-") and between.strip() == "" and a.unit is None:
            b = NumberMention(-b.value, b.unit, b.text[1:], b.start + 1, b.end)  # "10 -20" → range, not negative
            out[i + 1] = b
            between = "-"
        if _RANGE_SEP.match(between) and a.unit is None and b.unit is not None:
            out[i] = NumberMention(a.value, b.unit, a.text, a.start, a.end)
    return out


def check_numbers(field: str, text: str | None, reference: str, *, scope: str) -> list[dict[str, Any]]:
    """Reasons for numbers in ``text`` that are missing from ``reference``.

    ``scope`` is ``"quote"`` or ``"source_text"`` and selects the reason code.
    Details carry only the field name, the normalised value and units.
    """
    ref: dict[Decimal, set[str | None]] = {}
    for n in extract_numbers(reference):
        ref.setdefault(n.value, set()).add(n.unit)
    missing_code = "number_not_in_quote" if scope == "quote" else "number_not_in_source"
    reasons: list[dict[str, Any]] = []
    for n in extract_numbers(text):
        units = ref.get(n.value)
        if units is None:
            reasons.append({"code": missing_code, "detail": f"{field} value not found in the {scope}",
                            "field": field, "value": n.display, "unit": n.unit})
        elif n.unit is not None and n.unit not in units:
            reasons.append({"code": "unit_mismatch", "detail": f"{field} value found in the {scope} with a different unit",
                            "field": field, "value": n.display, "unit": n.unit,
                            f"{scope}_units": sorted(u or "none" for u in units)})
    return reasons
