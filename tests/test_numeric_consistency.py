"""v0.3 M2 deterministic numeric consistency (claim/finding vs quote; methods vs source text)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from m2_support import item, record
from sciforge.stages.numbers import check_numbers, extract_numbers
from sciforge.stages.validation import validate_item

REC = record()


def validate(quote, *, claim="Shear activates platelets.", finding=None, methods=None, source=None):
    text = source if source is not None else f"Intro sentence. {quote} Closing sentence."
    raw = item(REC.record_id, quote, claim=claim, finding=finding, methods=methods)
    validated, reasons, _ = validate_item(raw, request_texts={REC.record_id: text}, supplied_ids={REC.record_id})
    return validated, reasons


def codes(reasons):
    return [r["code"] for r in reasons]


def nums(text):
    return [(format(n.value.normalize(), "f"), n.unit) for n in extract_numbers(text)]


# ------------------------------------------------------------------ extraction rules

@pytest.mark.parametrize("text,expected", [
    ("increased by 40%", [("40", "%")]),
    ("40 percent", [("40", "%")]),
    ("10 µM", [("10", "µM")]), ("10 uM", [("10", "µM")]), ("10μM", [("10", "µM")]),
    ("50 dyn/cm2 for 10 minutes", [("50", "dyn/cm2"), ("10", "min")]),
    ("50 dyn/cm²", [("50", "dyn/cm2")]),
    ("0.25 and 0·5 mg/kg", [("0.25", None), ("0.5", "mg/kg")]),
    ("fell by −3.5 °C and -2", [("-3.5", "°C"), ("-2", None)]),
    ("1e-3 M", [("0.001", None)]),
    ("1.5×10^6 cells", [("1500000", "cells")]), ("1.5 x 10-6", [("0.0000015", None)]),
    ("1.5×10⁶", [("1500000", None)]), ("10^6", [("1000000", None)]),
    ("10–20 µM", [("10", "µM"), ("20", "µM")]), ("10-20%", [("10", "%"), ("20", "%")]),
    ("10 to 20 mg", [("10", "mg"), ("20", "mg")]),
    ("1,000 patients and 12,345,678 cells", [("1000", None), ("12345678", "cells")]),
    ("IL-6, CD62, H2O2, TNF-α and P-selectin", []),
    ("3D culture of 5HT neurons", []),
    ("type 2 diabetes at day 3", [("2", None), ("3", None)]),
    ("see doi:10.1234/abc.99 and PMID: 123456", []),
    ("", []),
])
def test_extraction(text, expected):
    assert nums(text) == expected


def test_values_are_decimals():
    assert extract_numbers("0.1")[0].value == Decimal("0.1")


# ------------------------------------------------------------------ claim / finding vs quote

def test_valid_matching_percentage():
    v, r = validate("P-selectin rose by 40% compared with controls.", claim="P-selectin rose by 40%.")
    assert v is not None and r == []


def test_changed_percentage_rejected_with_value_detail():
    v, r = validate("P-selectin rose by 40% compared with controls.", claim="P-selectin rose by 45%.")
    assert v is None and codes(r) == ["number_not_in_quote"]
    assert r[0]["field"] == "claim" and r[0]["value"] == "45" and r[0]["unit"] == "%"


def test_matching_concentration_with_unit():
    v, r = validate("Cells treated with 10 µM drug X died.", claim="10 uM drug X killed cells.")
    assert v is not None and r == []


def test_changed_concentration_rejected():
    _, r = validate("Cells treated with 10 µM drug X died.", claim="100 µM drug X killed cells.")
    assert codes(r) == ["number_not_in_quote"] and r[0]["value"] == "100"


def test_unit_mismatch():
    _, r = validate("Cells treated with 10 µM drug X died.", claim="10 mM drug X killed cells.")
    assert codes(r) == ["unit_mismatch"]
    assert r[0]["unit"] == "mM" and r[0]["quote_units"] == ["µM"]


def test_unitless_claim_value_matches_value_with_unit():
    v, _ = validate("Cells treated with 10 µM drug X died.", claim="Drug X at 10 killed cells.")
    assert v is not None


def test_multiple_values_all_checked():
    quote = "Of 120 patients, 30% responded and 12 had adverse events."
    assert validate(quote, claim="Of 120 patients, 30% responded; 12 had adverse events.")[0] is not None
    _, r = validate(quote, claim="Of 120 patients, 35% responded; 14 had adverse events.")
    assert [(x["code"], x["value"]) for x in r] == [("number_not_in_quote", "35"), ("number_not_in_quote", "14")]


def test_decimal_values():
    quote = "The mean ratio was 0.85 (SD 0.12)."
    assert validate(quote, claim="The mean ratio was 0.850.")[0] is not None  # numerically equal
    assert codes(validate(quote, claim="The mean ratio was 0.58.")[1]) == ["number_not_in_quote"]


def test_negative_values_and_unicode_minus():
    quote = "Temperature changed by −2.5 °C after exposure."
    assert validate(quote, claim="Temperature changed by -2.5 °C.")[0] is not None
    assert codes(validate(quote, claim="Temperature changed by 2.5 °C.")[1]) == ["number_not_in_quote"]


def test_scientific_notation():
    quote = "Counts reached 1.5×10^6 cells/µL at peak."
    assert validate(quote, claim="Counts reached 1.5e6 cells/µL.")[0] is not None
    assert validate(quote, claim="Counts reached 1.5 x 10^6 cells/µL.")[0] is not None
    assert codes(validate(quote, claim="Counts reached 1.5×10^7 cells/µL.")[1]) == ["number_not_in_quote"]


def test_thousands_separator():
    quote = "A cohort of 12,000 adults was followed for 5 years."
    assert validate(quote, claim="A cohort of 12000 adults was followed.")[0] is not None
    assert codes(validate(quote, claim="A cohort of 1,200 adults was followed.")[1]) == ["number_not_in_quote"]


def test_embedded_identifiers_ignored():
    quote = "Shear increased P-selectin and IL-6 expression in platelets."
    v, r = validate(quote, claim="CD62P and IL-6 rose with shear; TNF-α unchanged.")
    assert v is not None and r == []


def test_claim_without_numbers():
    assert validate("Shear increased P-selectin by 40% in 12 donors.", claim="Shear increases P-selectin.")[0] is not None


def test_quote_numbers_not_repeated_in_claim_are_fine():
    v, _ = validate("In 12 donors at 50 dyn/cm2, P-selectin rose by 40%.", claim="P-selectin rose by 40%.")
    assert v is not None


def test_finding_numbers_checked_against_quote():
    quote = "P-selectin rose by 40% compared with controls."
    assert validate(quote, finding="Rose by 40%.")[0] is not None
    _, r = validate(quote, finding="Rose by 4%.")
    assert codes(r) == ["number_not_in_quote"] and r[0]["field"] == "finding"


def test_methods_numbers_checked_against_source_text_not_quote():
    source = "Platelets were sheared at 50 dyn/cm2 for 10 minutes. P-selectin rose by 40% compared with controls."
    quote = "P-selectin rose by 40% compared with controls."
    v, _ = validate(quote, methods="Shear of 50 dyn/cm2 for 10 min.", source=source)
    assert v is not None  # numbers are in another sentence of the source → fine
    _, r = validate(quote, methods="Shear of 60 dyn/cm2 for 10 min.", source=source)
    assert codes(r) == ["number_not_in_source"] and r[0]["field"] == "methods" and r[0]["value"] == "60"
    _, r = validate(quote, methods="Shear of 50 Pa for 10 min.", source=source)
    assert codes(r) == ["unit_mismatch"] and r[0]["source_text_units"] == ["dyn/cm2"]


def test_range_in_quote_supports_endpoint_with_unit():
    assert validate("Doses of 10–20 µM were tested and all were active.", claim="20 µM was active.")[0] is not None


def test_numbers_not_checked_when_quote_invalid():
    _, r = validate("Not in the source text at all, clearly.", claim="Rose by 99%.",
                    source="Something else entirely here.")
    assert codes(r) == ["quote_not_in_source"]


def test_check_numbers_scope_codes():
    assert check_numbers("claim", "5 mg", "6 mg", scope="quote")[0]["code"] == "number_not_in_quote"
    assert check_numbers("methods", "5 mg", "6 mg", scope="source_text")[0]["code"] == "number_not_in_source"
