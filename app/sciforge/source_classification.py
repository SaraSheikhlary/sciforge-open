"""Deterministic source-type classification and source policy (no model, no I/O).

Every candidate gets a ``source_status`` built ONLY from bibliographic metadata
returned by the databases that were searched (never from title wording):

* ``"peer-reviewed journal article"`` — metadata says *journal article*:
  Crossref ``type == "journal-article"``, or a PubMed record that has a journal
  AND at least one journal-article publication type (e.g. ``Journal Article``,
  ``Review``, ``Randomized Controlled Trial``) and no ``Preprint`` type.
  **This is a metadata label, not a guarantee of peer review**: bibliographic
  databases record the publication type, not whether (or how well) a paper was
  peer reviewed.
* ``"preprint"`` — Crossref ``type == "posted-content"`` with ``subtype ==
  "preprint"``; PubMed publication type ``Preprint``; or a known preprint
  server identified by DOI pattern / container / publisher / institution
  (bioRxiv / medRxiv ``10.1101/<digits>``, arXiv ``10.48550/arXiv.``, Research
  Square ``10.21203/rs.``, SSRN ``10.2139/ssrn.``, Preprints.org
  ``10.20944/preprints``, ChemRxiv ``10.26434/chemrxiv``). A preprint signal
  from any merged source wins over every other label (preprints are always
  labelled as such).
* ``"conference paper"`` — Crossref ``proceedings-article``; PubMed ``Congress``.
* ``"book/chapter"`` — Crossref book types (``book``, ``book-chapter``,
  ``monograph``, ``edited-book``, ``book-part``, ...).
* ``"unknown"`` — no type metadata, other types (posted-content without the
  preprint subtype, reports, datasets, letters/editorials without a journal
  article type, ...), or conflicting labels between merged sources.

Source policy (``SCIFORGE_SOURCE_POLICY``):

* ``allow_all`` — ranking unchanged; every type may be selected.
* ``peer_reviewed_preferred`` (default) — stable re-ordering of the relevance
  ranking into tiers: text-relevant journal articles, then text-relevant
  others (preprints, conference papers, books, unknown), then journal articles
  without any title/abstract match, then the rest. Within a tier the relevance
  order is kept. Non-journal sources therefore fill the set only when not
  enough relevant journal articles pass verification.
* ``peer_reviewed_only`` — ONLY ``"peer-reviewed journal article"`` candidates
  are sent to verification (and therefore to the model). Preprints,
  conference papers, books/chapters and ``unknown`` are all excluded (fail
  closed). When fewer than the requested number remain, the shortfall is
  reported in the logs, summary, report and web app.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "BOOK", "CONFERENCE", "JOURNAL_ARTICLE", "PREPRINT", "SOURCE_POLICIES", "SOURCE_POLICY_ENV", "SOURCE_STATUSES",
    "UNKNOWN", "PolicyOutcome", "SourceClassification", "apply_source_policy", "classify_metadata",
    "classify_record", "crossref_metadata", "pubmed_metadata",
]

JOURNAL_ARTICLE = "peer-reviewed journal article"
PREPRINT = "preprint"
CONFERENCE = "conference paper"
BOOK = "book/chapter"
UNKNOWN = "unknown"
SOURCE_STATUSES = (JOURNAL_ARTICLE, PREPRINT, CONFERENCE, BOOK, UNKNOWN)
STATUS_NOTE = ("source_status is derived from bibliographic metadata only (Crossref type/subtype, PubMed "
               "publication types and journal, known preprint servers); 'peer-reviewed journal article' means the "
               "metadata describes a journal article — it is not a guarantee of peer review.")

POLICY_ALLOW_ALL = "allow_all"
POLICY_PREFERRED = "peer_reviewed_preferred"
POLICY_ONLY = "peer_reviewed_only"
SOURCE_POLICIES = (POLICY_ALLOW_ALL, POLICY_PREFERRED, POLICY_ONLY)
DEFAULT_SOURCE_POLICY = POLICY_PREFERRED
SOURCE_POLICY_ENV = "SCIFORGE_SOURCE_POLICY"

CROSSREF_JOURNAL_TYPES = frozenset({"journal-article"})
CROSSREF_CONFERENCE_TYPES = frozenset({"proceedings-article"})
CROSSREF_BOOK_TYPES = frozenset({"book", "book-chapter", "book-part", "book-section", "book-track", "book-set",
                                 "book-series", "edited-book", "monograph", "reference-book"})
PUBMED_JOURNAL_PUBTYPES = frozenset({
    "journal article", "review", "systematic review", "meta-analysis", "randomized controlled trial",
    "clinical trial", "clinical trial, phase i", "clinical trial, phase ii", "clinical trial, phase iii",
    "clinical trial, phase iv", "controlled clinical trial", "comparative study", "case reports",
    "observational study", "multicenter study", "evaluation study", "validation study",
})
PUBMED_PREPRINT_PUBTYPES = frozenset({"preprint"})
PUBMED_CONFERENCE_PUBTYPES = frozenset({"congress"})

# Known preprint servers: (label, DOI regex, container/publisher/institution names (lower-case, exact)).
PREPRINT_SERVERS: tuple[tuple[str, re.Pattern[str], frozenset[str]], ...] = (
    ("bioRxiv/medRxiv", re.compile(r"^10\.1101/(?:\d{4}\.\d{2}\.\d{2}\.)?\d{5,}(?:v\d+)?$"),
     frozenset({"biorxiv", "medrxiv", "biorxiv : the preprint server for biology",
                "medrxiv : the preprint server for health sciences"})),
    ("arXiv", re.compile(r"^10\.48550/arxiv\.", re.IGNORECASE), frozenset({"arxiv"})),
    ("Research Square", re.compile(r"^10\.21203/rs\.", re.IGNORECASE), frozenset({"research square", "res sq"})),
    ("SSRN", re.compile(r"^10\.2139/ssrn\.", re.IGNORECASE), frozenset({"ssrn", "ssrn electronic journal"})),
    ("Preprints.org", re.compile(r"^10\.20944/preprints", re.IGNORECASE), frozenset({"preprints.org"})),
    ("ChemRxiv", re.compile(r"^10\.26434/chemrxiv", re.IGNORECASE), frozenset({"chemrxiv"})),
)


def _clean_str(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _str_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [v.strip() for v in value if isinstance(v, str) and v.strip()]
    return []


def crossref_metadata(work: Mapping[str, Any]) -> dict[str, Any]:
    """Type-relevant metadata of one Crossref work (as returned by the API; nothing inferred)."""
    institutions = []
    inst = work.get("institution")
    for item in inst if isinstance(inst, list) else [inst] if isinstance(inst, dict) else []:
        if isinstance(item, dict):
            institutions.extend(_str_list(item.get("name")))
    return {"database": "crossref", "crossref_type": _clean_str(work.get("type")),
            "crossref_subtype": _clean_str(work.get("subtype")), "publisher": _clean_str(work.get("publisher")),
            "container_title": _str_list(work.get("container-title")), "institution": institutions,
            "doi": _clean_str(work.get("DOI"))}


def pubmed_metadata(doc: Mapping[str, Any], doi: str | None = None) -> dict[str, Any]:
    """Type-relevant metadata of one PubMed esummary document (``pubtype`` + journal)."""
    journals = [j for j in (_clean_str(doc.get("fulljournalname")), _clean_str(doc.get("source"))) if j]
    return {"database": "pubmed", "pubmed_publication_types": _str_list(doc.get("pubtype")),
            "journal": journals, "doi": doi}


def _preprint_server(doi: str | None, names: Iterable[str]) -> str | None:
    lowered = {n.lower() for n in names}
    for label, pattern, known in PREPRINT_SERVERS:
        if doi and pattern.match(doi):
            return f"{label} (DOI pattern)"
        if lowered & known:
            return f"{label} (container/publisher/institution name)"
    return None


@dataclass(frozen=True)
class SourceClassification:
    record_id: str
    source_status: str
    basis: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {"record_id": self.record_id, "source_status": self.source_status, "basis": list(self.basis)}


def classify_metadata(meta: Mapping[str, Any], doi: str | None = None) -> tuple[str, str]:
    """(status, basis) for ONE database's metadata of a record."""
    doi = (doi or _clean_str(meta.get("doi")) or "").lower() or None
    if meta.get("database") == "crossref":
        ctype = (meta.get("crossref_type") or "").lower()
        subtype = (meta.get("crossref_subtype") or "").lower()
        names = [*meta.get("container_title", []), *meta.get("institution", []),
                 *([meta["publisher"]] if meta.get("publisher") else [])]
        server = _preprint_server(doi, names)
        if ctype == "posted-content" and subtype == "preprint":
            return PREPRINT, "crossref type posted-content, subtype preprint"
        if server:
            return PREPRINT, f"known preprint server: {server}"
        if ctype in CROSSREF_JOURNAL_TYPES:
            return JOURNAL_ARTICLE, "crossref type journal-article"
        if ctype in CROSSREF_CONFERENCE_TYPES:
            return CONFERENCE, f"crossref type {ctype}"
        if ctype in CROSSREF_BOOK_TYPES:
            return BOOK, f"crossref type {ctype}"
        return UNKNOWN, f"crossref type {ctype or 'missing'}" + (f", subtype {subtype}" if subtype else "")
    if meta.get("database") == "pubmed":
        pubtypes = {p.lower() for p in meta.get("pubmed_publication_types", [])}
        journals = list(meta.get("journal", []))
        server = _preprint_server(doi, journals)
        if pubtypes & PUBMED_PREPRINT_PUBTYPES:
            return PREPRINT, "pubmed publication type Preprint"
        if server:
            return PREPRINT, f"known preprint server: {server}"
        if pubtypes & PUBMED_CONFERENCE_PUBTYPES and not pubtypes & PUBMED_JOURNAL_PUBTYPES:
            return CONFERENCE, "pubmed publication type Congress"
        if journals and pubtypes & PUBMED_JOURNAL_PUBTYPES:
            return JOURNAL_ARTICLE, ("pubmed journal present and publication type(s) "
                                     + ", ".join(sorted(pubtypes & PUBMED_JOURNAL_PUBTYPES)))
        if not pubtypes:
            return UNKNOWN, "pubmed publication types missing"
        return UNKNOWN, "pubmed publication types do not identify a journal article"
    return UNKNOWN, "no type metadata"


def classify_record(record_id: str, metadata: Sequence[Mapping[str, Any]], doi: str | None = None
                    ) -> SourceClassification:
    """Combine per-database labels of a (possibly merged) record.

    Any preprint signal -> preprint. Otherwise the known (non-unknown) labels must agree; conflicting labels
    -> unknown. No metadata at all -> unknown.
    """
    if not metadata:
        return SourceClassification(record_id, UNKNOWN, ("no type metadata retrieved",))
    labels = [classify_metadata(m, doi) for m in metadata]
    basis = tuple(dict.fromkeys(b for _, b in labels))
    statuses = [s for s, _ in labels]
    if PREPRINT in statuses:
        return SourceClassification(record_id, PREPRINT, basis)
    known = {s for s in statuses if s != UNKNOWN}
    if len(known) == 1:
        return SourceClassification(record_id, known.pop(), basis)
    if len(known) > 1:
        return SourceClassification(record_id, UNKNOWN, (*basis, "conflicting type metadata between sources"))
    return SourceClassification(record_id, UNKNOWN, basis)


# ------------------------------------------------------------------ policy


@dataclass
class PolicyOutcome:
    policy: str
    ordered_ids: list[str]
    excluded: list[dict[str, Any]] = field(default_factory=list)
    tiers: dict[str, int] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"policy": self.policy, "order_after_policy": list(self.ordered_ids),
                "excluded_by_policy": list(self.excluded), "excluded_count": len(self.excluded),
                "tier_by_id": dict(self.tiers)}


def apply_source_policy(ranked_ids: Sequence[str], statuses: Mapping[str, str], text_relevant: Mapping[str, bool],
                        policy: str) -> PolicyOutcome:
    """Re-order / filter a relevance ranking (list of record ids) by source policy (deterministic)."""
    if policy not in SOURCE_POLICIES:
        raise ValueError(f"unknown source policy {policy!r}")
    if policy == POLICY_ALLOW_ALL:
        return PolicyOutcome(policy, list(ranked_ids))
    if policy == POLICY_ONLY:
        kept, excluded = [], []
        for rid in ranked_ids:
            status = statuses.get(rid, UNKNOWN)
            if status == JOURNAL_ARTICLE:
                kept.append(rid)
            else:
                excluded.append({"record_id": rid, "source_status": status,
                                 "reason": "peer_reviewed_only: only 'peer-reviewed journal article' is allowed"})
        return PolicyOutcome(policy, kept, excluded)
    tiers: dict[str, int] = {}
    for rid in ranked_ids:
        journal = statuses.get(rid, UNKNOWN) == JOURNAL_ARTICLE
        relevant = bool(text_relevant.get(rid))
        tiers[rid] = (0 if journal else 1) if relevant else (2 if journal else 3)
    position = {rid: i for i, rid in enumerate(ranked_ids)}
    ordered = sorted(ranked_ids, key=lambda rid: (tiers[rid], position[rid]))
    return PolicyOutcome(policy, ordered, [], tiers)
