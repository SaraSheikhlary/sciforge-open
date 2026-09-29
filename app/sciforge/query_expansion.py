"""Deterministic, rule-based query expansion for v0.2 retrieval (no model calls, no I/O).

Given a research question, :func:`expand_queries` returns a :class:`QueryPlan`:

1. the ORIGINAL question, verbatim, as the primary query for both databases
   (exactly what v0.2 searched before expansion existed), followed by
2. up to :data:`MAX_FOCUSED_QUERIES` focused queries, each combining the
   *anchor* concept (e.g. ``platelet activation``) with ONE *facet* concept
   (e.g. ``shear stress``), formatted for PubMed (``"term"[tiab]`` with
   ``AND``/``OR``; no MeSH) and for Crossref (plain words for
   ``query.bibliographic``).

Concepts come from a small curated map (:data:`CONCEPTS`) whose triggers are
regular expressions over the lower-cased question; everything else is plain
stopword removal. Rules are applied in a fixed order, so the same question
always gives the same plan (deduplicated, capped). The plan contains search
terms only — never identifiers (DOI/PMID), author names, journals, years,
findings or citations; numbers and identifier-like tokens are dropped from the
keyword list.

This module only builds queries. Merging per-query search results
(:func:`merge_query_outcomes`) keeps the first occurrence of an identical hit
(same ``record_id``, i.e. same database and identifier) returned by several
queries and caps each database at ``max_results`` (the pipeline passes a cap
large enough to keep the whole candidate pool); cross-database duplicate
detection, merging and citation verification are still done by the existing,
unchanged v0.2 ``dedup`` and ``verify`` modules, and the final set is chosen by
:mod:`sciforge.source_selection`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from sciforge.models import Record, SearchOutcome

__all__ = ["CONCEPTS", "MAX_FOCUSED_QUERIES", "Concept", "PlannedQuery", "QueryPlan", "expand_queries",
           "merge_query_outcomes", "single_query_plan"]

MAX_FOCUSED_QUERIES = 6
MAX_KEYWORDS = 8
METHOD = "deterministic rule-based expansion (curated concept map + stopword removal; no model)"


@dataclass(frozen=True)
class Concept:
    """One curated concept. ``triggers``: every listed regex must match (AND); ``any_of``: at least one."""

    concept_id: str
    label: str
    terms: tuple[str, ...]           # search terms (first is the preferred label)
    role: str                        # "anchor" | "facet"
    triggers: tuple[str, ...] = ()
    any_of: tuple[str, ...] = ()

    def matches(self, text: str) -> bool:
        if self.triggers and not all(re.search(p, text) for p in self.triggers):
            return False
        if self.any_of and not any(re.search(p, text) for p in self.any_of):
            return False
        return bool(self.triggers or self.any_of)


_PLATELET = r"\bplatelets?\b|\bthrombocytes?\b"
_ACTIVATION = r"\bactivat\w*"
_SHEAR = r"\bshear\w*"
_LIPID = r"\blipids?\b|\blipid-\w+|\bphospholipid\w*|\bphosphatidylserine\b"

# Fixed order = output order. Anchors first (the first matching anchor is used), then facets.
CONCEPTS: tuple[Concept, ...] = (
    Concept("platelet_activation", "platelet activation", ("platelet activation",), "anchor",
            triggers=(_PLATELET,), any_of=(_ACTIVATION, r"\baggregat\w*", _SHEAR, _LIPID)),
    Concept("shear_stress", "shear stress", ("shear stress", "shear-induced"), "facet",
            any_of=(_SHEAR,)),
    Concept("mechanotransduction", "mechanotransduction", ("mechanotransduction",), "facet",
            any_of=(r"\bmechanotransduc\w*", r"\bmechanosens\w*", rf"(?:{_SHEAR}).*(?:{_ACTIVATION})",
                    rf"(?:{_ACTIVATION}).*(?:{_SHEAR})")),
    Concept("platelet_membrane", "platelet membrane", ("platelet membrane",), "facet",
            triggers=(_PLATELET,), any_of=(r"\bmembranes?\b", _LIPID)),
    Concept("membrane_lipids", "membrane lipids / phospholipids", ("membrane lipids", "phospholipids"), "facet",
            any_of=(r"\blipids?\b", r"\blipid-\w+", r"\bphospholipid\w*")),
    Concept("phosphatidylserine", "phosphatidylserine", ("phosphatidylserine",), "facet",
            any_of=(r"\bphosphatidylserine\b", r"\blipids?\b", r"\blipid-\w+", r"\bprocoagulant\w*")),
    Concept("lipid_signaling", "lipid signaling", ("lipid signaling",), "facet",
            any_of=(r"\blipids?\b", r"\blipid-\w+", r"\blipid signal\w*")),
)

STOPWORDS = frozenset("""
a about above after again against all also am an and any are as at be because been before being below between
both but by can could did do does doing down during each either few for from further had has have having how
however i if in into is it its itself just may might more most much must no nor not of off on once only or other
our out over own per same should so some such than that the their them then there these they this those through
to too under until up upon very via was we were what when where whether which while who whom why will with within
without would yet role roles effect effects impact impacts influence influences relationship association
associated mechanism mechanisms related mediated dependent based study studies evidence current literature
research question known unknown directly indirectly human humans affect affects alter alters cause causes change
changes contribute contributes determine determines drive drives explain explains increase increases decrease
decreases reduce reduces regulate regulates promote promotes inhibit inhibits compare compared versus
""".split())

_TOKEN_RE = re.compile(r"[a-z][a-z\-]*[a-z]")


@dataclass(frozen=True)
class PlannedQuery:
    query_id: str
    kind: str                            # "original" | "focused"
    pubmed: str
    crossref: str
    concepts: tuple[str, ...] = ()       # concept ids that produced the query

    def to_json(self) -> dict[str, Any]:
        return {"query_id": self.query_id, "kind": self.kind, "pubmed": self.pubmed, "crossref": self.crossref,
                "concepts": list(self.concepts)}


@dataclass(frozen=True)
class QueryPlan:
    question: str
    enabled: bool
    queries: tuple[PlannedQuery, ...]
    concepts: tuple[dict[str, Any], ...] = ()
    keywords: tuple[str, ...] = ()
    anchor: str | None = None
    notes: tuple[str, ...] = field(default=())

    def to_json(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "method": METHOD if self.enabled else "disabled: question used verbatim",
                "question": self.question, "anchor": self.anchor, "concepts": [dict(c) for c in self.concepts],
                "keywords": list(self.keywords), "queries": [q.to_json() for q in self.queries],
                "max_focused_queries": MAX_FOCUSED_QUERIES, "notes": list(self.notes)}


def _original(question: str) -> PlannedQuery:
    return PlannedQuery("q1", "original", question, question, ())


def single_query_plan(question: str, *, enabled: bool = False, note: str | None = None) -> QueryPlan:
    """The pre-expansion behaviour: one query, the question verbatim."""
    q = question.strip()
    return QueryPlan(question=q, enabled=enabled, queries=(_original(q),), notes=(note,) if note else ())


def keywords(question: str) -> list[str]:
    """Content words (stopwords, numbers and identifier-like tokens removed), first-occurrence order, deduped."""
    text = question.lower()
    # drop anything identifier-like before tokenising (DOIs, URLs, PMIDs, emails)
    text = re.sub(r"\b10\.\d{4,9}/\S+|https?://\S+|\bpmid:?\s*\d+|\S+@\S+", " ", text)
    out: list[str] = []
    for token in _TOKEN_RE.findall(text):
        token = token.strip("-")
        if len(token) < 3 or token in STOPWORDS or token in out:
            continue
        out.append(token)
    return out[:MAX_KEYWORDS]


def _pubmed_group(terms: Sequence[str]) -> str:
    parts = [f'"{t}"[tiab]' for t in terms]
    return parts[0] if len(parts) == 1 else "(" + " OR ".join(parts) + ")"


def _anchor_from_keywords(words: Sequence[str], facet_words: set[str]) -> list[str]:
    """Up to three non-facet content words (used when no curated anchor concept matched)."""
    return [w for w in words if w not in facet_words][:3]


def expand_queries(question: str) -> QueryPlan:
    """Original question + focused anchor×facet queries (deterministic, deduplicated, capped)."""
    q = question.strip()
    if not q:
        raise ValueError("research question must not be empty")
    text = q.lower()
    matched = [c for c in CONCEPTS if c.matches(text)]
    words = keywords(q)
    anchor_concept = next((c for c in matched if c.role == "anchor"), None)
    facets = [c for c in matched if c.role == "facet"]
    notes: list[str] = []

    anchor_pubmed: str | None
    anchor_crossref: list[str]
    if anchor_concept is not None:
        anchor_pubmed = _pubmed_group(anchor_concept.terms)
        anchor_crossref = list(anchor_concept.terms)
        anchor_label: str | None = anchor_concept.label
        anchor_ids: tuple[str, ...] = (anchor_concept.concept_id,)
    else:
        facet_words = {w for c in facets for t in c.terms for w in t.split()}
        facet_words |= {w for w in words if any(re.search(p, w) for c in facets for p in c.any_of + c.triggers)}
        core = _anchor_from_keywords(words, facet_words)
        anchor_pubmed = ("(" + " AND ".join(f"{w}[tiab]" for w in core) + ")") if len(core) > 1 else (
            f"{core[0]}[tiab]" if core else None)
        anchor_crossref = core
        anchor_label = " ".join(core) if core else None
        anchor_ids = ("keyword_anchor",) if core else ()
        if not facets:
            notes.append("no curated concept matched; only the original question is searched")

    queries: list[PlannedQuery] = [_original(q)]
    seen = {(q, q)}
    for facet in facets:
        if len(queries) - 1 >= MAX_FOCUSED_QUERIES:
            notes.append(f"focused queries capped at {MAX_FOCUSED_QUERIES}")
            break
        if anchor_pubmed:
            pubmed = f"{anchor_pubmed} AND {_pubmed_group(facet.terms)}"
            crossref = " ".join([*anchor_crossref, *facet.terms])
        else:
            pubmed = _pubmed_group(facet.terms)
            crossref = " ".join(facet.terms)
        key = (pubmed, crossref)
        if key in seen:
            continue
        seen.add(key)
        queries.append(PlannedQuery(f"q{len(queries) + 1}", "focused", pubmed, crossref,
                                    (*anchor_ids, facet.concept_id)))
    concepts = tuple({"concept_id": c.concept_id, "label": c.label, "role": c.role, "terms": list(c.terms)}
                     for c in matched)
    return QueryPlan(question=q, enabled=True, queries=tuple(queries), concepts=concepts, keywords=tuple(words),
                     anchor=anchor_label, notes=tuple(notes))


# ------------------------------------------------------------------ merging per-query results


def _aggregate_status(statuses: Sequence[str]) -> str:
    if all(s == "ok" for s in statuses):
        return "ok"
    if all(s == "failed" for s in statuses):
        return "failed"
    return "partial"


def merge_query_outcomes(database: str, outcomes: Sequence[SearchOutcome], max_results: int
                         ) -> tuple[SearchOutcome, list[dict[str, Any]]]:
    """Merge one database's per-query outcomes (in plan order) into one SearchOutcome.

    * One query: returned unchanged (identical to the pre-expansion behaviour).
    * Several: records are interleaved by rank (1st hit of every query, then 2nd, ...),
      an identical hit (same ``record_id``) already contributed by an EARLIER-listed
      query is skipped, and the total is capped at ``max_results``. Hits repeated within
      one query are left for the unchanged v0.2 dedup, exactly as before.
    * ``status``: ok if every query was ok, failed if every query failed, else partial.
    * ``total_hits``: the primary (original-question) query's reported total.

    Returns the merged outcome and per-query stats for the run log.
    """
    if len(outcomes) == 1:
        o = outcomes[0]
        return o, [{"status": o.status, "total_hits_reported": o.total_hits, "retrieved": len(o.records),
                    "contributed": len(o.records)}]
    first_owner: dict[str, int] = {}
    contributed = [0] * len(outcomes)
    merged: list[Record] = []
    depth = max((len(o.records) for o in outcomes), default=0)
    for rank in range(depth):
        for qi, o in enumerate(outcomes):
            if len(merged) >= max_results:
                break
            if rank >= len(o.records):
                continue
            rec = o.records[rank]
            owner = first_owner.setdefault(rec.record_id, qi)
            if owner != qi:
                continue
            merged.append(rec)
            contributed[qi] += 1
    stats = [{"status": o.status, "total_hits_reported": o.total_hits, "retrieved": len(o.records),
              "contributed": contributed[i]} for i, o in enumerate(outcomes)]
    outcome = SearchOutcome(database=database, status=_aggregate_status([o.status for o in outcomes]),  # type: ignore[arg-type]
                            records=merged, total_hits=outcomes[0].total_hits)
    return outcome, stats
