"""Deterministic relevance/diversity source selection for v0.2 retrieval (no model calls, no I/O).

Flow (see :mod:`sciforge.pipeline`)::

    expanded queries -> larger candidate pool per query and database
      -> deduplication (unchanged :mod:`sciforge.dedup`)
      -> title pre-score -> bounded abstract enrichment (:mod:`sciforge.abstract_enrichment`)
      -> deterministic scoring (this module: title + abstract components) + greedy concept-diverse ranking
      -> source policy (:mod:`sciforge.source_classification`)
      -> verification (unchanged :mod:`sciforge.verify`) with backfill
      -> at most ``target`` verified records (for the model layer: ``max_sources``)

Scoring uses ONLY text that was actually retrieved for a candidate (its title
and, when bounded abstract enrichment retrieved one, its abstract — see
:mod:`sciforge.abstract_enrichment`) plus the provenance of the candidate (which
expanded queries returned it, and at which rank). Nothing is inferred. Title
components (``components``):

* ``anchor_matched`` — the curated anchor concept of the question (e.g.
  platelet activation) appears in the text (+15);
* ``concepts_matched`` / ``concept_coverage`` — facet concepts of the question
  (shear stress, mechanotransduction, platelet membrane, membrane lipids /
  phospholipids, phosphatidylserine, lipid signaling) whose terms appear in the
  text (+10 each);
* ``keyword_hits`` — content words of the question (stopwords removed, prefix
  match) present in the text (+3 each; core-term relevance);
* ``query_origin`` — +2 if the verbatim-question query returned it, +1 per
  distinct query that returned it (at most +3).

Abstract components (``abstract_components``; only when an abstract was
retrieved; weighted at about half of the title weights and counting ONLY matches
the title does not already provide, so an abstract can add relevance but never
outweigh the same match in a title):

* ``abstract_anchor`` — anchor concept found in the abstract but not the title (+7);
* ``abstract_concept_coverage`` — facet concepts found only in the abstract (+5 each);
* ``abstract_core_term_relevance`` — question keywords found only in the abstract (+1 each).

``base_score`` = sum of both component groups. ``concepts_matched`` (used for
concept diversity below) is the union of title and abstract concepts.

Selection is greedy over the whole pool: at each step the candidate with the
highest ``selection_score = base_score + 12 * (question concepts it covers that
no already-selected candidate covers) - 15 * (already-selected candidates with
exactly the same concept profile) - 20 * (near-duplicate title of a selected
candidate)`` is taken. Ties: higher base score, then better (lower) best rank,
then ``record_id`` ascending — so the same input always yields the same order.
The full ranking is kept so verification can backfill in rank order.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from sciforge.models import Record, VerificationResult
from sciforge.normalize import normalize_title
from sciforge.query_expansion import CONCEPTS, keywords

__all__ = ["CONCEPT_TEXT_PATTERNS", "W_ABSTRACT_ANCHOR", "W_ABSTRACT_CONCEPT", "W_ABSTRACT_KEYWORD", "CandidateScore", "RankedCandidate", "SelectionOutcome", "VERIFY_BACKFILL_FACTOR",
           "rank_candidates", "score_candidates", "select_and_verify"]

METHOD = ("deterministic title + bounded-abstract relevance scoring + greedy concept-diversity ranking (no model), "
          "source policy ordering/filtering, verification backfill in policy order")

W_ANCHOR = 15
W_CONCEPT = 10
W_KEYWORD = 3
W_ORIGINAL_QUERY = 2
W_ABSTRACT_ANCHOR = 7
W_ABSTRACT_CONCEPT = 5
W_ABSTRACT_KEYWORD = 1
MAX_QUERY_COUNT_BONUS = 3
W_NOVELTY = 12
REDUNDANCY_PENALTY = 15
NEAR_DUPLICATE_PENALTY = 20
NEAR_DUPLICATE_JACCARD = 0.75
# Verification backfill: at most target * factor candidates are ever sent to verification.
VERIFY_BACKFILL_FACTOR = 3

_PLATELET = r"\bplatelets?\b|\bthrombocyt\w*"
ANCHOR_CONCEPT_IDS = frozenset(c.concept_id for c in CONCEPTS if c.role == "anchor")

# Text patterns (over the lower-cased title/abstract) per curated concept. Derived from the concept
# terms of :data:`sciforge.query_expansion.CONCEPTS`; every listed group must match (AND of ORs).
CONCEPT_TEXT_PATTERNS: dict[str, tuple[str, ...]] = {
    "platelet_activation": (_PLATELET, r"\bactivat\w*|\baggregat\w*"),
    "shear_stress": (r"\bshear\w*",),
    "mechanotransduction": (r"\bmechanotransduc\w*|\bmechanosens\w*",),
    "platelet_membrane": (r"\bmembran\w*",),
    "membrane_lipids": (r"\blipid\w*|\bphospholipid\w*",),
    "phosphatidylserine": (r"\bphosphatidylserine\b|\bprocoagulant\w*",),
    "lipid_signaling": (r"\blipid[- ]signal\w*|\blipid[- ]mediat\w*|\bsignal\w* lipid\w*",),
}

_WORD_RE = re.compile(r"[a-z0-9]+")


def _text_matches(concept_id: str, text: str) -> bool:
    groups = CONCEPT_TEXT_PATTERNS.get(concept_id)
    return bool(groups) and all(re.search(p, text) for p in groups)  # type: ignore[arg-type]


def _stems(question: str) -> list[str]:
    stems: list[str] = []
    for word in keywords(question):
        for part in word.split("-"):
            if len(part) >= 3:
                stem = part[:6]
                if stem not in stems:
                    stems.append(stem)
    return stems


def _title_tokens(title: str | None) -> frozenset[str]:
    norm = normalize_title(title) or ""
    return frozenset(_WORD_RE.findall(norm))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _qnum(query_id: str) -> int:
    digits = query_id.lstrip("q")
    return int(digits) if digits.isdigit() else 10**6


@dataclass(frozen=True)
class CandidateScore:
    """Deterministic score of one deduplicated candidate (no model involved)."""

    record_id: str
    title: str | None
    concepts_matched: tuple[str, ...]
    anchor_matched: bool
    keyword_hits: tuple[str, ...]
    query_ids: tuple[str, ...]
    databases: tuple[str, ...]
    best_rank: int | None
    components: dict[str, int] = field(default_factory=dict)
    base_score: int = 0
    text_used: tuple[str, ...] = ("title",)
    abstract_components: dict[str, int] = field(default_factory=dict)
    title_concepts: tuple[str, ...] = ()
    abstract_only_concepts: tuple[str, ...] = ()

    @property
    def text_relevant(self) -> bool:
        """True when the title or abstract matched at least one question concept or keyword."""
        return bool(self.concepts_matched or self.keyword_hits)

    @property
    def facet_concepts(self) -> frozenset[str]:
        return frozenset(c for c in self.concepts_matched if c not in ANCHOR_CONCEPT_IDS)

    def to_json(self) -> dict[str, Any]:
        return {"record_id": self.record_id, "title": self.title, "text_used": list(self.text_used),
                "concepts_matched": list(self.concepts_matched), "anchor_matched": self.anchor_matched,
                "keyword_hits": list(self.keyword_hits), "query_ids": list(self.query_ids),
                "databases": list(self.databases), "best_rank": self.best_rank,
                "components": dict(self.components), "abstract_components": dict(self.abstract_components),
                "abstract_only_concepts": list(self.abstract_only_concepts), "base_score": self.base_score}


@dataclass(frozen=True)
class RankedCandidate:
    rank: int
    score: CandidateScore
    selection_score: int
    new_concepts: tuple[str, ...]
    near_duplicate_of: str | None
    same_profile_selected: int = 0

    @property
    def record_id(self) -> str:
        return self.score.record_id

    def to_json(self) -> dict[str, Any]:
        return {"rank": self.rank, **self.score.to_json(), "new_concepts": list(self.new_concepts),
                "near_duplicate_of": self.near_duplicate_of,
                "same_profile_selected": self.same_profile_selected, "selection_score": self.selection_score}


def score_candidates(question: str, records: Sequence[Record],
                     origins: Mapping[str, Iterable[tuple[str, str, int]]],
                     abstracts: Mapping[str, str] | None = None) -> list[CandidateScore]:
    """Score every deduplicated record.

    ``origins`` maps a (pre-dedup) ``record_id`` to ``(query_id, database, rank)`` tuples (rank 1-based);
    a merged record collects the origins of every record id in its provenance. ``abstracts`` (optional)
    maps record ids to abstract text that was actually retrieved (bounded abstract enrichment).
    """
    qtext = question.lower()
    relevant = [c for c in CONCEPTS if c.matches(qtext)]
    relevant_ids = [c.concept_id for c in relevant]
    anchor_ids = {c.concept_id for c in relevant if c.role == "anchor"}
    stems = _stems(question)
    out: list[CandidateScore] = []

    def analyse(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        low = text.lower()
        tokens = _WORD_RE.findall(low)
        matched = tuple(cid for cid in relevant_ids if _text_matches(cid, low))
        hits = tuple(st for st in stems if any(t.startswith(st) for t in tokens))
        return matched, hits

    for record in records:
        used = ["title"] if record.title else []
        t_matched, t_hits = analyse(record.title or "")
        abstract = (abstracts or {}).get(record.record_id)
        a_matched: tuple[str, ...] = ()
        a_hits: tuple[str, ...] = ()
        if abstract:
            used.append("abstract")
            a_matched, a_hits = analyse(abstract)
        matched = tuple(cid for cid in relevant_ids if cid in t_matched or cid in a_matched)
        hits = tuple(st for st in stems if st in t_hits or st in a_hits)
        anchor = any(cid in anchor_ids for cid in matched)
        t_anchor = any(cid in anchor_ids for cid in t_matched)
        ids = [p.source_record_id for p in record.provenance] or [record.record_id]
        if record.record_id not in ids:
            ids.insert(0, record.record_id)
        seen: set[tuple[str, str, int]] = set()
        for rid in ids:
            seen.update(origins.get(rid, ()))
        query_ids = tuple(sorted({q for q, _, _ in seen}, key=_qnum))
        databases = tuple(sorted({d for _, d, _ in seen}))
        best_rank = min((r for _, _, r in seen), default=None)
        t_facets = [c for c in t_matched if c not in anchor_ids]
        components = {
            "anchor": W_ANCHOR if t_anchor else 0,
            "concept_coverage": W_CONCEPT * len(t_facets),
            "core_term_relevance": W_KEYWORD * len(t_hits),
            "query_origin": (W_ORIGINAL_QUERY if "q1" in query_ids else 0) + min(MAX_QUERY_COUNT_BONUS,
                                                                                 len(query_ids)),
        }
        abstract_only = tuple(c for c in a_matched if c not in t_matched)
        abstract_components: dict[str, int] = {}
        if abstract:
            abstract_components = {
                "abstract_anchor": W_ABSTRACT_ANCHOR if (anchor and not t_anchor) else 0,
                "abstract_concept_coverage": W_ABSTRACT_CONCEPT * sum(1 for c in abstract_only
                                                                      if c not in anchor_ids),
                "abstract_core_term_relevance": W_ABSTRACT_KEYWORD * sum(1 for h in a_hits if h not in t_hits),
            }
        out.append(CandidateScore(record_id=record.record_id, title=record.title, concepts_matched=matched,
                                  anchor_matched=anchor, keyword_hits=hits, query_ids=query_ids, databases=databases,
                                  best_rank=best_rank, components=components,
                                  base_score=sum(components.values()) + sum(abstract_components.values()),
                                  text_used=tuple(used), abstract_components=abstract_components,
                                  title_concepts=t_matched, abstract_only_concepts=abstract_only))
    return out


def rank_candidates(scores: Sequence[CandidateScore]) -> list[RankedCandidate]:
    """Greedy concept-diverse ranking of ALL candidates (deterministic; see module docstring)."""
    remaining = sorted(scores, key=lambda s: s.record_id)
    covered: set[str] = set()
    profiles: list[frozenset[str]] = []
    chosen_tokens: list[tuple[str, frozenset[str]]] = []
    tokens = {s.record_id: _title_tokens(s.title) for s in scores}
    ranked: list[RankedCandidate] = []
    while remaining:
        best: tuple[tuple[int, int, int, str], RankedCandidate] | None = None
        for s in remaining:
            new = tuple(sorted(c for c in s.facet_concepts if c not in covered))
            same = sum(1 for p in profiles if p == s.facet_concepts)
            dup = next((rid for rid, tok in chosen_tokens
                        if _jaccard(tokens[s.record_id], tok) >= NEAR_DUPLICATE_JACCARD), None)
            sel = (s.base_score + W_NOVELTY * len(new) - REDUNDANCY_PENALTY * same
                   - (NEAR_DUPLICATE_PENALTY if dup else 0))
            key = (-sel, -s.base_score, s.best_rank if s.best_rank is not None else 10**6, s.record_id)
            if best is None or key < best[0]:
                best = (key, RankedCandidate(rank=len(ranked) + 1, score=s, selection_score=sel, new_concepts=new,
                                             near_duplicate_of=dup, same_profile_selected=same))
        assert best is not None
        chosen = best[1]
        ranked.append(chosen)
        covered.update(chosen.score.facet_concepts)
        profiles.append(chosen.score.facet_concepts)
        chosen_tokens.append((chosen.record_id, tokens[chosen.record_id]))
        remaining.remove(chosen.score)
    return ranked


@dataclass
class SelectionOutcome:
    """Result of :func:`select_and_verify`."""

    target: int
    checked: list[Record]                       # records sent to verification, in rank order
    verification: list[VerificationResult]      # aligned with ``checked``
    selected_ids: list[str]                     # accepted (verification passed), in rank order
    rounds: list[dict[str, Any]]
    max_checked: int
    accepted_statuses: tuple[str, ...]
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        vmap = {v.record_id: v.status for v in self.verification}
        return {"target": self.target, "accepted_verification_statuses": list(self.accepted_statuses),
                "max_candidates_verified": self.max_checked, "candidates_verified": len(self.checked),
                "verification_rounds": self.rounds,
                "verification_failed_ids": [r.record_id for r in self.checked
                                            if r.record_id not in self.selected_ids],
                "verification_status_by_id": {r.record_id: vmap.get(r.record_id) for r in self.checked},
                "backfilled": len(self.rounds) > 1, "selected_record_ids": list(self.selected_ids),
                "error": self.error}


def select_and_verify(ranked: Sequence[RankedCandidate], records_by_id: Mapping[str, Record],
                      verify: Callable[[list[Record]], list[VerificationResult]], target: int, *,
                      accept_partially_verified: bool = False,
                      backfill_factor: int = VERIFY_BACKFILL_FACTOR) -> SelectionOutcome:
    """Verify the top-``target`` candidates; backfill from the ranking while fewer than ``target`` pass.

    A candidate "passes" when its verification status is ``verified`` (or ``partially_verified`` when
    ``accept_partially_verified``). At most ``target * backfill_factor`` candidates are verified in total,
    so a failed verification never silently shrinks the final set while verified candidates remain within
    that bound. ``verify`` is the unchanged v0.2 :meth:`Verifier.verify_all`. An exception from ``verify``
    stops the loop (recorded in ``error``); results of earlier rounds are kept.
    """
    if target < 1:
        raise ValueError("target must be >= 1")
    accepted_statuses = ("verified", "partially_verified") if accept_partially_verified else ("verified",)
    max_checked = min(len(ranked), max(target, target * max(1, backfill_factor)))
    outcome = SelectionOutcome(target=target, checked=[], verification=[], selected_ids=[], rounds=[],
                               max_checked=max_checked, accepted_statuses=accepted_statuses)
    pos = 0
    while len(outcome.selected_ids) < target and pos < max_checked:
        need = target - len(outcome.selected_ids)
        batch = [records_by_id[c.record_id] for c in ranked[pos:min(pos + need, max_checked)]]
        pos += len(batch)
        try:
            results = verify(batch)
        except Exception as exc:  # noqa: BLE001 - verification must not crash the run
            outcome.error = f"{type(exc).__name__}: {exc}"
            break
        by_id = {v.record_id: v for v in results}
        passed = []
        for rec in batch:
            v = by_id.get(rec.record_id)
            outcome.checked.append(rec)
            if v is not None:
                outcome.verification.append(v)
                if v.status in accepted_statuses:
                    outcome.selected_ids.append(rec.record_id)
                    passed.append(rec.record_id)
        outcome.rounds.append({"round": len(outcome.rounds) + 1, "verified_ids": [r.record_id for r in batch],
                               "passed_ids": passed})
    return outcome
