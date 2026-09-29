"""Shared builders for the v0.3 Milestone 2 tests (offline only)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx

from conftest import Sleeper, mock_client
from sciforge.config import Settings
from sciforge.http_utils import HttpFetcher, RateLimiter
from sciforge.llm.budget import BudgetLimits, BudgetTracker
from sciforge.logging_utils import RunLog
from sciforge.models import Record, VerificationResult

TS = "2026-09-28T00:00:00Z"

# A record with every bibliographic field set to a distinctive value.
BIB = {
    "title": "Mechanosensitive Zebrafish Thrombocyte Activation Study",
    "authors": ["Quixote Albemarle", "Vandersloot Perpetua"],
    "journal": "Annals of Improbable Hemorheology",
    "year": 1987,
    "doi": "10.5555/zqx.1987.424242",
    "pmid": "31415926",
}

STRUCTURED_ABSTRACT_XML = (
    '<AbstractText Label="BACKGROUND" NlmCategory="BACKGROUND">Platelet activation under <i>high</i> shear '
    'is poorly understood.</AbstractText>'
    '<AbstractText Label="METHODS" NlmCategory="METHODS">Human platelets were exposed to shear stress of '
    '50 dyn/cm<sup>2</sup> for 10 minutes.</AbstractText>'
    '<AbstractText Label="RESULTS" NlmCategory="RESULTS">Shear exposure increased P-selectin expression by '
    '40% compared with static controls.</AbstractText>'
    '<AbstractText Label="CONCLUSIONS" NlmCategory="CONCLUSIONS">High shear stress directly activates '
    'platelets in vitro.</AbstractText>'
)
STRUCTURED_ABSTRACT_TEXT = (
    "BACKGROUND: Platelet activation under high shear is poorly understood. "
    "METHODS: Human platelets were exposed to shear stress of 50 dyn/cm2 for 10 minutes. "
    "RESULTS: Shear exposure increased P-selectin expression by 40% compared with static controls. "
    "CONCLUSIONS: High shear stress directly activates platelets in vitro."
)
PLAIN_ABSTRACT_TEXT = "Von Willebrand factor unfolds under elongational flow and binds platelet GPIb more strongly."
CROSSREF_JATS = (
    "<jats:title>Abstract</jats:title><jats:p>Red cell deformability declines with storage age; "
    "<jats:italic>ex vivo</jats:italic> rheometry showed a 25% loss after 35 days.</jats:p>"
)
CROSSREF_TEXT = ("Red cell deformability declines with storage age; ex vivo rheometry showed a 25% loss "
                 "after 35 days.")


def record(*, pmid: str | None = BIB["pmid"], doi: str | None = BIB["doi"], title: str = BIB["title"],
           authors: list[str] | None = None, journal: str = BIB["journal"], year: int = BIB["year"],
           source: str = "pubmed") -> Record:
    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if (source == "pubmed" and pmid) else (
        f"https://doi.org/{doi}" if doi else None)
    return Record(title=title, authors=list(authors or BIB["authors"]), year=year, doi=doi, pmid=pmid,
                  journal=journal, source_database=source, source_url=url, retrieval_timestamp=TS)


def verified(rec: Record, status: str = "verified") -> VerificationResult:
    return VerificationResult(record_id=rec.record_id, status=status, verified_at=TS)


def pubmed_article(pmid: str, abstract_xml: str | None) -> str:
    abstract = f"<Abstract>{abstract_xml}</Abstract>" if abstract_xml is not None else ""
    return (f'<PubmedArticle><MedlineCitation Status="MEDLINE" Owner="NLM"><PMID Version="1">{pmid}</PMID>'
            f"<Article><ArticleTitle>Some title not used</ArticleTitle>{abstract}</Article></MedlineCitation>"
            f"</PubmedArticle>")


def efetch_xml(articles: list[tuple[str, str | None]], *, doctype: bool = True) -> str:
    head = '<?xml version="1.0" ?>\n'
    if doctype:
        head += ('<!DOCTYPE PubmedArticleSet PUBLIC "-//NLM//DTD PubMedArticle, 1st January 2025//EN" '
                 '"https://dtd.nlm.nih.gov/ncbi/pubmed/out/pubmed_250101.dtd">\n')
    return head + "<PubmedArticleSet>" + "".join(pubmed_article(p, a) for p, a in articles) + "</PubmedArticleSet>"


def xml_response(text: str, status: int = 200) -> httpx.Response:
    return httpx.Response(status, content=text.encode(), headers={"Content-Type": "text/xml"})


def crossref_response(doi: str, abstract: str | None) -> httpx.Response:
    work: dict[str, Any] = {"DOI": doi, "title": ["Some title"]}
    if abstract is not None:
        work["abstract"] = abstract
    payload = {"status": "ok", "message-type": "work", "message": work}
    return httpx.Response(200, content=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})


class Api:
    """Mock PubMed efetch + Crossref /works/{doi} with per-id content."""

    def __init__(self, pubmed: dict[str, str | None] | None = None, crossref: dict[str, str | None] | None = None,
                 efetch_status: int = 200, efetch_body: str | None = None) -> None:
        self.pubmed = pubmed or {}
        self.crossref = crossref or {}
        self.efetch_status = efetch_status
        self.efetch_body = efetch_body
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path.endswith("efetch.fcgi"):
            if self.efetch_status != 200:
                return httpx.Response(self.efetch_status)
            if self.efetch_body is not None:
                return xml_response(self.efetch_body)
            ids = request.url.params.get("id", "").split(",")
            return xml_response(efetch_xml([(i, self.pubmed[i]) for i in ids if i in self.pubmed]))
        if path.startswith("/works/"):
            doi = path[len("/works/"):]
            if doi in self.crossref:
                return crossref_response(doi, self.crossref[doi])
            return httpx.Response(404)
        return httpx.Response(500)

    def efetch_requests(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.path.endswith("efetch.fcgi")]

    def crossref_requests(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.path.startswith("/works/")]


def make_fetcher(handler: Callable[[httpx.Request], httpx.Response], settings: Settings,
                 sleeper: Sleeper | None = None) -> HttpFetcher:
    return HttpFetcher(mock_client(handler), settings, RunLog(settings.secret_values()), sleep=sleeper or Sleeper())


def no_throttle() -> RateLimiter:
    return RateLimiter(0, sleep=Sleeper())


def tracker(max_sources: int = 10, max_attempts: int = 30) -> BudgetTracker:
    return BudgetTracker(BudgetLimits(max_sources=max_sources, max_attempts=max_attempts, max_spend_usd=None))


def item(record_id: str, quote: str, **overrides: Any) -> dict[str, Any]:
    base = {
        "source_record_id": record_id,
        "claim": "High shear stress increases platelet P-selectin expression.",
        "quote": quote,
        "finding": "P-selectin expression rose by 40% versus static controls.",
        "methods": "Human platelets exposed to 50 dyn/cm2 shear for 10 minutes.",
        "limitations": "In vitro study; abstract only.",
        "relevance": "Directly addresses shear-induced platelet activation.",
        "evidence_category": "established",
        "confidence": "moderate",
    }
    base.update(overrides)
    return base


def question_output(**overrides: Any) -> dict[str, Any]:
    base = {
        "research_question": "Does high shear stress directly activate human platelets?",
        "scope": "In vitro and in vivo studies of human platelets under shear.",
        "assumptions": ["Shear magnitude is reported in dyn/cm2."],
        "key_concepts": ["shear stress", "platelet activation", "P-selectin"],
        "ambiguities": ["Threshold for 'high' shear is not defined."],
    }
    base.update(overrides)
    return base
