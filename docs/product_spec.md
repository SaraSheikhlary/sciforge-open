# SciForge v0.1 Product Specification

## 1. Summary

SciForge takes a scientific research question and returns a structured, source-traceable research brief. Its defining property is traceability: every claim in the output is linked to a verified source and labeled by evidence strength.

## 2. Users

- **Primary:** researchers (graduate students, postdocs, faculty, industry scientists) starting or updating a literature investigation.
- **Secondary:** research-adjacent readers (reviewers, program managers, science writers) who need a fast, auditable view of what a literature says.

Users are assumed to have domain expertise and to use SciForge as an aid to their own judgment.

## 3. Problem

- Manual literature review is slow and hard to reproduce.
- General-purpose language models produce fluent summaries but may invent citations, misstate findings, or present guesses as evidence, with no audit trail.

## 4. Core user flow

1. The user enters a research question, plus optional constraints: publication date range, target organisms / cell types / materials, and papers they already know about (as DOIs or PMIDs).
2. SciForge restates the question precisely and lists its assumptions.
3. SciForge searches the literature and logs every query.
4. SciForge screens results, extracts evidence, and builds evidence records.
5. SciForge verifies citations and checks claims against sources.
6. SciForge produces the report and saves it together with its search log and evidence records.

## 5. Output: the research report

| Section | Content |
|---|---|
| A. Research Question | Precise restatement, scope, assumptions |
| B. Search Strategy | Concepts and synonyms, exact queries, databases, dates run, result counts, inclusion/exclusion criteria |
| C. Key Findings | Main conclusions, each labeled and linked to evidence records |
| D. Evidence Matrix | One row per evidence record |
| E. Conflicting Evidence | Disagreements side by side with likely explanations |
| F. Limitations | Limits of the evidence and of this investigation (including abstract-only sources) |
| G. Research Gaps | Unanswered or weakly answered questions |
| H. Candidate Hypotheses | Testable hypotheses, labeled as such, tied to gaps and evidence |
| I. Proposed Next Steps | Concrete literature, computational, or experimental steps |
| J. Sources | Full citations with DOI / PMID / URL and verification status |

Every statement in sections C, E, G, and H carries one label: **established**, **conflicting**, **inference**, or **hypothesis**.

## 6. Functional requirements

| ID | Requirement |
|---|---|
| FR-1 | Accept a free-text question and optional date range, targets, and seed identifiers. |
| FR-2 | Produce a precise question definition with scope and assumptions. |
| FR-3 | Generate search queries from key concepts and synonyms; log each query, database, timestamp, and result count. |
| FR-4 | Retrieve records from at least PubMed and Crossref; additional sources are optional. |
| FR-5 | Prefer primary research; label preprints as not peer-reviewed. |
| FR-6 | Extract evidence records in the schema defined in `agent/research_workflow.md`. |
| FR-7 | Record for each source whether full text or only the abstract was read. |
| FR-8 | Verify every cited DOI or PMID against Crossref or PubMed and confirm title, authors, and year match. |
| FR-9 | Check each claim against the source text it cites; mark it supported, partially supported, unsupported, or not verified. |
| FR-10 | Exclude unsupported claims from Key Findings, or flag them visibly. |
| FR-11 | Produce the report as Markdown, with evidence records as JSON and the search log as Markdown. |
| FR-12 | Use `null` or `"not verified"` for any field that could not be confirmed. Never fabricate. |

## 7. Non-functional requirements

- **Traceability:** every claim in a report resolves to an evidence record, which resolves to a verified source.
- **Reproducibility:** a saved search log is sufficient to rerun the search.
- **Security:** secrets only via environment variables (see `SECURITY.md`).
- **Transparency:** reports state which model was used, when, and what could not be accessed.
- **Cost control:** configurable caps on the number of sources and model calls per investigation.

## 8. Out of scope for v0.1 through v0.3

- Reading paywalled full text or bypassing access controls
- Uploading or analyzing private documents or experimental data
- Multi-user accounts, hosting, or collaboration features
- Automated publication of reports anywhere

## 9. Success criteria

Measured with the evaluation framework in `evaluation/`:

- **Citation existence:** 100% of cited sources resolve and match their metadata.
- **Claim support:** at least 90% of Key Findings claims are judged supported by their cited source on expert review.
- **Labeling accuracy:** evidence-category labels agree with expert labels on at least 85% of statements.
- **Recall of key papers:** on benchmark questions, the report cites a majority of the expert-designated key papers.
- **Honesty:** when benchmark evidence is deliberately thin, the report says so rather than overstating.

These targets are initial design goals, to be revised once baseline measurements exist.

## 10. Model layer

SciForge uses the xAI API for language-model steps (question definition, evidence extraction, research gaps, hypotheses, report drafting). Query generation and source ranking are deterministic (no model). Claim checking is deterministic only (exact quotes, numeric consistency, citation/id validation); semantic (model-based) claim checking is not implemented. The API key is read from `XAI_API_KEY`; the model is configurable through `XAI_MODEL`. Model calls are isolated behind one interface so the provider can be changed without touching the rest of the system.
