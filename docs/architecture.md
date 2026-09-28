# SciForge v0.1 Architecture

This document describes the planned architecture. No component is implemented yet.

## 1. Design principles

1. **Separation of concerns.** Six components with narrow interfaces, testable in isolation.
2. **Deterministic where possible.** Retrieval and citation verification are plain code, not model calls. The language model is used only where judgment or language is needed.
3. **Everything is logged.** Queries, retrieved records, model prompts and responses, and verification results are saved per investigation.
4. **No fabrication path.** A citation can only enter a report if it came from a retrieval result and passed verification. The model cannot introduce a source on its own.
5. **Secrets from the environment only.**

## 2. Components

```
 ┌─────────────────────────────────────────────────────────────┐
 │ 1. User Interface          CLI (v0.2) → web UI (later)      │
 └──────────────────────────────┬──────────────────────────────┘
                                │ InvestigationRequest
 ┌──────────────────────────────▼──────────────────────────────┐
 │ 2. Agent Orchestration     runs the workflow, holds state   │
 └───┬──────────────┬──────────────┬──────────────┬────────────┘
     │              │              │              │
 ┌───▼────────┐ ┌───▼─────────┐ ┌──▼──────────┐ ┌─▼───────────┐
 │3. Literature│ │4. Evidence  │ │5. Scientific│ │6. Report    │
 │  Retrieval  │ │  Extraction │ │  Evaluation │ │  Generation │
 └───┬────────┘ └───┬─────────┘ └──┬──────────┘ └─────────────┘
     │              │              │
     ▼              ▼              ▼
 PubMed,        Model layer     Crossref / PubMed
 Crossref, ...  (xAI API)       + model layer
```

### 2.1 User Interface
- **v0.2:** command-line tool, `sciforge investigate "<question>" [--from YEAR] [--to YEAR] [--target TEXT] [--seed DOI_OR_PMID]`.
- **Later:** a minimal web interface that shows the report with each claim linked to its evidence record.
- Contains no research logic; it builds an `InvestigationRequest` and displays results.

### 2.2 Agent Orchestration
- Runs the workflow in `agent/research_workflow.md` as an explicit sequence of stages: define → plan search → retrieve → screen → extract → evaluate → report.
- Holds investigation state and writes it to disk after each stage, so a failed run can resume.
- Enforces limits (maximum sources, maximum model calls, timeouts) and records failures in the report rather than hiding them.

### 2.3 Literature Retrieval
- Plain HTTP clients for public bibliographic APIs. v0.2 targets PubMed (NCBI E-utilities) and Crossref; candidates for later versions include Europe PMC, OpenAlex, arXiv, and bioRxiv/medRxiv.
- Retrieves open-access full text only where it is legitimately available; otherwise uses abstracts and records `access_level`.
- Logs every query with database, timestamp, parameters, and result count.
- Respects each service's rate limits and usage policy.
- Returns normalized `SourceRecord` objects (identifiers, title, authors, year, venue, abstract, full-text availability).

### 2.4 Evidence Extraction
- Uses the model layer to extract `EvidenceRecord`s from each source's text, following the schema in `agent/research_workflow.md`.
- Every extracted record keeps a pointer to the exact source and, where possible, the passage it came from.
- Fields that cannot be confirmed from the text are set to `null` or `"not verified"`.

### 2.5 Scientific Evaluation
- **Citation verification (deterministic):** resolves each DOI or PMID against Crossref or PubMed and compares title, authors, and year.
- **Claim-to-source check (model-assisted):** asks the model whether the cited passage supports the claim; records supported / partially supported / unsupported / not verified.
- **Evidence labeling:** assigns established / conflicting / inference / hypothesis, and groups conflicting records.
- Produces a validation summary: sources checked, claims verified, claims flagged.

### 2.6 Report Generation
- Assembles the ten-section report (see `docs/product_spec.md`) from verified evidence records.
- Renders Markdown; the evidence matrix and sources are generated from structured data, not free-written by the model.
- Includes run metadata: date, model name, databases queried, and anything that failed or could not be accessed.

## 3. Model layer

- A single `ModelClient` interface wraps the xAI API. All components call the model only through it.
- Configuration from the environment: `XAI_API_KEY` (required), `XAI_MODEL` (optional override).
- Structured outputs are requested as JSON and validated against schemas; invalid responses are retried a bounded number of times, then recorded as failures.
- Prompts and responses are logged per investigation, with the API key never logged.

## 4. Data model (summary)

- `InvestigationRequest`: question, date range, targets, seed identifiers
- `SourceRecord`: normalized bibliographic record plus access level
- `EvidenceRecord`: schema in `agent/research_workflow.md`
- `ValidationResult`: per-source and per-claim verification outcomes
- `Report`: the ten sections plus run metadata

## 5. Per-investigation output

```
runs/<timestamp>_<slug>/
├── request.json
├── search_log.md
├── sources.json
├── evidence.json
├── validation.json
├── model_log.jsonl
└── report.md
```

`runs/` is excluded from version control by default.

## 6. Proposed repository layout

```
sciforge-open/
├── README.md
├── LICENSE
├── SECURITY.md
├── docs/            product spec, architecture
├── agent/           system prompt, research workflow
├── evaluation/      evaluation framework, benchmarks
├── app/             application code (from v0.2)
├── tools/           retrieval and verification clients (from v0.2)
└── examples/        example reports built only from public literature (later)
```

## 7. Technology choices (proposed)

- Python 3.11+
- `httpx` for HTTP, `pydantic` for schemas, `typer` for the CLI, `pytest` for tests
- OpenAI-compatible client pointed at the xAI API base URL, or direct HTTP calls
