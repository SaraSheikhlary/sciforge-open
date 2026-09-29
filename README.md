# SciForge

**An open-source research assistant that turns a scientific question into a traceable, evidence-based research brief.**

> Status: **v0.2 — deterministic literature retrieval and citation-verification engine.** The command-line tool searches PubMed and Crossref, deduplicates the records, and checks every DOI/PMID against the issuing database. It does not yet extract evidence, draw conclusions, or call a language model; the xAI model layer and report generation are planned for later versions. See [What's next](#whats-next).

## What SciForge is

SciForge is a scientific research intelligence tool. The goal (reached step by step; see [What the current version does](#what-the-current-version-does)) is this: you give it a research question; it searches the scientific literature, extracts evidence from the sources it finds, checks that every claim traces back to a real source, and produces a structured report:

1. **Research question definition** — the question restated precisely, with scope and assumptions
2. **Literature search** — the exact queries, databases, dates, and result counts, so the search can be rerun
3. **Evidence matrix** — one row per finding: claim, source, experimental system, methods, result, limitations, confidence
4. **Citation and source traceability** — every claim linked to a DOI, PMID, or URL that was checked to exist
5. **Conflicting evidence** — where studies disagree, side by side, with likely reasons
6. **Research gaps** — what the literature does not yet answer
7. **Candidate hypotheses** — testable ideas, clearly labeled as hypotheses
8. **Proposed next steps** — concrete literature, computational, or experimental steps

## What problem it solves

Large language models can summarize science fluently, but they also invent citations, blur the line between what a paper showed and what the model guessed, and give answers no one can audit. Manual literature review avoids those problems but is slow.

SciForge is built around one rule: **no claim without a verifiable source.** Every finding is tied to a specific paper, every citation is checked against a bibliographic database before it appears in a report, and every statement is labeled as one of:

- **Established evidence** — directly supported by cited sources
- **Conflicting evidence** — cited sources disagree
- **Inference** — a reasoned conclusion drawn from the evidence, not stated in any source
- **Hypothesis** — a testable proposal not yet supported by direct evidence

When the evidence is thin, SciForge says so instead of filling the gap.

## What the current version does

v0.2 is a **deterministic literature retrieval and citation-verification engine**. No language model is involved. Given a research question, it:

1. **Searches** PubMed (NCBI E-utilities `esearch` + `esummary`) and Crossref (`/works`), the two current retrieval sources. The question itself is always searched, plus up to six focused queries from a deterministic, rule-based expansion (stopwords removed, a small curated concept/synonym map, e.g. "platelet activation" combined with "shear stress", "phosphatidylserine" …). No model is involved and the expansion only produces query strings; results from all queries are merged before the unchanged deduplication and verification. Disable with `--no-query-expansion` or `SCIFORGE_QUERY_EXPANSION=false`.
2. **Normalizes** each result into a record (title, authors, year, DOI, PMID, journal, source URL, retrieval time). Missing fields stay `null`; nothing is guessed.
3. **Deduplicates** conservatively: by DOI, then PMID, then only on an exact normalized title + year + first-author match.
4. **Verifies** every record: the DOI is resolved on Crossref and the PMID on PubMed, and title, year, and first author are compared. Each record is marked `verified`, `partially_verified`, or `not_verified`.
5. **Logs everything** to `runs/<UTC timestamp>/`: every request, every error, the records, and the verification details. Network and API failures are recorded and never crash the run.

It produces no conclusions, evidence matrix, or report yet. Full details: [`docs/v0.2-retrieval-engine.md`](docs/v0.2-retrieval-engine.md).

| Path | Contents |
|---|---|
| `app/sciforge/` | The Python package: `cli.py`, `pipeline.py`, `pubmed.py`, `crossref.py`, `dedup.py`, `verify.py`, `normalize.py`, `models.py`, `http_utils.py`, `logging_utils.py`, `config.py` |
| `streamlit_app.py` | Web app (Streamlit): thin UI over `app/sciforge/app_service.py`; see [`docs/web-app.md`](docs/web-app.md) |
| `tests/` | Offline pytest suite (HTTP mocked with `httpx.MockTransport`; real network access is blocked) |
| `docs/v0.2-retrieval-engine.md` | v0.2 architecture, data flow, configuration, output files, dedup and verification rules, limitations |
| `docs/product_spec.md`, `docs/architecture.md` | Product specification and full planned architecture |
| `agent/` | System prompt and research workflow for the planned model-based stages |
| `evaluation/` | Evaluation framework and benchmark template |
| `.env.example` | Environment-variable template (placeholders only) |
| `SECURITY.md` | How secrets are handled and how to report a vulnerability |
| `LICENSE` | MIT License |

## What's next

1. **Retrieval and citation verification (v0.2, current):** command-line tool that searches PubMed and Crossref and verifies DOIs/PMIDs.
2. **Evidence extraction and reports (v0.3):** the xAI model layer, model-based query generation and evidence extraction, claim-to-source checks, and Markdown reports.
3. **Evaluation and interface (v0.4+):** a benchmark suite, then a simple web interface.

## How to run it

Requires Python 3.10 or newer.

```bash
git clone https://github.com/SaraSheikhlary/sciforge-open.git
cd sciforge-open
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # package + pytest

# Configuration comes from environment variables only (see .env.example):
export SCIFORGE_CONTACT_EMAIL="you@example.org"   # recommended: sent to NCBI and Crossref for polite API access
export NCBI_API_KEY="your-key-here"               # optional: raises the PubMed rate limit from 3 to 10 requests/s

sciforge investigate "platelet activation under shear stress"
sciforge investigate "liposome drug delivery" --max-results 10 --from-year 2020 --to-year 2024 --output-dir runs/
python -m sciforge --help      # same CLI without the console script
```

No API key is required for v0.2. `XAI_API_KEY` is reserved for the model layer in a later version and is not read by v0.2.

`sciforge investigate` options:

| Option | Default | Meaning |
|---|---|---|
| `question` | required | Research question; always searched verbatim, plus rule-based focused queries |
| `--max-results N` | 20 | Maximum records per source (1–1000) |
| `--from-year YYYY` / `--to-year YYYY` | none | Inclusive publication-year range (1800–2100; either may be omitted) |
| `--output-dir DIR` | `runs` | Parent directory for run folders |
| `--no-query-expansion` | expansion on | Search with the question verbatim only |
| `-v`, `--verbose` | off | Log requests and errors to stderr (secrets redacted) |

Optional tuning variables: `SCIFORGE_TIMEOUT_SECONDS` (default 20), `SCIFORGE_MAX_RETRIES` (default 2), `SCIFORGE_BACKOFF_SECONDS` (default 1), `SCIFORGE_QUERY_EXPANSION` (default true).

Exit codes: `0` run completed (any errors are recorded in the outputs), `2` invalid arguments or configuration, `3` every database search failed (outputs are still written).

Each run writes `runs/<UTC timestamp>/` (for example `runs/20260928T234100Z/`), which is gitignored:

| File | Contents |
|---|---|
| `search_log.json` | Every request (database, query, parameters with secrets redacted, timestamp, HTTP status, result count) and every error |
| `sources.json` | Deduplicated records with provenance and any field conflicts |
| `verification.json` | Per-record verification status with per-identifier, per-field comparison details |
| `summary.json` | Question, version, start/end times, counts per source, duplicates merged, verification counts, errors, failed databases |

Run the tests (offline; no network access needed or allowed):

```bash
pytest -q
```

Never commit an API key. See [SECURITY.md](SECURITY.md).

## Web app (demo)

A Streamlit web interface ("SciForge — AI for Scientific Discovery") runs the v0.3 pipeline behind a
simple form: research question, optional year range, maximum sources, and Demo/Live mode, with progress
steps and tabs for Overview, Evidence, Conflicts, Research Gaps, Hypotheses, Sources and Validation.

```bash
pip install -e ".[web]"
streamlit run streamlit_app.py
```

- **Demo Mode (default)** works offline with no API key. It runs one bundled example investigation on
  clearly labelled **synthetic** records (fake `10.0000/demo.*` DOIs, `[SYNTHETIC DEMO]` titles) with a
  scripted fake model. Nothing it shows is a real finding.
- **Live Mode** is off by default. It is enabled only when the deployment gate
  `SCIFORGE_LIVE_ENABLED=true` is set **and** both `XAI_API_KEY` and `XAI_MODEL` are configured
  (environment variables, or Streamlit secrets as a fallback; environment wins). Any other gate value keeps
  it disabled, and the service refuses gated-off live requests itself. It uses the existing xAI client and
  budgets. **Live
  Mode is untested with a real key and has not been validated live.**

Architecture, privacy rules, secret configuration and Streamlit Community Cloud deployment:
[`docs/web-app.md`](docs/web-app.md).

## Limitations to keep in mind

- v0.2 retrieves and verifies bibliographic records only. It does not read abstracts or full text, extract evidence, or judge whether a paper supports any claim.
- Citation verification confirms that a DOI/PMID resolves and that its title, year, and first author match; it cannot confirm that a paper says what anyone claims it says. Records found only in Crossref are verified against Crossref itself.
- The question is sent verbatim plus a few rule-based focused queries; the concept map is small and hand-curated (currently platelet/shear/lipid-oriented), so for other topics results still depend heavily on wording.
- Later versions will only read what is openly accessible; for paywalled papers they will often have only the abstract, and will label findings accordingly. Reports will be aids to expert judgment, not substitutes for it.

## License

MIT — see [LICENSE](LICENSE).
