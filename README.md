# SciForge

**An open-source research assistant that turns a scientific question into a traceable, evidence-based research brief.**

> Status: **SciForge V0.4 public beta (package version 0.4.0).** Deterministic PubMed/Crossref retrieval and DOI/PMID verification (the `sciforge investigate` CLI, no language model), plus an xAI model layer for evidence extraction, research gaps, stress-tested hypotheses, reports and a deterministic Evidence-to-Hypothesis Graph, available from Python and the Streamlit web app (offline Demo Mode; gated, sign-in-protected Live Mode). Hypotheses are unvalidated, AI-generated hypotheses for further investigation, not validated discoveries. See [What's next](#whats-next).

## What SciForge is

SciForge is a scientific research intelligence tool. The goal (reached step by step; see [What the current version does](#what-the-current-version-does)) is this: you give it a research question; it searches the scientific literature, extracts evidence from the sources it finds, checks that every claim traces back to a real source, and produces a structured report:

1. **Research question definition** — the question restated precisely, with scope and assumptions
2. **Literature search** — the exact queries, databases, dates, and result counts, so the search can be rerun
3. **Evidence matrix** — one row per finding: claim, source, experimental system, methods, result, limitations, confidence
4. **Citation and source traceability** — every claim linked to a DOI, PMID, or URL that was checked to exist
5. **Conflicting evidence** — where studies disagree, side by side, with likely reasons
6. **Research gaps** — what the literature does not yet answer
7. **Candidate hypotheses** — testable ideas, clearly labeled as unvalidated, AI-generated hypotheses; since v0.4
   each one is stress-tested (structured prediction, alternative explanation and falsification test, a separate
   critic model call, an optional revision, and deterministic validation that outranks both)
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

The retrieval core (introduced in v0.2, extended since) is a **deterministic literature retrieval and citation-verification engine**; the `sciforge investigate` CLI uses only this core and involves no language model. Given a research question, it:

1. **Searches** PubMed (NCBI E-utilities `esearch` + `esummary`) and Crossref (`/works`), the two current retrieval sources. The question itself is always searched, plus up to six focused queries from a deterministic, rule-based expansion (stopwords removed, a small curated concept/synonym map, e.g. "platelet activation" combined with "shear stress", "phosphatidylserine" …). No model is involved and the expansion only produces query strings; results from all queries are merged before the unchanged deduplication and verification. Disable with `--no-query-expansion` or `SCIFORGE_QUERY_EXPANSION=false`.
2. **Normalizes** each result into a record (title, authors, year, DOI, PMID, journal, source URL, retrieval time). Missing fields stay `null`; nothing is guessed.
3. **Deduplicates** conservatively: by DOI, then PMID, then only on an exact normalized title + year + first-author match.
4. **Verifies** every record: the DOI is resolved on Crossref and the PMID on PubMed, and title, year, and first author are compared. Each record is marked `verified`, `partially_verified`, or `not_verified`.
5. **Logs everything** to `runs/<UTC timestamp>/`: every request, every error, the records, and the verification details. Network and API failures are recorded and never crash the run.

The CLI itself produces no conclusions or report; the model layer (evidence, gaps, hypotheses, report, evidence graph) runs from Python and the web app ([`docs/v0.3-model-layer.md`](docs/v0.3-model-layer.md), [`docs/v0.4-hypothesis-engine.md`](docs/v0.4-hypothesis-engine.md), [`docs/web-app.md`](docs/web-app.md)). Retrieval details: [`docs/v0.2-retrieval-engine.md`](docs/v0.2-retrieval-engine.md).

| Path | Contents |
|---|---|
| `app/sciforge/` | The Python package: `cli.py`, `pipeline.py`, `pubmed.py`, `crossref.py`, `dedup.py`, `verify.py`, `normalize.py`, `models.py`, `http_utils.py`, `logging_utils.py`, `config.py` |
| `streamlit_app.py` | Web app (Streamlit): thin UI over `app/sciforge/app_service.py`; see [`docs/web-app.md`](docs/web-app.md) |
| `tests/` | Offline pytest suite (HTTP mocked with `httpx.MockTransport`; real network access is blocked) |
| `docs/v0.2-retrieval-engine.md` | v0.2 architecture, data flow, configuration, output files, dedup and verification rules, limitations |
| `docs/v0.4-hypothesis-engine.md` | v0.4 hypothesis reliability / stress-testing engine (structured hypotheses, critic, revision, deterministic validation) and the deterministic Evidence-to-Hypothesis Graph (`evidence_graph.json`, Evidence Graph tab) |
| `docs/product_spec.md`, `docs/architecture.md` | Product specification and full planned architecture |
| `agent/` | System prompt and research workflow for the planned model-based stages |
| `evaluation/` | Evaluation framework and benchmark template |
| `.env.example` | Environment-variable template (placeholders only) |
| `SECURITY.md` | How secrets are handled and how to report a vulnerability |
| `LICENSE` | MIT License |

## What's next

1. **Retrieval and citation verification (v0.2):** done.
2. **Model layer, reports and web app (v0.3):** done; semantic claim entailment is not implemented.
3. **Hypothesis stress-testing and Evidence-to-Hypothesis Graph (V0.4 public beta, current).**
4. **Next:** a benchmark/evaluation suite, a live-validated semantic claim-check stage, and full-text evidence.

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

No API key is required for the `sciforge investigate` CLI; it never reads `XAI_API_KEY`. The key is used only by the model layer (Python / web Live Mode), server-side.

`sciforge investigate` options:

| Option | Default | Meaning |
|---|---|---|
| `question` | required | Research question; always searched verbatim, plus rule-based focused queries |
| `--max-results N` | 20 | Per-database result size (1–1000): each query requests `max(N, candidate pool)` records per source; up to `2 × N` selected records are verified |
| `--from-year YYYY` / `--to-year YYYY` | none | Inclusive publication-year range (1800–2100; either may be omitted) |
| `--output-dir DIR` | `runs` | Parent directory for run folders |
| `--no-query-expansion` | expansion on | Search with the question verbatim only |
| `-v`, `--verbose` | off | Log requests and errors to stderr (secrets redacted) |

Optional tuning variables: `SCIFORGE_TIMEOUT_SECONDS` (default 20), `SCIFORGE_MAX_RETRIES` (default 2), `SCIFORGE_BACKOFF_SECONDS` (default 1), `SCIFORGE_QUERY_EXPANSION` (default true), `SCIFORGE_CANDIDATE_POOL_PER_QUERY` (default 10, 1–100), `SCIFORGE_ABSTRACT_ENRICHMENT_LIMIT` (default 50, 0–500; 0 disables abstract enrichment), `SCIFORGE_SOURCE_POLICY` (`allow_all`, `peer_reviewed_preferred` (default) or `peer_reviewed_only`; any other value is a configuration error).

Retrieval flow: expanded queries (the question verbatim + up to 6 rule-based focused queries) → a larger candidate pool per query and database → unchanged deduplication → title/provenance pre-score → **bounded abstract enrichment** (at most `SCIFORGE_ABSTRACT_ENRICHMENT_LIMIT` candidates, chosen deterministically by pre-score, best search rank, record id; batched PubMed efetch, or the Crossref `abstract` field when the search result carried one) → deterministic relevance/concept-diversity scoring from titles, retrieved abstracts (separate, lower-weighted components) and query provenance (no model) → metadata-based source-type classification and the **source policy** → unchanged DOI/PMID verification, backfilling from the ranking when a selected record fails → final set. `search_log.json` records which candidates were considered for enrichment, which got abstracts, which stayed title-only, every score component, the source type of every candidate and the final selection. Details: [`docs/v0.2-retrieval-engine.md`](docs/v0.2-retrieval-engine.md).

Source types (`source_status` on every selected source): `peer-reviewed journal article`, `preprint`, `conference paper`, `book/chapter` or `unknown`, derived **only** from bibliographic metadata (Crossref `type`/`subtype`, PubMed publication types and journal, known preprint servers by DOI prefix/container/publisher — never from title wording). "Peer-reviewed journal article" means the metadata describes a journal article; it is **not a guarantee of peer review**. Preprints are labelled PREPRINT in reports and the web app. Source policy: `allow_all` keeps the relevance order; `peer_reviewed_preferred` puts relevant journal articles first and lets preprints, conference papers, books and unknown-type records fill remaining slots; `peer_reviewed_only` sends only journal articles to verification and the model (preprints, conference papers, books/chapters and unknown are all excluded) and reports when fewer sources than requested remain.

Model-layer settings (v0.3 / web Live Mode; environment only, see `.env.example`):

| Variable | Default | Meaning |
|---|---|---|
| `SCIFORGE_MODEL_MAX_ATTEMPTS` | 15 | API attempts per investigation (retries included) |
| `SCIFORGE_MODEL_MAX_SOURCES` | 10 | Sources sent to the model (the web app caps this at its slider, default 5) |
| `SCIFORGE_MODEL_MAX_INPUT_TOKENS` | 200000 | Cumulative input-token budget per investigation |
| `SCIFORGE_MODEL_MAX_OUTPUT_TOKENS` | 2000 | Global per-call output cap (fallback for every stage) |
| `SCIFORGE_MODEL_MAX_OUTPUT_TOKENS_QUESTION` / `_EVIDENCE` / `_GAPS` / `_HYPOTHESES` / `_REPORT` | unset (= global) | Per-stage output caps, 16–128000 (recommended 2000 / 2000 / 4000 / 8000 / 4000) |
| `SCIFORGE_MODEL_MAX_OUTPUT_TOKENS_HYPOTHESIS_CRITIC` / `_HYPOTHESIS_REVISION` | unset (= global) | v0.4 critic / revision output caps (recommended 4000 / 4000) |
| `SCIFORGE_MODEL_REASONING_EFFORT` | high | `low`, `medium`, `high` or `xhigh`; used by the question stage and any stage without an override |
| `SCIFORGE_MODEL_REASONING_EFFORT_EVIDENCE` / `_GAPS` / `_HYPOTHESES` / `_REPORT` | unset (= global) | Per-stage reasoning effort (recommended medium / medium / high / medium) |
| `SCIFORGE_MODEL_REASONING_EFFORT_HYPOTHESIS_CRITIC` / `_HYPOTHESIS_REVISION` | unset (= global) | v0.4 critic / revision reasoning effort (recommended high / medium) |
| `SCIFORGE_MAX_SPEND_USD` | 15 (CLI/library); **2 in the web app's Live Mode** when unset | Spend cap per investigation (`none` disables); needs both prices. **This is the hard financial guard** |
| `SCIFORGE_MODEL_TIMEOUT_SECONDS` | 120 | Per-request xAI timeout |

The output-token settings are sent as the request's `max_output_tokens` and used for the pre-call worst-case estimate, but they are **not a total token ceiling**: xAI reports reasoning tokens separately from output tokens, so reasoning is billed on top. Requests send `reasoning: {"effort": <stage effort>}` and `store: false`; output-side tokens are charged as `output_tokens + reasoning_tokens`, and xAI's `cost_in_usd_ticks`, when supplied, is recorded as the actual cost. `SCIFORGE_MAX_SPEND_USD` is the hard financial guard. The budget records reported `output_tokens`, `reasoning_tokens` (`usage.output_tokens_details.reasoning_tokens`), `total_tokens`, the output-side tokens charged (reasoning included once, never double counted) and the xAI-reported cost when present (`reported` / `unavailable`; never fabricated). Reasoning content is never stored.

Claim checks: SciForge runs deterministic checks only — exact-quote matching, numeric/unit consistency and citation/id validation — and reports what ran with pass/fail counts. Semantic (model-based) claim checking is **not implemented**; no report or UI claims otherwise.

Invalid numeric or reasoning-effort values are configuration errors (the run refuses to start); blank per-stage values fall back to the global setting.

Exit codes: `0` run completed (any errors are recorded in the outputs), `2` invalid arguments or configuration, `3` every database search failed (outputs are still written).

Each run writes `runs/<UTC timestamp>/` (for example `runs/20260928T234100Z/`), which is gitignored:

| File | Contents |
|---|---|
| `search_log.json` | Every request (database, query, parameters with secrets redacted, timestamp, HTTP status, result count) and every error; the query plan with per-query candidate counts, dedup results, abstract enrichment (considered / with abstract / title-only; lengths only, no abstract text), per-candidate selection scores (title and abstract components), source types and source-policy decisions |
| `sources.json` | Deduplicated records sent to verification, with provenance, any field conflicts and `source_status` (with its metadata basis) |
| `verification.json` | Per-record verification status with per-identifier, per-field comparison details |
| `summary.json` | Question, queries used per database, version, start/end times, candidate counts per source and per query, duplicates merged, selection (target, backfill, selected ids and scores), verification counts, errors, failed databases |

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
- **Sign-in for Live Mode (public deployments).** `SCIFORGE_LIVE_REQUIRE_AUTH` defaults to true (only the exact
  value `false` disables it). Then Live Mode additionally requires Streamlit's native OIDC login
  (`st.login()`, configured by an `[auth]` section in `.streamlit/secrets.toml`; requires the `Authlib`
  package, included in the `web` extra) **and** an email on `SCIFORGE_LIVE_ALLOWED_EMAILS` (comma-separated,
  from environment or secrets; case-insensitive exact match). An empty allowlist means nobody may use Live
  Mode; missing/incomplete `[auth]` or a missing Authlib keeps Live Mode unavailable (fail closed). Demo Mode
  never requires sign-in. The service layer re-checks the decision before any literature lookup or model client.
- **Usage limit, kill switch and web spend default.** With sign-in required, each allowlisted user may start at
  most `SCIFORGE_LIVE_MAX_RUNS_PER_USER` (default 3) Live runs per rolling 24 hours, counted server-side in a
  small file outside the repository (`SCIFORGE_LIVE_QUOTA_PATH`, default
  `~/.local/state/sciforge/live_quota.json`) that stores only a hash of the verified email
  (HMAC-SHA256 with `SCIFORGE_LIVE_QUOTA_SALT` if set). A run counts when it starts, so failed runs count.
  `SCIFORGE_LIVE_QUOTA_BACKEND=postgres` stores the same hashes in a shared PostgreSQL table instead (Streamlit
  SQL connection `[connections.sciforge_quota]`, schema `deploy/sql/001_live_quota.sql`, never auto-created;
  any database problem refuses Live runs — fail closed).
  `SCIFORGE_LIVE_KILL_SWITCH` (default off; unset/empty/`false` = off, any other value = on) switches Live Mode
  off regardless of every other setting. In the web app, Live runs use a $2 spend cap when
  `SCIFORGE_MAX_SPEND_USD` is unset.
- **Labels.** Demo Mode is a synthetic, offline demonstration (no real literature, no xAI). Every source shows its
  source type first; preprints are labelled "Preprint — not peer-reviewed". Validation is deterministic only
  (exact quote, numeric and citation checks); semantic claim entailment is not implemented. Hypotheses are
  unvalidated, AI-generated hypotheses for further investigation, not validated discoveries. The v0.4
  hypothesis engine (generation → critic → revision → deterministic validation) is documented in
  [`docs/v0.4-hypothesis-engine.md`](docs/v0.4-hypothesis-engine.md).
- **Evidence Graph (v0.4).** Every model investigation (Demo and Live) writes a deterministic
  `evidence_graph.json` (source → evidence → gap ← hypothesis ← prediction / falsification test, built by code
  from validated outputs only, no model call) and the web app shows it on the Evidence Graph tab with a node
  selector for details.

Architecture, privacy rules, secret configuration and Streamlit Community Cloud deployment:
[`docs/web-app.md`](docs/web-app.md).

## Limitations to keep in mind

- v0.2 retrieves and verifies bibliographic records. It reads abstracts only for ranking (bounded enrichment, at most `SCIFORGE_ABSTRACT_ENRICHMENT_LIMIT` candidates); it does not read full text, extract evidence, or judge whether a paper supports any claim.
- Citation verification confirms that a DOI/PMID resolves and that its title, year, and first author match; it cannot confirm that a paper says what anyone claims it says. Records found only in Crossref are verified against Crossref itself.
- The question is sent verbatim plus a few rule-based focused queries; the concept map is small and hand-curated (currently platelet/shear/lipid-oriented), so for other topics results still depend heavily on wording.
- Source selection scores candidates from their titles, abstracts retrieved by bounded enrichment (candidates beyond the limit, or without an available abstract, are ranked on title only) and query provenance; it is a transparent heuristic, not a relevance judgment, and can miss relevant papers.
- Source types come from bibliographic metadata and can be missing or wrong; "peer-reviewed journal article" is not a guarantee of peer review.
- Later versions will only read what is openly accessible; for paywalled papers they will often have only the abstract, and will label findings accordingly. Reports will be aids to expert judgment, not substitutes for it.

## License

MIT — see [LICENSE](LICENSE).
