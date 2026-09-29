# SciForge web app (Milestone 1)

A Streamlit interface for SciForge: **SciForge — AI for Scientific Discovery**.

> **Status.** Demo Mode is complete and fully tested offline. **Live Mode is untested with a real xAI
> API key and has not been validated live.** It reuses the existing, unit-tested `XAIClient` and budget
> code, but no end-to-end run against the real API has been made. Do not rely on Live Mode output yet.

## Architecture

```
streamlit_app.py            thin UI: widgets, progress display, tabs (no scientific logic)
        │ calls
        ▼
app/sciforge/app_service.py application interface: input validation, mode selection, credential presence
        │                   check, temp run directory, progress events, result shaping, display privacy guard
        ├── Demo Mode ─► app/sciforge/demo_data.py  (SYNTHETIC records + scripted FakeModelClient + offline
        │                                            MockTransport serving the synthetic abstracts)
        └── Live Mode ─► sciforge.pipeline.run_investigation (v0.2: PubMed + Crossref, dedup, verification)
                         + sciforge.llm.xai.XAIClient with ModelSettings budgets
        ▼
sciforge.investigation_pipeline.run_model_investigation   (existing v0.3 pipeline, unchanged)
   source texts → question definition → evidence extraction + deterministic validation →
   research gaps → hypotheses → report (narrative + deterministic template, citations by code)
```

`run_web_investigation(request, progress=callback)` takes an `InvestigationRequest` (question, optional
start/end year, maximum sources, mode `demo`/`live`) and returns a `WebInvestigationResult`:

| Field | Contents |
|---|---|
| `status` | `ok`, `degraded` (no validated evidence), `budget_exhausted`, `model_auth_error`, `error`, `invalid_input`, `live_unavailable` (Live Mode requested while gated off or without credentials) |
| `sections` | The report sections A–J exactly as rendered by `sciforge.stages.report.build_report` |
| `evidence`, `conflicts` | Accepted evidence items (claim, verbatim quote, category, confidence, `[S#]` source ref) |
| `gaps`, `hypotheses` | Accepted research gaps (labelled inference) and candidate hypotheses (labelled hypothesis) |
| `sources` | Citation lines from `render_citation` over the v0.2 records (`[S#]` cited, `R#` retrieved but not cited); unresolvable ids as `[UNRESOLVED CITATION: id]` |
| `validation` | Report validation status, citation counts, issues, evidence/gap/hypothesis/paragraph accept/reject counts with reason codes, budget usage, privacy-guard summary |
| `limitations`, `notices`, `errors` | Human-readable text |

The UI shows eight progress steps (Defining question, Searching literature, Verifying sources, Extracting
evidence, Checking evidence, Identifying research gaps, Generating hypotheses, Building report) and seven
result tabs (Overview, Evidence, Conflicts, Research Gaps, Hypotheses, Sources, Validation). Progress for
model stages is taken from the stage of each model request; evidence checking is the deterministic
validation that runs inside extraction.

Input limits: question 10–2000 characters; years 1800–2100 with start ≤ end (both optional); maximum
sources 1–20. In Live Mode the maximum sources is used as the v0.2 results-per-database limit and as the
model source limit (never above `SCIFORGE_MODEL_MAX_SOURCES`).

### Citations

Every citation shown in the UI comes from the deterministic renderer
(`sciforge.stages.report.render_citation` / `unresolved_marker`) applied to v0.2 records. The model only
ever sees opaque record ids; it cannot create or edit citations.

### Privacy

- Questions are not persisted. The pipelines write their normal run files to a fresh directory under the
  system temp directory (prefix `sciforge-web-`), never the repository, and that directory is deleted
  before the result is returned. The optional raw-rejected-output debug file is always disabled.
- Every displayed string passes `guard_display`: secret redaction (`sciforge.logging_utils.redact_text`
  with the configured key values), key-shaped tokens, filesystem paths, and any raw rejected model output
  (needles from `sciforge.output_guard.rejected_text_values`). The run directory is also checked with
  `sciforge.output_guard.find_leaks` before deletion; the count is shown on the Validation tab.
- Rejected model items are shown only as reason codes and counts.
- API keys and environment values are never displayed. The UI only reports *whether* Live Mode is
  enabled for the deployment and *whether* `XAI_API_KEY` and `XAI_MODEL` are configured.
- Unexpected errors are shown as a generic message with the exception type only; `.streamlit/config.toml`
  sets `client.showErrorDetails = "none"` so tracebacks and paths never reach the browser, and
  `browser.gatherUsageStats = false`.
- Streamlit keeps the latest result in the in-memory session state of the browser session only.

## Local startup

```bash
git clone https://github.com/SaraSheikhlary/sciforge-open.git
cd sciforge-open
python -m venv .venv && source .venv/bin/activate
pip install -e ".[web]"          # package + streamlit (add ,dev for pytest: ".[dev,web]")
streamlit run streamlit_app.py   # opens http://localhost:8501
```

`streamlit_app.py` also works from a plain checkout without installing the package (it adds `app/` to
the import path), as long as the dependencies are installed (`pip install -r requirements.txt`).

Tests (offline; network access is blocked by the test suite): `pytest -q` (UI tests use
`streamlit.testing.v1.AppTest`).

## Demo Mode (default)

Demo Mode needs no API key and no network. It runs the real v0.3 pipeline on a bundled, clearly labelled
**synthetic** example investigation (topic: study preregistration and reported effect sizes — the
metascience of the published literature):

- five synthetic v0.2-shaped records with DOIs `10.0000/demo.*` (an unassigned prefix), titles starting
  with `[SYNTHETIC DEMO]`, authors `Demo Author …`, journal "SciForge Synthetic Demo Journal (not a real
  journal)", and invented abstracts starting with `SYNTHETIC DEMO ABSTRACT`. Their verification statuses
  (three verified, one partially verified, one not verified) are assigned by the demo data, not checked;
- a scripted `FakeModelClient` that answers each stage from the ids in the request, so the deterministic
  checks run for real. One evidence item deliberately quotes text that is not in the abstract, so the
  Validation tab shows one `quote_not_in_source` rejection (its text is never displayed);
- abstracts are served through an `httpx.MockTransport` in Crossref `/works/{doi}` shape (no socket is
  opened).

Demo Mode always analyses the same dataset; the entered question is echoed in the question definition but
not searched. The year filter and maximum sources are applied to the synthetic records. The UI labels all
demo output as synthetic; none of it is a real finding.

## Live Mode (experimental, not validated live)

### Deployment gate

Live Mode needs an explicit opt-in per deployment:

| Condition | Live Mode |
|---|---|
| `SCIFORGE_LIVE_ENABLED` unset (default), empty, or any value other than `true` | **disabled** — "Live Mode is disabled for this deployment", whatever credentials exist |
| `SCIFORGE_LIVE_ENABLED=true` but `XAI_API_KEY` or `XAI_MODEL` missing/empty | **unavailable** — the missing names are listed (never values) |
| `SCIFORGE_LIVE_ENABLED=true` and both `XAI_API_KEY` and `XAI_MODEL` non-empty | enabled (experimental) |

Only the exact value `true` enables the gate (case-insensitive, surrounding whitespace ignored); `1`,
`yes`, `on`, and unquoted TOML booleans in `secrets.toml` all count as false. The gate is enforced twice:
the UI disables the Live option, and `run_web_investigation` itself refuses a live request
(`status = "live_unavailable"`) before creating a temp directory, calling PubMed/Crossref, or building
an `XAIClient`. The live runner checks again. Demo Mode ignores the gate and credentials completely.

### What Live Mode does

Live Mode is disabled in the UI unless the gate is on **and** both `XAI_API_KEY` and `XAI_MODEL` are present. When enabled,
it runs v0.2 retrieval and verification (PubMed + Crossref) and then the v0.3 pipeline with the existing
`XAIClient` (Responses API, `store=false`). All existing limits apply unchanged, read by
`ModelSettings.from_env`: `SCIFORGE_MODEL_MAX_ATTEMPTS`, `SCIFORGE_MODEL_MAX_SOURCES`,
`SCIFORGE_MODEL_MAX_INPUT_TOKENS`, `SCIFORGE_MODEL_MAX_OUTPUT_TOKENS`, `SCIFORGE_MAX_SPEND_USD` (default
15 USD; while the cap is on, `SCIFORGE_PRICE_INPUT_PER_MTOK` and `SCIFORGE_PRICE_OUTPUT_PER_MTOK` must be
set or Live Mode refuses to start — set `SCIFORGE_MAX_SPEND_USD=none` to disable the cap explicitly). See
`.env.example` for all variables. A budget stop or an authentication failure is reported as a status; the
remaining report sections are built by code only.

Live Mode has only been exercised in offline tests (mock HTTP transport and fake model). It has **not**
been run with a real key.

## Secret configuration

Order of precedence for `SCIFORGE_LIVE_ENABLED`, `XAI_API_KEY` and `XAI_MODEL`:

1. **Environment variables** (a non-empty value always wins; e.g. `SCIFORGE_LIVE_ENABLED=false` in the
   environment disables Live Mode even if secrets say `"true"`).
2. **Streamlit secrets** (`st.secrets`): `.streamlit/secrets.toml` locally, or the app's *Secrets*
   setting on Streamlit Community Cloud. Only these three names are read from `st.secrets`. Write the gate
   as a quoted string: `SCIFORGE_LIVE_ENABLED = "true"`.

All other settings (budgets, prices, spend cap, `NCBI_API_KEY`, `SCIFORGE_CONTACT_EMAIL`) are read from
environment variables only.

Local secrets file: copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` (gitignored) and
fill in real values. Never commit it. The key is used only server-side in the `Authorization` header of
requests to the xAI API; it is never sent to the browser, logged, or displayed.

## Streamlit Community Cloud deployment

1. Push the repository to GitHub (the app needs `streamlit_app.py`, `requirements.txt`, `pyproject.toml`,
   `app/` and `.streamlit/config.toml`; never `.streamlit/secrets.toml`).
2. On <https://share.streamlit.io>, create an app from the repository, branch, and main file
   `streamlit_app.py`; choose Python 3.10 or newer.
3. `requirements.txt` installs the package from `app/` with the `web` extra (`.[web]`).
4. Demo Mode works with no further setup; Live Mode stays disabled by default (no `SCIFORGE_LIVE_ENABLED`).
5. Live Mode (optional, not yet validated live): in *App settings → Secrets* add
   `SCIFORGE_LIVE_ENABLED = "true"`, `XAI_API_KEY` and `XAI_MODEL`, plus either both price variables or
   `SCIFORGE_MAX_SPEND_USD = "none"`. Leave the gate out (or `"false"`) for a public demo-only deployment. Community Cloud
   also exports root-level secrets as environment variables, which is how the non-credential settings are
   read. Consider restricting app access, because every Live Mode run spends from your xAI account.

## Limitations

- Live Mode is untested with a real xAI key; not validated live. It is off by default
  (`SCIFORGE_LIVE_ENABLED` must be `true`).
- Evidence comes from abstracts only; claim support is checked deterministically (exact quotes, numbers,
  ids), not for scientific quality.
- Demo Mode ignores the content of the entered question (fixed synthetic dataset).
- Progress for "Searching literature" and "Verifying sources" is reported when the v0.2 engine finishes
  (both happen inside one v0.2 call).
- No authentication, rate limiting or per-user quotas in the app itself.
- When the server starts, Streamlit itself may look up the machine's external IP address to print URLs
  in the console; this is Streamlit behaviour, unrelated to SciForge or xAI.
