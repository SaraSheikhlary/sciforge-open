# Security Policy

SciForge is a public repository. Treat everything committed here as visible to anyone.

## Secrets

SciForge never stores secrets in the repository. All credentials are read from environment variables at runtime.

v0.2 (the current version) is a deterministic retrieval and citation-verification engine and needs **no API key**.

| Variable | Used by v0.2 | Required | Purpose |
|---|---|---|---|
| `NCBI_API_KEY` | Yes | No (optional) | Raises the NCBI E-utilities (PubMed) rate limit from 3 to 10 requests/s |
| `SCIFORGE_CONTACT_EMAIL` | Yes | No (recommended) | Contact address for polite API access: sent to NCBI as `email` and to Crossref as `mailto` and in the User-Agent |
| `SCIFORGE_TIMEOUT_SECONDS`, `SCIFORGE_MAX_RETRIES`, `SCIFORGE_BACKOFF_SECONDS` | Yes | No | Request timeout and retry tuning (not secrets) |
| `XAI_API_KEY` | No | Planned (later version) | Access to the xAI API for the planned model layer; reserved, not read by v0.2 |
| `XAI_MODEL` | No | Planned (later version) | Model name override for the planned model layer |

How v0.2 protects these values:

- They are read only from the environment; SciForge does not read `.env` files itself (`.env.example` contains placeholders only).
- `api_key`, `email`, and `mailto` values are replaced with `[REDACTED]` in logged request parameters, URLs, error messages, and log lines, and every run output file is scrubbed for the literal values before it is written.
- The settings object's text representation omits the key and email.
- Run outputs are written to `runs/`, which is gitignored.

Rules for contributors:

- Never commit an API key, token, password, or `.env` file. `.env` and similar files are listed in `.gitignore`.
- Never paste a key into an issue, pull request, discussion, or log output.
- Code must read secrets only from the environment and must never print, log, or include them in reports or error messages.
- Example configuration uses obvious placeholders such as `your-key-here`.
- If a key is ever committed, treat it as compromised: revoke it with the provider immediately, then remove it from history. Removing the commit alone is not enough.

## Data handling

- SciForge queries public bibliographic services (in v0.2: PubMed and Crossref). Research questions are sent to those services verbatim as search queries; from the planned model layer onward they will also be sent to the model provider. Do not enter confidential or unpublished information into SciForge.
- Generated reports are saved locally and are not uploaded anywhere by SciForge.

## Supported versions

SciForge is pre-release (v0.x). Only the latest commit on `main` is supported.

## Reporting a vulnerability

Please do not open a public issue for a security problem. Use GitHub's private vulnerability reporting for this repository (the **Security** tab, then **Report a vulnerability**). Include a description, steps to reproduce, and the potential impact. You can expect an acknowledgement within 7 days.
