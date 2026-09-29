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

- SciForge queries public bibliographic services (in v0.2: PubMed and Crossref). Research questions are sent to those services verbatim as search queries (plus rule-based focused queries built from their keywords); from the planned model layer onward they will also be sent to the model provider. Do not enter confidential or unpublished information into SciForge.
- Public web deployments: Live Mode requires Streamlit OIDC sign-in and an email on `SCIFORGE_LIVE_ALLOWED_EMAILS` unless `SCIFORGE_LIVE_REQUIRE_AUTH` is exactly `false` (fail closed: missing `[auth]` configuration, missing Authlib, or an empty allowlist keep Live Mode unavailable). Only the access decision is logged, never emails or identity-provider tokens. The `[auth]` `client_secret` and `cookie_secret` belong in `.streamlit/secrets.toml` / the host's secrets store only, like `XAI_API_KEY`.
- Live usage limit: the quota store (`SCIFORGE_LIVE_QUOTA_PATH`, outside the repository, mode 0600) holds only a keyed hash of the verified email (HMAC-SHA256 with the secret `SCIFORGE_LIVE_QUOTA_SALT`; plain SHA-256 if no salt is set, which is guessable from candidate emails) and run timestamps — never the plaintext email. Keep the salt in the secrets store. Unreadable/unwritable store = Live Mode refused (fail closed). `SCIFORGE_LIVE_KILL_SWITCH` (any value other than unset/empty/`false`) disables Live Mode immediately.
- Postgres quota backend (`SCIFORGE_LIVE_QUOTA_BACKEND=postgres`): database credentials belong only in the `[connections.<name>]` secrets section (read by Streamlit's `st.connection`; the SciForge service never receives them). Rows contain only the identity hash and a timestamp — never emails, API keys, OIDC tokens, prompts, results or reasoning. Use TLS (`sslmode=require`) and a least-privilege role (`SELECT, INSERT, DELETE` on `sciforge_live_quota_events`). Database errors are logged by exception type only and refuse Live runs (fail closed).
- Generated reports are saved locally and are not uploaded anywhere by SciForge.

## Supported versions

SciForge is pre-release (v0.x). Only the latest commit on `main` is supported.

## Reporting a vulnerability

Please do not open a public issue for a security problem. Use GitHub's private vulnerability reporting for this repository (the **Security** tab, then **Report a vulnerability**). Include a description, steps to reproduce, and the potential impact. You can expect an acknowledgement within 7 days.
