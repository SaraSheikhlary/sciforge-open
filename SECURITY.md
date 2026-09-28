# Security Policy

SciForge is a public repository. Treat everything committed here as visible to anyone.

## Secrets

SciForge never stores secrets in the repository. All credentials are read from environment variables at runtime.

| Variable | Required | Purpose |
|---|---|---|
| `XAI_API_KEY` | Yes (from v0.2) | Access to the xAI API, the model layer |
| `XAI_MODEL` | No | Model name override; a documented default is used otherwise |
| `NCBI_API_KEY` | No | Higher rate limits for NCBI E-utilities (PubMed) |
| `SCIFORGE_CONTACT_EMAIL` | Recommended | Contact address sent to Crossref and NCBI as their usage policies request |

Rules for contributors:

- Never commit an API key, token, password, or `.env` file. `.env` and similar files are listed in `.gitignore`.
- Never paste a key into an issue, pull request, discussion, or log output.
- Code must read secrets only from the environment and must never print, log, or include them in reports or error messages.
- Example configuration uses obvious placeholders such as `your-key-here`.
- If a key is ever committed, treat it as compromised: revoke it with the provider immediately, then remove it from history. Removing the commit alone is not enough.

## Data handling

- SciForge queries public bibliographic services (for example PubMed, Crossref). Research questions are sent to those services and to the model provider. Do not enter confidential or unpublished information into SciForge.
- Generated reports are saved locally and are not uploaded anywhere by SciForge.

## Supported versions

SciForge is pre-release (v0.x). Only the latest commit on `main` is supported.

## Reporting a vulnerability

Please do not open a public issue for a security problem. Use GitHub's private vulnerability reporting for this repository (the **Security** tab, then **Report a vulnerability**). Include a description, steps to reproduce, and the potential impact. You can expect an acknowledgement within 7 days.
