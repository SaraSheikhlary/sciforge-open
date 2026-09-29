-- SciForge Live Mode usage limit (SCIFORGE_LIVE_QUOTA_BACKEND=postgres).
-- Idempotent; run once per database BEFORE enabling the postgres backend. The app never creates or alters
-- tables: if this schema is missing, Live Mode is refused (fail closed).
--
-- Stores ONLY a keyed hash of the verified email (h1:<HMAC-SHA256> with SCIFORGE_LIVE_QUOTA_SALT, or
-- s1:<SHA-256> without a salt) and the run start time. Never emails, API keys, OIDC tokens, prompts,
-- results or reasoning.

CREATE TABLE IF NOT EXISTS sciforge_live_quota_events (
    id            BIGSERIAL PRIMARY KEY,
    identity_hash TEXT        NOT NULL CHECK (identity_hash ~ '^(h1|s1):[0-9a-f]{64}$'),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS sciforge_live_quota_events_identity_created_idx
    ON sciforge_live_quota_events (identity_hash, created_at);

-- Least privilege for the app role (replace <APP_DB_USER>; run as the owner):
-- GRANT SELECT, INSERT, DELETE ON sciforge_live_quota_events TO <APP_DB_USER>;
-- GRANT USAGE ON SEQUENCE sciforge_live_quota_events_id_seq TO <APP_DB_USER>;
