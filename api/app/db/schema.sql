

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------- documents

CREATE TABLE IF NOT EXISTS documents (
    id            BIGSERIAL PRIMARY KEY,
    doc_key       TEXT NOT NULL,                 -- stable external id (e.g. arXiv id); uniqueness is per-tenant below
    title         TEXT NOT NULL,
    filename      TEXT NOT NULL,
    source        TEXT NOT NULL DEFAULT 'upload', -- 'seed' | 'upload'
    n_pages       INTEGER,
    n_chunks      INTEGER NOT NULL DEFAULT 0,
    status        TEXT NOT NULL DEFAULT 'pending', -- pending|parsing|embedding|ready|failed
    error         TEXT,
    bytes         BIGINT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    ingested_at   TIMESTAMPTZ
);

-- ---------------------------------------------------------------- chunks

CREATE TABLE IF NOT EXISTS chunks (
    id            BIGSERIAL PRIMARY KEY,
    document_id   BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal       INTEGER NOT NULL,              -- position within document
    page_start    INTEGER NOT NULL,
    page_end      INTEGER NOT NULL,
    section       TEXT,                          -- best-effort heading
    raw_content   TEXT NOT NULL,                 -- verbatim source text
    content       TEXT NOT NULL,                 -- indexed text (= raw, or context-prefixed)
    context_note  TEXT,                          -- the LLM-generated situating blurb, if any
    n_tokens      INTEGER NOT NULL,
    embedding     vector(384),                   -- bge-small-en-v1.5
    tsv           tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, ordinal)
);

-- Vector arm. m/ef_construction per pgvector guidance for read-heavy corpora.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);

-- Lexical arm.
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);

CREATE INDEX IF NOT EXISTS chunks_document_id ON chunks (document_id);

-- ---------------------------------------------------------------- users
-- Local email/password accounts. Conversations, uploaded documents, and their chunks
-- carry user_id. Seed papers keep user_id NULL so every account still searches them.

CREATE TABLE IF NOT EXISTS users (
    id             BIGSERIAL PRIMARY KEY,
    email          TEXT NOT NULL UNIQUE,
    password_hash  TEXT NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sessions (
    token       TEXT PRIMARY KEY,
    user_id     BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expires_at  TIMESTAMPTZ NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS sessions_user_id ON sessions (user_id);

-- Document tenancy: seed papers stay user_id NULL (shared). Uploads belong to an account.
-- Conversations already have user_id. This is the matching column on the index, not another
-- login page. Partial uniques: one shared doc_key, and one (user_id, doc_key) per owner.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS user_id BIGINT REFERENCES users(id) ON DELETE CASCADE;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS user_id BIGINT REFERENCES users(id) ON DELETE CASCADE;

ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_doc_key_key;

CREATE UNIQUE INDEX IF NOT EXISTS documents_shared_doc_key
    ON documents (doc_key)
    WHERE user_id IS NULL;

CREATE UNIQUE INDEX IF NOT EXISTS documents_owned_doc_key
    ON documents (user_id, doc_key)
    WHERE user_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS documents_user_id ON documents (user_id);
CREATE INDEX IF NOT EXISTS chunks_user_id ON chunks (user_id);

-- ---------------------------------------------------------------- conversations

CREATE TABLE IF NOT EXISTS conversations (
    id            BIGSERIAL PRIMARY KEY,
    user_id       BIGINT REFERENCES users(id) ON DELETE CASCADE,
    title         TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE conversations ADD COLUMN IF NOT EXISTS user_id BIGINT REFERENCES users(id) ON DELETE CASCADE;

CREATE INDEX IF NOT EXISTS conversations_user_id ON conversations (user_id, id DESC);

CREATE TABLE IF NOT EXISTS messages (
    id              BIGSERIAL PRIMARY KEY,
    conversation_id BIGINT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,               -- 'user' | 'assistant'
    content         TEXT NOT NULL,
    abstained       BOOLEAN NOT NULL DEFAULT false,
    latency_ms      INTEGER,
    prompt_tokens   INTEGER,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS messages_conversation_id ON messages (conversation_id, id);

-- Which chunks were cited by an assistant message, and in what display order.
CREATE TABLE IF NOT EXISTS citations (
    id            BIGSERIAL PRIMARY KEY,
    message_id    BIGINT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    chunk_id      BIGINT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    marker        INTEGER NOT NULL,              -- the [n] shown to the user
    UNIQUE (message_id, marker)
);

-- ---------------------------------------------------------------- retrieval traces
-- One row per candidate chunk considered for a message. This is the data behind the
-- "why this answer" panel: each arm's rank, the fused score, the rerank score, and
-- whether the chunk survived into the final prompt.

CREATE TABLE IF NOT EXISTS retrieval_traces (
    id              BIGSERIAL PRIMARY KEY,
    message_id      BIGINT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    chunk_id        BIGINT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    vector_rank     INTEGER,                     -- NULL = not returned by that arm
    vector_score    REAL,                        -- cosine similarity (1 - distance)
    lexical_rank    INTEGER,
    lexical_score   REAL,                        -- ts_rank_cd
    rrf_score       REAL NOT NULL,
    rrf_rank        INTEGER NOT NULL,
    rerank_score    REAL,
    final_rank      INTEGER,                     -- NULL = did not make the context window
    used_in_context BOOLEAN NOT NULL DEFAULT false
);

CREATE INDEX IF NOT EXISTS retrieval_traces_message_id ON retrieval_traces (message_id);

-- ---------------------------------------------------------------- eval

CREATE TABLE IF NOT EXISTS eval_runs (
    id            BIGSERIAL PRIMARY KEY,
    label         TEXT NOT NULL,                 -- ablation config name
    config        JSONB NOT NULL,
    metrics       JSONB,
    n_questions   INTEGER,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
