-- Document tenancy: seed rows stay shared (user_id IS NULL); uploads belong to an account.
-- Conversations already have user_id. This is the matching column on the index, not another
-- login page.

ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS user_id BIGINT REFERENCES users(id) ON DELETE CASCADE;

ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS user_id BIGINT REFERENCES users(id) ON DELETE CASCADE;

-- Seed corpus keeps a single shared doc_key. Uploads are unique per owner.
ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_doc_key_key;

CREATE UNIQUE INDEX IF NOT EXISTS documents_shared_doc_key
    ON documents (doc_key)
    WHERE user_id IS NULL;

CREATE UNIQUE INDEX IF NOT EXISTS documents_owned_doc_key
    ON documents (user_id, doc_key)
    WHERE user_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS documents_user_id ON documents (user_id);
CREATE INDEX IF NOT EXISTS chunks_user_id ON chunks (user_id);

-- Existing chunks inherit the parent document's owner (NULL for seed / legacy uploads).
UPDATE chunks c
   SET user_id = d.user_id
  FROM documents d
 WHERE c.document_id = d.id
   AND c.user_id IS DISTINCT FROM d.user_id;
