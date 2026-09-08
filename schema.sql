
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source        text        NOT NULL,           -- site key, e.g. 'pubmed'
    source_name   text        NOT NULL,           -- human readable site name
    source_url    text        NOT NULL,           -- canonical page URL
    title         text,
    page_number   integer,                        -- for paged sources, else NULL
    lang          text,
    doc_metadata  jsonb       NOT NULL DEFAULT '{}'::jsonb,
    raw_text      text,                           -- full cleaned text (pre-chunk)
    created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id  bigint NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index  integer NOT NULL,
    chunk_text   text NOT NULL,
    char_length  integer NOT NULL,
    token_count  integer,
    embedding    vector(512) NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON chunks USING hnsw (embedding vector_cosine_ops);
