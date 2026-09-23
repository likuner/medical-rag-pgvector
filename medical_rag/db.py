"""PostgreSQL / pgvector access layer.

Creates the schema (documents + chunks), manages connections, and provides
bulk-insert helpers. Vectors are stored in a `vector(N)` column and indexed
with HNSW for fast cosine similarity search.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Iterable, Sequence

import psycopg
from psycopg.types.json import Jsonb
from pgvector.psycopg import register_vector

from .config import CONFIG

log = logging.getLogger(__name__)

def _schema_sql() -> str:
    """Build the DDL, reading the embedding dim at call time (the model loads
    before schema creation and may set the true dimensionality)."""
    dim = CONFIG.embedding.dim
    return f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source        text        NOT NULL,           -- site key, e.g. 'who_zh'
    source_name   text        NOT NULL,           -- human readable site name
    source_url    text        NOT NULL,           -- canonical page URL
    title         text,
    page_number   integer,                        -- for paged sources, else NULL
    lang          text,
    doc_metadata  jsonb       NOT NULL DEFAULT '{{}}'::jsonb,
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
    embedding    vector({dim}) NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
"""


def connect() -> psycopg.Connection:
    """Open a connection, ensure the vector extension exists, and register the
    vector adapter (registration requires the `vector` type to be present)."""
    conn = psycopg.connect(CONFIG.db.dsn, autocommit=False)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    register_vector(conn)
    return conn


def init_schema(conn: psycopg.Connection) -> None:
    cur = conn.cursor()
    cur.execute(_schema_sql())
    conn.commit()
    cur.close()


def create_vector_index(conn: psycopg.Connection, *, lists: int | None = None) -> None:
    """Create an HNSW index over the embedding column (cosine distance)."""
    if lists is None and CONFIG.embedding.dim >= 1536:
        lists = 100
    cur = conn.cursor()
    # HNSW is available in pgvector >= 0.5; it's robust on small datasets.
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_chunks_embedding
        ON chunks USING hnsw (embedding vector_cosine_ops);
        """
    )
    conn.commit()
    cur.close()


def insert_document(
    conn: psycopg.Connection,
    *,
    source: str,
    source_name: str,
    source_url: str,
    title: str | None,
    page_number: int | None,
    lang: str | None,
    doc_metadata: dict[str, Any] | None,
    raw_text: str | None,
) -> int:
    """Insert a document row and return its id."""
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO documents
            (source, source_name, source_url, title, page_number, lang,
             doc_metadata, raw_text)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (
            source,
            source_name,
            source_url,
            title,
            page_number,
            lang,
            Jsonb(doc_metadata or {}),
            raw_text,
        ),
    )
    doc_id = cur.fetchone()[0]
    cur.close()
    return doc_id


def insert_chunks(
    conn: psycopg.Connection,
    document_id: int,
    rows: Sequence[dict[str, Any]],
) -> int:
    """Bulk-insert chunks for a single document.

    Each row needs: chunk_index, chunk_text, char_length, token_count,
    embedding (list[float]).
    """
    if not rows:
        return 0
    cur = conn.cursor()
    data = [
        (
            document_id,
            r["chunk_index"],
            r["chunk_text"],
            r["char_length"],
            r.get("token_count"),
            r["embedding"],
        )
        for r in rows
    ]
    cur.executemany(
        """
        INSERT INTO chunks
            (document_id, chunk_index, chunk_text, char_length, token_count, embedding)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        data,
    )
    cur.close()
    return len(data)


def upsert_document_thread(
    conn: psycopg.Connection,
    *,
    source: str,
    source_url: str,
    page_number: int | None,
    doc_metadata: dict[str, Any] | None,
) -> int | None:
    """Uniqueness guard: find an existing document id for the same source url.

    Returns the existing id if found, else None. Used to make re-runs
    idempotent (we skip documents whose URL we already ingested).
    """
    cur = conn.cursor()
    cur.execute(
        "SELECT id FROM documents WHERE source = %s AND source_url = %s LIMIT 1",
        (source, source_url),
    )
    row = cur.fetchone()
    cur.close()
    return row[0] if row else None
