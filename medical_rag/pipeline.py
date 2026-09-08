"""End-to-end pipeline: crawl -> clean -> chunk -> embed -> store in pgvector.

The pipeline is organised so that a single run touches all five sites, but it
can be scoped to specific sites with ``sites=[...]``.  Reruns are idempotent:
documents whose ``(source, source_url)`` already exists are skipped, and the
vector index is (re)built at the end if it does not exist.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from . import db as dblib
from .chunking import Chunk, chunk_document
from .cleaners import clean_text, dedupe_lines, detect_lang, html_to_text
from .config import CONFIG, PipelineConfig
from .crawling.base import RawDoc
from .crawling.registry import get_crawlers
from .embedder import Embedder

log = logging.getLogger(__name__)

_MIN_DOC_TEXT = 120


@dataclass
class DocRecord:
    source: str
    source_name: str
    source_url: str
    title: Optional[str]
    page_number: Optional[int]
    lang: Optional[str]
    doc_metadata: dict[str, Any]
    raw_text: str
    chunks: list[Chunk] = field(default_factory=list)


def _clean_raw(raw: RawDoc) -> str:
    if raw.content_kind == "html":
        text = html_to_text(raw.raw)
    else:
        text = clean_text(raw.raw)
    text = dedupe_lines(text)
    text = clean_text(text)
    return text


def ingest(
    conn,
    embedder: Embedder,
    records: list[DocRecord],
    *,
    batch_size: int = 128,
) -> dict[str, int]:
    """Insert records (doc + chunks + embeddings) into the database."""
    stats = {"documents": 0, "chunks": 0}
    for rec in records:
        doc_id = dblib.insert_document(
            conn,
            source=rec.source,
            source_name=rec.source_name,
            source_url=rec.source_url,
            title=rec.title,
            page_number=rec.page_number,
            lang=rec.lang,
            doc_metadata=rec.doc_metadata,
            raw_text=rec.raw_text,
        )
        texts = [c.text for c in rec.chunks]
        vecs: np.ndarray = embedder.embed(texts)
        rows = [
            {**c.as_row(), "embedding": vecs[i].tolist()}
            for i, c in enumerate(rec.chunks)
        ]
        dblib.insert_chunks(conn, doc_id, rows)
        conn.commit()
        stats["documents"] += 1
        stats["chunks"] += len(rows)
        if stats["documents"] % 5 == 0:
            log.info("  ingested %d docs / %d chunks", stats["documents"], stats["chunks"])
    return stats


def run(
    *,
    sites: list[str] | None = None,
    max_pages: Optional[int] = None,
    recreate: bool = False,
    embed_backend: Optional[str] = None,
) -> dict[str, Any]:
    cfg: PipelineConfig = CONFIG

    # ------------------------------------------------------------------ embed
    if embed_backend:
        cfg.embedding.backend = embed_backend
    embedder = Embedder(cfg.embedding).load()
    log.info("Embedding backend=%s dim=%d model=%s",
             embedder.backend, embedder.dim, getattr(cfg.embedding, "model_name", "-"))

    # -------------------------------------------------------------------- db
    conn = dblib.connect()
    if recreate:
        conn.execute("DROP TABLE IF EXISTS chunks, documents")
        conn.commit()
    dblib.init_schema(conn)          # reads cfg.embedding.dim at call time
    log.info("DB schema ready (dim=%d).", cfg.embedding.dim)

    # ------------------------------------------------------------------ crawl
    crawlers = get_crawlers(sites, cfg.crawl)
    records: list[DocRecord] = []
    lang_counter: dict[str, int] = {}

    for crawler in crawlers:
        log.info("Crawling %s (%s)", crawler.key, crawler.name)
        for raw in crawler.crawl(max_pages):
            text = _clean_raw(raw)
            if len(text) < _MIN_DOC_TEXT:
                continue
            # Idempotency: skip documents we already stored on a previous run.
            existing = dblib.upsert_document_thread(
                conn, source=raw.source, source_url=raw.source_url,
                page_number=raw.page_number, doc_metadata=raw.doc_metadata,
            )
            if existing is not None:
                log.info("  already present, skip: %s", raw.source_url)
                continue
            lang = raw.lang or detect_lang(text)
            lang_counter[lang] = lang_counter.get(lang, 0) + 1
            chunks = chunk_document(text, cfg.chunk)
            if not chunks:
                continue
            meta = dict(raw.doc_metadata)
            meta.setdefault("chunks", len(chunks))
            records.append(
                DocRecord(
                    source=raw.source,
                    source_name=raw.source_name,
                    source_url=raw.source_url,
                    title=raw.title,
                    page_number=raw.page_number,
                    lang=lang,
                    doc_metadata=meta,
                    raw_text=text,
                    chunks=chunks,
                )
            )
            log.info("  ~ %d chars, %d chunks -> %s",
                     len(text), len(chunks), raw.source_url)

    log.info("Collected %d new documents from %d crawler(s). langs=%s",
             len(records), len(crawlers), lang_counter)

    if not records:
        embedder.save_state()
        conn.close()
        return {"documents": 0, "chunks": 0, "embedding": embedder.backend,
                "embedding_dim": cfg.embedding.dim}

    # -------------------------------------------------------- TF-IDF corpus
    if embedder.backend == "tfidf":
        corpus = [c.text for rec in records for c in rec.chunks]
        embedder.fit(corpus)
        embedder.save_state()

    # ------------------------------------------------------------------ store
    stats = ingest(conn, embedder, records)
    if embedder.backend != "tfidf":
        embedder.save_state()
    dblib.create_vector_index(conn)
    conn.close()

    log.info("Done: %s", stats)
    return {
        "documents": stats["documents"],
        "chunks": stats["chunks"],
        "embedding": embedder.backend,
        "embedding_dim": cfg.embedding.dim,
        "langs": lang_counter,
    }


def search(query: str, k: int = 5, sites: list[str] | None = None) -> list[dict]:
    """Similarity search demo against the stored chunks."""
    from .embedder import Embedder

    embedder = Embedder.load_state(CONFIG.embedding)
    qvec = embedder.embed_query(query).tolist()
    conn = dblib.connect()
    cur = conn.cursor()
    where = "c.embedding IS NOT NULL"
    if sites:
        where += " AND d.source = ANY(%s)"

    cur.execute(
        f"""
        SELECT d.source, d.source_url, d.title, c.chunk_index,
               c.chunk_text, 1 - (c.embedding <=> %s::vector) AS similarity
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE {where}
        ORDER BY c.embedding <=> %s::vector
        LIMIT %s
        """,
        [qvec, sites, qvec, k] if sites else [qvec, qvec, k],
    )
    rows = [
        {
            "source": r[0],
            "url": r[1],
            "title": r[2],
            "chunk_index": r[3],
            "text": r[4],
            "similarity": round(float(r[5]), 4),
        }
        for r in cur.fetchall()
    ]
    cur.close()
    conn.close()
    return rows
