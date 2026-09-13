"""Ingestion pipeline: PDF → pages → chunks → (optional contextualisation) → embeddings → Postgres.

Idempotent on `doc_key`: re-ingesting a document replaces its chunks rather than duplicating
them, so a failed run can simply be retried.

Contextualisation is a flagged stage. It is the only part of ingestion that needs an LLM, and
disabling it must leave a fully working index. The enrichment module is not shipped; keep
`CONTEXTUALIZE=false`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from app.core.config import Settings, get_settings
from app.rag.chunking import chunk_pages, extract_pages, extract_title
from app.rag.embedder import Embedder


@dataclass
class IngestReport:
    doc_key: str
    document_id: int
    n_pages: int
    n_chunks: int
    contextualized: bool
    seconds: float


def ingest_pdf(
    conn,
    pdf_path: str | Path,
    *,
    doc_key: str | None = None,
    source: str = "upload",
    user_id: int | None = None,
    settings: Settings | None = None,
) -> IngestReport:
    settings = settings or get_settings()
    pdf_path = Path(pdf_path)
    doc_key = doc_key or pdf_path.stem
    started = time.perf_counter()

    pages = extract_pages(str(pdf_path))
    chunks = chunk_pages(
        pages,
        target_tokens=settings.chunk_target_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )
    if not chunks:
        raise ValueError(f"no extractable text in {pdf_path.name} (scanned or image-only PDF?)")

    title = extract_title(str(pdf_path), fallback=doc_key)

    row = conn.execute(
        """
        INSERT INTO documents (doc_key, title, filename, source, n_pages, status, bytes, user_id)
        VALUES (%s, %s, %s, %s, %s, 'parsing', %s, %s)
        ON CONFLICT (doc_key) WHERE user_id IS NULL DO UPDATE
            SET title = EXCLUDED.title,
                filename = EXCLUDED.filename,
                source = EXCLUDED.source,
                n_pages = EXCLUDED.n_pages,
                status = 'parsing',
                error = NULL,
                bytes = EXCLUDED.bytes
        RETURNING id
        """
        if user_id is None
        else """
        INSERT INTO documents (doc_key, title, filename, source, n_pages, status, bytes, user_id)
        VALUES (%s, %s, %s, %s, %s, 'parsing', %s, %s)
        ON CONFLICT (user_id, doc_key) WHERE user_id IS NOT NULL DO UPDATE
            SET title = EXCLUDED.title,
                filename = EXCLUDED.filename,
                source = EXCLUDED.source,
                n_pages = EXCLUDED.n_pages,
                status = 'parsing',
                error = NULL,
                bytes = EXCLUDED.bytes
        RETURNING id
        """,
        (doc_key, title, pdf_path.name, source, len(pages), pdf_path.stat().st_size, user_id),
    ).fetchone()
    document_id = row["id"]

    # Replace rather than append, so re-ingest is safe.
    conn.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))

    # ---- optional index-time enrichment
    context_notes: list[str | None] = [None] * len(chunks)
    contextualized = False
    if settings.contextualize and settings.has_llm:
        from app.rag.contextualize import contextualize_chunks

        doc_summary = "\n\n".join(c.text for c in chunks[:3])[:3000]
        context_notes = contextualize_chunks(
            [c.text for c in chunks], document_title=title, document_head=doc_summary
        )
        contextualized = True

    # The text we index differs from the text we display: `content` may carry the situating
    # blurb, `raw_content` is always verbatim source.
    index_texts = [
        f"{note}\n\n{chunk.text}" if note else chunk.text
        for chunk, note in zip(chunks, context_notes, strict=True)
    ]

    conn.execute("UPDATE documents SET status = 'embedding' WHERE id = %s", (document_id,))
    vectors = Embedder.instance().embed_passages(index_texts)

    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO chunks (document_id, user_id, ordinal, page_start, page_end, section,
                                raw_content, content, context_note, n_tokens, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (
                    document_id,
                    user_id,
                    chunk.ordinal,
                    chunk.page_start,
                    chunk.page_end,
                    chunk.section,
                    chunk.text,
                    index_text,
                    note,
                    chunk.n_tokens,
                    str(vector),
                )
                for chunk, index_text, note, vector in zip(
                    chunks, index_texts, context_notes, vectors, strict=True
                )
            ],
        )

    conn.execute(
        """
        UPDATE documents
           SET n_chunks = %s, status = 'ready', ingested_at = now()
         WHERE id = %s
        """,
        (len(chunks), document_id),
    )

    return IngestReport(
        doc_key=doc_key,
        document_id=document_id,
        n_pages=len(pages),
        n_chunks=len(chunks),
        contextualized=contextualized,
        seconds=time.perf_counter() - started,
    )


def mark_failed(conn, doc_key: str, error: str, *, user_id: int | None = None) -> None:
    if user_id is None:
        conn.execute(
            """
            UPDATE documents SET status = 'failed', error = %s
             WHERE doc_key = %s AND user_id IS NULL
            """,
            (error[:2000], doc_key),
        )
    else:
        conn.execute(
            """
            UPDATE documents SET status = 'failed', error = %s
             WHERE doc_key = %s AND user_id = %s
            """,
            (error[:2000], doc_key, user_id),
        )
