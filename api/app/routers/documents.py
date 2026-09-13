"""Document endpoints: list, upload, inspect, delete, and serve the source PDF.

Uploads are ingested in a background task so the request returns immediately; the client polls
document status. At this corpus size ingestion takes a couple of seconds, but the shape is the
one that scales — replacing the background task with a real queue is the first production change
and needs no API change.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.core.auth import get_current_user
from app.core.config import get_settings
from app.core.paths import CORPUS_DIR
from app.db.pool import connection
from app.rag.ingest import ingest_pdf, mark_failed
from app.schemas.models import DocumentOut

router = APIRouter(prefix="/documents", tags=["documents"])

UPLOAD_DIR = Path("uploads")
MAX_UPLOAD_BYTES = 30 * 1024 * 1024


@router.get("", response_model=list[DocumentOut])
def list_documents(user: dict = Depends(get_current_user)) -> list[DocumentOut]:
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT id, doc_key, title, filename, source, n_pages, n_chunks,
                   status, error, created_at, ingested_at
            FROM documents
            WHERE user_id IS NULL OR user_id = %s
            ORDER BY created_at DESC, id DESC
            """,
            (user["id"],),
        ).fetchall()
    return [DocumentOut(**row) for row in rows]


def _pdf_path(filename: str, source: str, user_id: int | None) -> Path:
    if source == "seed":
        return CORPUS_DIR / filename
    # Legacy shared uploads (user_id NULL) live in uploads/; owned files in uploads/{user_id}/.
    if user_id is None:
        return UPLOAD_DIR / filename
    return UPLOAD_DIR / str(user_id) / filename


@router.get("/{document_id}/file")
def get_document_file(document_id: int) -> FileResponse:
    """Serve the original PDF so the UI can open a citation at its page.

    Unauthenticated on purpose: the evidence iframe cannot send a Bearer header.
    Seed papers are the shared corpus. Upload IDs are unguessable enough for a local demo;
    a signed cookie on this route is the leftover hardening step, not another login page.
    """
    with connection() as conn:
        row = conn.execute(
            "SELECT filename, source, user_id FROM documents WHERE id = %s", (document_id,)
        ).fetchone()
    if not row:
        raise HTTPException(404, "document not found")

    path = _pdf_path(row["filename"], row["source"], row["user_id"])
    if not path.exists():
        raise HTTPException(404, f"file missing on disk: {row['filename']}")
    # `inline` is required: passing `filename` alone makes Starlette send
    # `Content-Disposition: attachment`, which makes the browser download the PDF instead of
    # rendering it in the evidence pane's iframe.
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=row["filename"],
        content_disposition_type="inline",
    )


def _ingest_upload(path: Path, doc_key: str, user_id: int) -> None:
    try:
        with connection() as conn:
            ingest_pdf(conn, path, doc_key=doc_key, source="upload", user_id=user_id)
    except Exception as exc:  # noqa: BLE001
        with connection() as conn:
            mark_failed(conn, doc_key, str(exc), user_id=user_id)


@router.post("/upload", response_model=DocumentOut, status_code=202)
async def upload_document(
    file: UploadFile, background: BackgroundTasks, user: dict = Depends(get_current_user)
) -> DocumentOut:
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "only PDF files are accepted")

    UPLOAD_DIR.mkdir(exist_ok=True)
    doc_key = Path(file.filename).stem

    # Stream to a temp file first so an oversized upload is rejected without being kept, and a
    # failed write never leaves a partial file where the reader expects a valid PDF.
    size = 0
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                Path(tmp.name).unlink(missing_ok=True)
                raise HTTPException(413, "file exceeds 30MB limit")
            tmp.write(chunk)
        tmp_path = Path(tmp.name)

    dest_dir = UPLOAD_DIR / str(user["id"])
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{doc_key}.pdf"
    shutil.move(str(tmp_path), dest)

    with connection() as conn:
        row = conn.execute(
            """
            INSERT INTO documents (doc_key, title, filename, source, status, bytes, user_id)
            VALUES (%s, %s, %s, 'upload', 'pending', %s, %s)
            ON CONFLICT (user_id, doc_key) WHERE user_id IS NOT NULL DO UPDATE
                SET status = 'pending', error = NULL, bytes = EXCLUDED.bytes
            RETURNING id, doc_key, title, filename, source, n_pages, n_chunks,
                      status, error, created_at, ingested_at
            """,
            (doc_key, doc_key, dest.name, size, user["id"]),
        ).fetchone()

    background.add_task(_ingest_upload, dest, doc_key, user["id"])
    return DocumentOut(**row)


@router.delete("/{document_id}", status_code=204)
def delete_document(document_id: int, user: dict = Depends(get_current_user)) -> None:
    with connection() as conn:
        row = conn.execute(
            """
            DELETE FROM documents
             WHERE id = %s AND user_id = %s
            RETURNING filename, source
            """,
            (document_id, user["id"]),
        ).fetchone()
    if not row:
        raise HTTPException(404, "document not found")
    # Only remove uploaded files; the seed corpus is version-controlled input.
    if row["source"] == "upload":
        _pdf_path(row["filename"], row["source"], user["id"]).unlink(missing_ok=True)


@router.post("/reingest", status_code=202)
def reingest_seed(
    background: BackgroundTasks, _user: dict = Depends(get_current_user)
) -> dict:
    """Re-ingest the seed corpus — used after toggling CONTEXTUALIZE."""
    settings = get_settings()
    pdfs = sorted(CORPUS_DIR.glob("*.pdf"))
    if not pdfs:
        raise HTTPException(404, f"no seed PDFs found in {CORPUS_DIR}")

    def run() -> None:
        for pdf in pdfs:
            try:
                with connection() as conn:
                    ingest_pdf(conn, pdf, source="seed", settings=settings)
            except Exception as exc:  # noqa: BLE001
                with connection() as conn:
                    mark_failed(conn, pdf.stem, str(exc))

    background.add_task(run)
    return {"scheduled": len(pdfs), "contextualize": settings.contextualize}
