"""Ingest every PDF in the corpus directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.core.config import get_settings
from app.core.paths import CORPUS_DIR
from app.db.pool import connection, init_schema
from app.rag.ingest import ingest_pdf, mark_failed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", default=str(CORPUS_DIR))
    ap.add_argument("--source", default="seed")
    ap.add_argument("--reset", action="store_true", help="delete all documents first")
    args = ap.parse_args()

    settings = get_settings()
    init_schema()

    pdfs = sorted(Path(args.corpus_dir).glob("*.pdf"))
    if not pdfs:
        print(f"no PDFs in {args.corpus_dir}", file=sys.stderr)
        return 1

    print(f"ingesting {len(pdfs)} PDFs  (contextualize={settings.contextualize})")

    if args.reset:
        with connection() as conn:
            conn.execute("DELETE FROM documents")
        print("  cleared existing documents")

    total_chunks = 0
    failures = 0
    for i, pdf in enumerate(pdfs, start=1):
        try:
            with connection() as conn:
                report = ingest_pdf(conn, pdf, source=args.source, settings=settings)
            total_chunks += report.n_chunks
            print(
                f"  [{i:2d}/{len(pdfs)}] {report.doc_key:14s} "
                f"{report.n_pages:2d}p  {report.n_chunks:3d} chunks  {report.seconds:5.2f}s"
            )
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  [{i:2d}/{len(pdfs)}] {pdf.stem:14s} FAILED: {exc}", file=sys.stderr)
            with connection() as conn:
                mark_failed(conn, pdf.stem, str(exc))

    print(f"\ndone: {total_chunks} chunks from {len(pdfs) - failures}/{len(pdfs)} documents")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
