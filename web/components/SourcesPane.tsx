"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { IconUpload } from "@/components/icons";
import { deleteDocument, getDocuments, uploadDocument } from "@/lib/api";
import type { Document } from "@/lib/types";

const STATUS_LABEL: Record<Document["status"], string> = {
  pending: "Queued",
  parsing: "Reading pages",
  embedding: "Embedding",
  ready: "Ready",
  failed: "Failed",
};

function StatusBadge({ status }: { status: Document["status"] }) {
  const styles: Record<Document["status"], string> = {
    ready: "text-positive",
    failed: "text-danger",
    pending: "text-ink-subtle",
    parsing: "text-warning",
    embedding: "text-warning",
  };
  const busy = status === "pending" || status === "parsing" || status === "embedding";
  return (
    <span className={`text-[11px] font-medium ${styles[status]}`}>
      {busy && (
        <span
          aria-hidden="true"
          className="mr-1 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-current align-middle"
        />
      )}
      {STATUS_LABEL[status]}
    </span>
  );
}

export function SourcesPane({
  onSelectDocument,
  activeDocumentId,
}: {
  onSelectDocument: (doc: Document) => void;
  activeDocumentId: number | null;
}) {
  const [docs, setDocs] = useState<Document[]>([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [reloadToken, setReloadToken] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(() => setReloadToken((n) => n + 1), []);

  /**
   * Load the document list, then re-poll only while an ingest is still in flight. The poll
   * reschedules itself from the response rather than running on a fixed interval, so an idle
   * page issues no requests and a slow response cannot stack up overlapping ones.
   */
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    async function poll() {
      try {
        const next = await getDocuments();
        if (cancelled) return;
        setDocs(next);
        setError(null);
        if (next.some((d) => d.status !== "ready" && d.status !== "failed")) {
          timer = setTimeout(poll, 1500);
        }
      } catch {
        if (!cancelled) {
          setError("Could not reach the API. Is the backend running on port 8000?");
        }
      }
    }

    void poll();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [reloadToken]);

  async function handleFiles(files: FileList | null) {
    if (!files?.length) return;
    setError(null);
    setUploading(true);
    try {
      for (const file of Array.from(files)) await uploadDocument(file);
      refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploading(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  }

  const readyCount = docs.filter((d) => d.status === "ready").length;
  const totalChunks = docs.reduce((sum, d) => sum + d.n_chunks, 0);

  return (
    <div className="flex h-full flex-col bg-rail">
      <div className="px-4 pb-2 pt-4">
        <h2 className="text-[11px] font-semibold uppercase tracking-wider text-ink-subtle">
          Sources
        </h2>
        <p className="mt-1 text-[12px] text-ink-muted">
          {readyCount} {readyCount === 1 ? "document" : "documents"} · {totalChunks} chunks
        </p>
      </div>

      <div className="px-4 pb-3">
        {/* Elevated against the recessed rail, so the one action here reads as actionable. */}
        <label
          htmlFor="pdf-upload"
          className="flex cursor-pointer items-center justify-center gap-2 rounded-lg bg-surface px-3 py-2.5 text-[13px] font-medium text-ink shadow-raised transition-colors hover:text-accent"
        >
          <IconUpload className="h-4 w-4" />
          {uploading ? "Uploading…" : "Add PDFs"}
        </label>
        <input
          ref={inputRef}
          id="pdf-upload"
          type="file"
          accept="application/pdf"
          multiple
          disabled={uploading}
          onChange={(e) => void handleFiles(e.target.files)}
          className="sr-only"
        />
        <p className="mt-1.5 text-center text-[11px] text-ink-subtle">
          Text-based PDFs, up to 30 MB
        </p>
        {error && (
          <p role="alert" className="mt-2 text-[12px] text-danger">
            {error}
          </p>
        )}
      </div>

      <ul className="scroll-area flex-1 overflow-y-auto px-2 pb-3" aria-label="Indexed documents">
        {docs.map((doc) => {
          const isActive = doc.id === activeDocumentId;
          return (
            <li key={doc.id}>
              <div
                className={`group rounded-lg px-2 py-2 transition-colors ${
                  isActive ? "bg-surface shadow-raised" : "hover:bg-rail-hover"
                }`}
              >
                <button
                  type="button"
                  onClick={() => onSelectDocument(doc)}
                  aria-current={isActive ? "true" : undefined}
                  className="block w-full text-left"
                >
                  <span
                    className={`line-clamp-2 text-[12.5px] font-medium leading-[1.4] ${
                      isActive ? "text-accent" : "text-ink"
                    }`}
                  >
                    {doc.title}
                  </span>
                  <span className="mt-1 flex items-center gap-2 text-[11px] text-ink-subtle">
                    <span className="font-mono">{doc.doc_key}</span>
                    {doc.n_pages ? <span>{doc.n_pages}p</span> : null}
                    <StatusBadge status={doc.status} />
                  </span>
                </button>
                {doc.error && <p className="mt-1 text-[11px] text-danger">{doc.error}</p>}
                {doc.source === "upload" && (
                  <button
                    type="button"
                    onClick={async () => {
                      await deleteDocument(doc.id);
                      refresh();
                    }}
                    className="mt-1 text-[11px] text-ink-subtle underline opacity-0 transition-opacity hover:text-danger focus-visible:opacity-100 group-hover:opacity-100"
                  >
                    Remove <span className="sr-only">{doc.title}</span>
                  </button>
                )}
              </div>
            </li>
          );
        })}
        {docs.length === 0 && !error && (
          <li className="px-2 py-4 text-[12px] text-ink-subtle">
            No documents yet. Upload a PDF to get started.
          </li>
        )}
      </ul>
    </div>
  );
}
