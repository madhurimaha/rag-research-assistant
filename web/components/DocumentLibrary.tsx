"use client";

import { useRef } from "react";
import { IconDoc, IconUpload } from "@/components/icons";
import type { Document } from "@/lib/types";

const STATUS_LABEL: Record<Document["status"], string> = {
  pending: "Queued",
  parsing: "Reading pages",
  embedding: "Embedding",
  ready: "Ready",
  failed: "Failed",
};

export function DocumentLibrary({
  documents,
  uploading,
  uploadError,
  onUpload,
  onOpenDocument,
  onDeleteDocument,
  onStartChat,
}: {
  documents: Document[];
  uploading: boolean;
  uploadError: string | null;
  onUpload: (files: FileList | null) => void;
  onOpenDocument: (document: Document) => void;
  onDeleteDocument: (document: Document) => void;
  onStartChat: () => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const readyCount = documents.filter((document) => document.status === "ready").length;

  return (
    <section className="scroll-area h-full overflow-y-auto bg-surface px-6 py-7 sm:px-8">
      <div className="mx-auto max-w-6xl">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <p className="text-[12px] font-semibold uppercase tracking-wider text-accent">
              Your knowledge base
            </p>
            <h2 className="mt-1 text-[24px] font-semibold tracking-[-0.018em]">
              Document library
            </h2>
            <p className="mt-1 text-[13px] text-ink-muted">
              {readyCount} of {documents.length} documents ready for questions
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={onStartChat}
              className="rounded-lg bg-rail px-3.5 py-2.5 text-[13px] font-medium text-ink transition-colors hover:bg-rail-hover"
            >
              Ask documents
            </button>
            <button
              type="button"
              onClick={() => inputRef.current?.click()}
              disabled={uploading}
              className="inline-flex items-center gap-2 rounded-lg bg-accent px-3.5 py-2.5 text-[13px] font-medium text-white shadow-raised transition-colors hover:bg-accent-hover disabled:opacity-50"
            >
              <IconUpload className="h-4 w-4" />
              {uploading ? "Uploading…" : "Add PDFs"}
            </button>
            <input
              ref={inputRef}
              type="file"
              accept="application/pdf"
              multiple
              className="sr-only"
              onChange={(event) => {
                onUpload(event.target.files);
                event.target.value = "";
              }}
            />
          </div>
        </div>

        {uploadError && (
          <p role="alert" className="mt-4 rounded-lg bg-warning-soft px-3 py-2 text-[13px] text-danger">
            {uploadError}
          </p>
        )}

        <ul className="mt-7 grid grid-cols-1 gap-4 lg:grid-cols-2 2xl:grid-cols-3" aria-label="Document library">
          {documents.map((document) => {
            const busy = ["pending", "parsing", "embedding"].includes(document.status);
            return (
              <li key={document.id} className="flex min-h-52 flex-col rounded-2xl bg-sunken p-5 shadow-raised">
                <div className="flex items-start justify-between gap-3">
                  <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-accent-soft text-accent">
                    <IconDoc className="h-5 w-5" />
                  </span>
                  <span
                    role="status"
                    aria-live="polite"
                    className={`inline-flex items-center gap-1.5 rounded-full bg-surface px-2.5 py-1 text-[12px] font-medium ${
                      document.status === "ready"
                        ? "text-positive"
                        : document.status === "failed"
                          ? "text-danger"
                          : "text-warning"
                    }`}
                  >
                    {busy && <span aria-hidden="true" className="h-1.5 w-1.5 animate-pulse rounded-full bg-current" />}
                    {STATUS_LABEL[document.status]}
                  </span>
                </div>

                <h3 className="mt-4 line-clamp-3 text-[15px] font-semibold leading-snug" title={document.title}>
                  {document.title}
                </h3>
                <p className="mt-2 font-mono text-[12px] text-ink-subtle">{document.doc_key}</p>
                <dl className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-[13px] text-ink-muted">
                  <div className="flex gap-1">
                    <dt className="sr-only">Pages</dt>
                    <dd>{document.n_pages ?? "—"} pages</dd>
                  </div>
                  <div className="flex gap-1">
                    <dt className="sr-only">Chunks</dt>
                    <dd>{document.n_chunks} chunks</dd>
                  </div>
                  <div className="flex gap-1">
                    <dt className="sr-only">Source</dt>
                    <dd>{document.source === "seed" ? "Demo collection" : "Uploaded"}</dd>
                  </div>
                </dl>

                {document.error && <p className="mt-2 text-[13px] text-danger">{document.error}</p>}

                <div className="mt-auto flex items-center gap-2 pt-5">
                  <button
                    type="button"
                    onClick={() => onOpenDocument(document)}
                    disabled={document.status === "failed"}
                    className="rounded-lg bg-surface px-3 py-2 text-[13px] font-medium text-accent shadow-raised transition-colors hover:bg-accent-soft disabled:cursor-not-allowed disabled:opacity-45"
                  >
                    Open document
                  </button>
                  {document.source === "upload" && (
                    <button
                      type="button"
                      onClick={() => onDeleteDocument(document)}
                      className="rounded-lg px-3 py-2 text-[13px] font-medium text-ink-muted transition-colors hover:bg-warning-soft hover:text-danger"
                    >
                      Remove
                      <span className="sr-only"> {document.title}</span>
                    </button>
                  )}
                </div>
              </li>
            );
          })}
        </ul>
      </div>
    </section>
  );
}
