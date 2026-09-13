"use client";

import { useRef } from "react";
import { IconClose, IconDoc, IconHistory, IconPlus, IconUpload } from "@/components/icons";
import type { ConversationSummary, Document } from "@/lib/types";

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
    <span role="status" aria-live="polite" className={`text-[12px] font-medium ${styles[status]}`}>
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
  documents,
  conversations,
  activeConversationId,
  onSelectDocument,
  activeDocumentId,
  uploading,
  uploadError,
  onUpload,
  onDeleteDocument,
  onNewChat,
  onOpenConversation,
  onOpenLibrary,
  libraryOpen,
  onClose,
  uploadInputId = "pdf-upload",
}: {
  documents: Document[];
  conversations: ConversationSummary[];
  activeConversationId: number | null;
  onSelectDocument: (doc: Document) => void;
  activeDocumentId: number | null;
  uploading: boolean;
  uploadError: string | null;
  onUpload: (files: FileList | null) => void;
  onDeleteDocument: (doc: Document) => void;
  onNewChat: () => void;
  onOpenConversation: (id: number) => void;
  onOpenLibrary: () => void;
  libraryOpen: boolean;
  onClose?: () => void;
  uploadInputId?: string;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const readyCount = documents.filter((d) => d.status === "ready").length;
  const totalChunks = documents.reduce((sum, d) => sum + d.n_chunks, 0);

  return (
    <div className="flex h-full flex-col bg-rail">
      <div className="px-3 pb-3 pt-4">
        {onClose && (
          <div className="mb-3 flex items-center justify-between px-1">
            <p className="text-[13px] font-semibold">Your workspace</p>
            <button
              type="button"
              onClick={onClose}
              className="rounded-md p-1.5 text-ink-subtle transition-colors hover:bg-rail-hover hover:text-ink"
            >
              <IconClose className="h-4 w-4" />
              <span className="sr-only">Close documents and chats</span>
            </button>
          </div>
        )}
        <button
          type="button"
          onClick={onNewChat}
          className="flex w-full items-center justify-center gap-2 rounded-lg bg-accent px-3 py-2.5 text-[13px] font-medium text-white shadow-raised transition-colors hover:bg-accent-hover"
        >
          <IconPlus className="h-4 w-4" />
          New chat
        </button>
        <label
          htmlFor={uploadInputId}
          className="mt-2 flex cursor-pointer items-center justify-center gap-2 rounded-lg bg-surface px-3 py-2.5 text-[13px] font-medium text-ink shadow-raised transition-colors hover:text-accent"
        >
          <IconUpload className="h-4 w-4" />
          {uploading ? "Uploading…" : "Add PDFs"}
        </label>
        <input
          ref={inputRef}
          id={uploadInputId}
          type="file"
          accept="application/pdf"
          multiple
          disabled={uploading}
          onChange={(e) => {
            onUpload(e.target.files);
            e.target.value = "";
          }}
          className="sr-only"
        />
        <p className="mt-1.5 text-center text-[12px] text-ink-subtle">
          Text-based PDFs, up to 30 MB
        </p>
        {uploadError && (
          <p role="alert" className="mt-2 text-[13px] text-danger">
            {uploadError}
          </p>
        )}
      </div>

      <div className="flex min-h-0 flex-1 flex-col border-t border-border/70 pt-3">
        <div className="flex items-center justify-between gap-2 px-4 pb-2">
          <h2 className="text-[12px] font-semibold uppercase tracking-wider text-ink-subtle">
            Documents
          </h2>
          <button
            type="button"
            onClick={onOpenLibrary}
            aria-current={libraryOpen ? "page" : undefined}
            className={`rounded px-1.5 py-0.5 text-[12px] font-medium transition-colors ${libraryOpen ? "bg-accent-soft text-accent" : "text-ink-subtle hover:bg-rail-hover hover:text-accent"}`}
            title={`${readyCount} ready documents · ${totalChunks} indexed chunks`}
          >
            View all · {readyCount}
          </button>
        </div>
        <ul className="scroll-area max-h-[48%] overflow-y-auto px-2 pb-3" aria-label="Documents">
        {documents.map((doc) => {
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
                  <span className="flex items-start gap-2">
                    <IconDoc className={`mt-0.5 h-4 w-4 shrink-0 ${isActive ? "text-accent" : "text-ink-subtle"}`} />
                    <span className={`line-clamp-2 text-[12.5px] font-medium leading-[1.4] ${isActive ? "text-accent" : "text-ink"}`}>
                      {doc.title}
                    </span>
                  </span>
                  <span className="mt-1 flex items-center gap-2 text-[12px] text-ink-subtle">
                    <span className="font-mono">{doc.doc_key}</span>
                    {doc.n_pages ? <span>{doc.n_pages}p</span> : null}
                    <StatusBadge status={doc.status} />
                  </span>
                </button>
                {doc.error && <p className="mt-1 text-[12px] text-danger">{doc.error}</p>}
                {doc.source === "upload" && (
                  <button
                    type="button"
                    onClick={async () => {
                      onDeleteDocument(doc);
                    }}
                    className="mt-1 text-[12px] text-ink-subtle underline opacity-0 transition-opacity hover:text-danger focus-visible:opacity-100 group-hover:opacity-100"
                  >
                    Remove <span className="sr-only">{doc.title}</span>
                  </button>
                )}
              </div>
            </li>
          );
        })}
        {documents.length === 0 && !uploadError && (
          <li className="px-2 py-4 text-[13px] text-ink-subtle">
            No documents yet. Upload a PDF to get started.
          </li>
        )}
        </ul>

        <div className="mx-3 border-t border-border/70" />
        <div className="flex items-center gap-2 px-4 pb-2 pt-3">
          <IconHistory className="h-3.5 w-3.5 text-ink-subtle" />
          <h2 className="text-[12px] font-semibold uppercase tracking-wider text-ink-subtle">
            Past chats
          </h2>
        </div>
        <ul className="scroll-area min-h-0 flex-1 overflow-y-auto px-2 pb-3" aria-label="Past chats">
          {conversations.map((conversation) => {
            const active = conversation.id === activeConversationId;
            return (
              <li key={conversation.id}>
                <button
                  type="button"
                  onClick={() => onOpenConversation(conversation.id)}
                  aria-current={active ? "page" : undefined}
                  className={`w-full truncate rounded-lg px-2.5 py-2 text-left text-[12.5px] transition-colors ${active ? "bg-surface font-medium text-accent shadow-raised" : "text-ink-muted hover:bg-rail-hover hover:text-ink"}`}
                  title={conversation.title || `Chat ${conversation.id}`}
                >
                  {conversation.title || `Chat ${conversation.id}`}
                </button>
              </li>
            );
          })}
          {conversations.length === 0 && (
            <li className="px-2 py-3 text-[13px] text-ink-subtle">Your conversations will appear here.</li>
          )}
        </ul>
      </div>
    </div>
  );
}
