"use client";

import { IconClose, IconDoc, IconExternal } from "@/components/icons";
import { documentFileUrl } from "@/lib/api";
import type { Citation, Document } from "@/lib/types";

/**
 * Right-hand pane: the evidence behind a citation.
 *
 * Shows the exact quoted passage *next to* the PDF opened at the cited page. Page-level
 * attribution plus the verbatim snippet is what lets a reader verify a claim, and it is also the
 * granularity the benchmark labels — so the UI and the evaluation agree on what a citation means.
 *
 * Span-level highlighting inside the PDF was scoped out: PDF.js text layers split spans
 * mid-word and rarely match extracted text exactly, so reliable highlighting is hours of
 * high-risk work for a detail that is glanced at.
 */
export function EvidencePane({
  citation,
  document: doc,
  onClose,
}: {
  citation: Citation | null;
  document: Document | null;
  onClose: () => void;
}) {
  const target = citation
    ? { id: citation.document_id, page: citation.page_start, title: citation.title }
    : doc
      ? { id: doc.id, page: 1, title: doc.title }
      : null;

  if (!target) {
    return (
      <div className="flex h-full flex-col items-center justify-center bg-rail px-8 text-center">
        <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-surface text-ink-subtle shadow-raised">
          <IconDoc className="h-5 w-5" />
        </span>
        <p className="mt-3 text-[13px] font-medium text-ink-muted">No source open</p>
        <p className="mt-1 max-w-[26ch] text-[12px] leading-relaxed text-ink-subtle">
          Select a citation in an answer, or a document on the left, to read the source here.
        </p>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col bg-rail">
      <div className="flex items-start justify-between gap-2 px-4 pb-3 pt-4">
        <div className="min-w-0">
          <h2 className="truncate text-[13px] font-semibold" title={target.title}>
            {target.title}
          </h2>
          <p className="mt-0.5 text-[12px] text-ink-subtle">
            {citation ? (
              <>
                Page {citation.page_start}
                {citation.page_end !== citation.page_start ? `–${citation.page_end}` : ""}
                {citation.section ? ` · ${citation.section}` : ""}
              </>
            ) : (
              "Full document"
            )}
          </p>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="shrink-0 rounded-md p-1.5 text-ink-subtle transition-colors hover:bg-rail-hover hover:text-ink"
        >
          <IconClose />
          <span className="sr-only">Close source panel</span>
        </button>
      </div>

      {citation && (
        <blockquote className="mx-4 mb-3 rounded-xl bg-surface px-4 py-3 shadow-raised">
          <p className="text-[10px] font-semibold uppercase tracking-wider text-accent">
            Cited passage
          </p>
          {/* The accent rule ties the quote to the citation chip that opened it. */}
          <p className="mt-2 border-l-2 border-accent pl-3 text-[13px] leading-[1.6] text-ink">
            {citation.snippet}…
          </p>
        </blockquote>
      )}

      <div className="mx-4 mb-3 flex-1 overflow-hidden rounded-xl bg-surface shadow-raised">
        {/* The browser's native PDF viewer, opened at the cited page via a #page fragment.
            Titled for screen readers; a direct link is offered as a fallback because embedded
            PDF viewing is inconsistent across browsers. */}
        <iframe
          key={`${target.id}-${target.page}`}
          src={documentFileUrl(target.id, target.page)}
          title={`${target.title}, page ${target.page}`}
          className="h-full w-full border-0"
        />
      </div>

      <div className="px-4 pb-3">
        <a
          href={documentFileUrl(target.id, target.page)}
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-1.5 text-[12px] text-accent hover:underline"
        >
          <IconExternal className="h-3.5 w-3.5" />
          Open PDF in a new tab
        </a>
      </div>
    </div>
  );
}
