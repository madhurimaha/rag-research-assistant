"use client";

import { useEffect, useRef, useState } from "react";
import { IconClose } from "@/components/icons";
import { getExplain } from "@/lib/api";
import { summarizeTrace } from "@/lib/trace-summary";
import type { RetrievalConfig, TraceRow } from "@/lib/types";

/**
 * "Why this answer?" — the retrieval trace for one message.
 *
 * Two audiences, two tiers. The summary answers the question the button asks, in words, for
 * anyone reading an answer. The candidate table below it is a diagnostic surface: `rrf_score`
 * and `rerank_score` are internal and unnormalized, and it exposes chunk ids and retriever
 * internals, so it is gated behind NEXT_PUBLIC_SHOW_DIAGNOSTICS and would sit behind an
 * operator role in a multi-user deployment.
 *
 * The table is worth keeping for a reviewer because the interesting cases are only visible
 * there: a chunk ranked first by both arms can still be demoted by the reranker, which is the
 * clearest available argument for the two-stage design.
 *
 * Implemented as a modal dialog with focus trapping and Escape-to-close, rather than pulling in
 * a dialog library for a single surface.
 */

/** Defaults on, so a fresh clone shows a reviewer the whole pipeline. */
const SHOW_DIAGNOSTICS = process.env.NEXT_PUBLIC_SHOW_DIAGNOSTICS !== "false";
export function ExplainDialog({
  messageId,
  config,
  onClose,
}: {
  messageId: number;
  config?: RetrievalConfig;
  onClose: () => void;
}) {
  const [rows, setRows] = useState<TraceRow[] | null>(null);
  const [question, setQuestion] = useState("");
  const [error, setError] = useState<string | null>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    getExplain(messageId)
      .then((data) => {
        setRows(data.candidates);
        setQuestion(data.question);
      })
      .catch(() => setError("Could not load the retrieval trace."));
  }, [messageId]);

  // Move focus into the dialog on open, and restore it to the trigger on close.
  useEffect(() => {
    const previous= document.activeElement as HTMLElement | null;
    closeRef.current?.focus();
    return () => previous?.focus();
  }, []);

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") {
        onClose();
        return;
      }
      if (e.key !== "Tab" || !panelRef.current) return;
      // Keep Tab inside the dialog while it is open.
      const focusable = panelRef.current.querySelectorAll<HTMLElement>(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
      );
      if (focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    }
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  const used = rows?.filter((r) => r.used_in_context).length ?? 0;
  const summary = rows ? summarizeTrace(rows, config) : null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-[rgb(26_24_21/0.36)] p-4 backdrop-blur-[2px]"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="explain-title"
        className="flex max-h-[85vh] w-full max-w-4xl flex-col rounded-2xl bg-surface shadow-overlay"
      >
        <div className="flex items-start justify-between gap-4 border-b border-border px-5 py-3.5">
          <div className="min-w-0">
            <h2 id="explain-title" className="text-[15px] font-semibold tracking-[-0.011em]">
              Why this answer
            </h2>
            <p className="mt-0.5 truncate text-[12px] text-ink-subtle">{question}</p>
          </div>
          <button
            ref={closeRef}
            type="button"
            onClick={onClose}
            className="shrink-0 rounded-md p-1.5 text-ink-subtle transition-colors hover:bg-rail hover:text-ink"
          >
            <IconClose />
            <span className="sr-only">Close</span>
          </button>
        </div>

        {config && (
          <div className="flex flex-wrap gap-x-4 gap-y-1 border-b border-border bg-sunken px-5 py-2.5 text-[12px] text-ink-muted">
            <span>
              Retrieval:{" "}
              <strong className="font-semibold">
                {config.use_hybrid ? "hybrid (vector + keyword)" : "vector only"}
              </strong>
            </span>
            {config.use_hybrid && <span>RRF k={config.rrf_k}</span>}
            <span>
              Reranker:{" "}
              <strong className="font-semibold">
                {config.use_rerank ? `top ${config.rerank_depth}` : "off"}
              </strong>
            </span>
            <span>{config.candidates_per_arm} candidates per arm</span>
            <span>{used} chunks used</span>
          </div>
        )}

        {/* tabIndex makes the scrollable trace reachable by keyboard: the table is taller than
            the dialog, and without it a keyboard user cannot scroll to the lower candidates. */}
        <div
          className="scroll-area flex-1 overflow-auto px-5 py-3"
          role="group"
          aria-label="Answer explanation"
          tabIndex={0}
        >
          {error && (
            <p role="alert" className="text-sm text-danger">
              {error}
            </p>
          )}
          {!rows && !error && (
            <p className="text-sm text-ink-subtle">Loading trace…</p>
          )}

          {rows && summary && (
            <section className="max-w-[74ch]">
              <p className="text-[14px] font-medium leading-relaxed text-ink">
                {summary.headline}
              </p>
              {summary.points.length > 0 && (
                <ul className="mt-2.5 space-y-1.5">
                  {summary.points.map((point) => (
                    <li
                      key={point}
                      className="flex gap-2 text-[13px] leading-relaxed text-ink-muted"
                    >
                      <span aria-hidden="true" className="mt-[7px] h-1 w-1 shrink-0 rounded-full bg-border-strong" />
                      {point}
                    </li>
                  ))}
                </ul>
              )}
            </section>
          )}

          {rows && SHOW_DIAGNOSTICS && (
            <>
              <h3 className="mt-6 text-[11px] font-semibold uppercase tracking-wider text-ink-subtle">
                Retrieval diagnostics
              </h3>
              <p className="mt-1.5 mb-3 max-w-[74ch] text-[12.5px] leading-relaxed text-ink-muted">
                Each row is a chunk the retriever considered. The two arms rank independently,
                Reciprocal Rank Fusion merges them by position, then a cross-encoder rescores the
                top candidates. Only the highest-scoring chunks are sent to the model.{" "}
                <span className="text-ink-subtle">
                  Scores are internal and unnormalized — compare them within a row, not against
                  a fixed scale.
                </span>
              </p>
              <table className="w-full border-collapse text-[12px]">
                <caption className="sr-only">
                  Retrieval candidates with per-stage ranks and scores
                </caption>
                <thead>
                  <tr className="border-b border-border text-left text-ink-subtle">
                    <th scope="col" className="py-1.5 pr-2 font-medium">
                      Used
                    </th>
                    <th scope="col" className="py-1.5 pr-2 font-medium">
                      Source
                    </th>
                    <th scope="col" className="py-1.5 pr-2 text-right font-medium">
                      Vector
                    </th>
                    <th scope="col" className="py-1.5 pr-2 text-right font-medium">
                      Keyword
                    </th>
                    <th scope="col" className="py-1.5 pr-2 text-right font-medium">
                      RRF
                    </th>
                    <th scope="col" className="py-1.5 pr-2 text-right font-medium">
                      Rerank
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {rows.slice(0, 25).map((row) => (
                    <tr
                      key={row.chunk_id}
                      className={`border-b border-border align-top ${
                        row.used_in_context ? "bg-accent-soft" : ""
                      }`}
                    >
                      <td className="py-1.5 pr-2 whitespace-nowrap">
                        {row.used_in_context ? (
                          <span className="font-semibold text-accent">
                            #{row.final_rank}
                          </span>
                        ) : (
                          <span className="text-ink-subtle">—</span>
                        )}
                      </td>
                      <td className="max-w-[300px] py-1.5 pr-2">
                        <span className="font-mono text-[10px] text-ink-subtle">
                          {row.doc_key} p{row.page_start}
                        </span>
                        <span className="block truncate text-ink-muted">
                          {row.snippet}
                        </span>
                      </td>
                      <td className="py-1.5 pr-2 text-right font-mono whitespace-nowrap">
                        {row.vector_rank ? (
                          <>
                            #{row.vector_rank}
                            <span className="ml-1 text-[10px] text-ink-subtle">
                              {row.vector_score?.toFixed(3)}
                            </span>
                          </>
                        ) : (
                          <span className="text-ink-subtle">—</span>
                        )}
                      </td>
                      <td className="py-1.5 pr-2 text-right font-mono whitespace-nowrap">
                        {row.lexical_rank ? (
                          `#${row.lexical_rank}`
                        ) : (
                          <span className="text-ink-subtle">—</span>
                        )}
                      </td>
                      <td className="py-1.5 pr-2 text-right font-mono">
                        {row.rrf_score.toFixed(4)}
                      </td>
                      <td className="py-1.5 pr-2 text-right font-mono">
                        {row.rerank_score !== null ? (
                          row.rerank_score.toFixed(2)
                        ) : (
                          <span className="text-ink-subtle">—</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {rows.length > 25 && (
                <p className="mt-2 text-[11px] text-ink-subtle">
                  Showing the top 25 of {rows.length} candidates.
                </p>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
