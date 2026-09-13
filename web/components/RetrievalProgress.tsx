"use client";

import type { StreamMeta } from "@/lib/types";

/**
 * Progress shown between submitting a question and the first token.
 *
 * Retrieval takes roughly a second, dominated by the cross-encoder (~1.0s of a ~1.1s total), so
 * without this the UI sits silent long enough to read as broken. The stages come from the real
 * `timings_ms` in the `meta` event rather than a fake animation — which also means the panel
 * doubles as a live view of where latency actually goes.
 */

const STAGE_LABEL: Record<string, string> = {
  embed_query: "Embedded the question",
  sql: "Searched vectors and keywords",
  rerank: "Reranked candidates",
};

export function RetrievalProgress({
  meta,
  documentCount,
}: {
  meta?: StreamMeta;
  documentCount: number;
}) {
  // Before `meta` arrives the server is still retrieving.
  if (!meta) {
    return (
      <p
        className="flex items-center gap-2 text-[13px] text-ink-muted"
        role="status"
        aria-live="polite"
      >
        <span
          aria-hidden="true"
          className="inline-block h-2 w-2 animate-pulse rounded-full bg-accent"
        />
        Searching {documentCount} {documentCount === 1 ? "document" : "documents"}…
      </p>
    );
  }

  const stages = Object.entries(meta.timings_ms).filter(([key]) => key in STAGE_LABEL);

  return (
    <div role="status" aria-live="polite">
      <ul className="space-y-1">
        {stages.map(([key, ms]) => (
          <li key={key} className="flex items-center gap-2 text-[13px] text-ink-muted">
            <span aria-hidden="true" className="text-positive">
              ✓
            </span>
            {STAGE_LABEL[key]}
            <span className="font-mono text-[12px] text-ink-subtle">{Math.round(ms)}ms</span>
          </li>
        ))}
        <li className="flex items-center gap-2 text-[13px] text-ink-muted">
          <span
            aria-hidden="true"
            className="inline-block h-2 w-2 animate-pulse rounded-full bg-accent"
          />
          Writing the answer from {meta.config.top_k} passages…
        </li>
      </ul>
    </div>
  );
}
