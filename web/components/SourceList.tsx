"use client";

import type { Citation } from "@/lib/types";

/**
 * Sources for one answer, grouped by document.
 *
 * A flat chip per citation is unreadable in practice: answers routinely cite four or five
 * passages from the same paper, so the chips truncate to the same string and differ only by a
 * page number. Grouping gives one row per document — markers, then title, then the pages — so
 * "five passages from one paper" is distinguishable at a glance from "five papers".
 */
export function SourceList({
  citations,
  onCite,
}: {
  citations: Citation[];
  onCite: (c: Citation) => void;
}) {
  if (citations.length === 0) return null;

  const groups = new Map<number, { title: string; items: Citation[] }>();
  for (const c of citations) {
    const group = groups.get(c.document_id) ?? { title: c.title, items: [] };
    group.items.push(c);
    groups.set(c.document_id, group);
  }

  const docCount = groups.size;

  return (
    <section className="mt-4 border-t border-border pt-3">
      <h4 className="text-[12px] font-semibold uppercase tracking-wide text-ink-subtle">
        Sources · {citations.length} {citations.length === 1 ? "passage" : "passages"} from{" "}
        {docCount} {docCount === 1 ? "document" : "documents"}
      </h4>
      <ul className="mt-2 space-y-1.5">
        {[...groups.entries()].map(([docId, group]) => {
          const ordered = [...group.items].sort((a, b) => a.marker - b.marker);
          return (
            <li key={docId} className="flex items-start gap-2">
              <span className="flex shrink-0 gap-1 pt-[1px]">
                {ordered.map((c) => (
                  <button
                    key={c.marker}
                    type="button"
                    onClick={() => onCite(c)}
                    title={`${c.title} — page ${c.page_start}`}
                    className="inline-flex h-5 min-w-5 items-center justify-center rounded border border-accent bg-accent-soft px-1 text-[13px] font-semibold text-accent transition-colors hover:bg-accent hover:text-white"
                  >
                    {c.marker}
                    <span className="sr-only">
                      : open page {c.page_start} of {c.title}
                    </span>
                  </button>
                ))}
              </span>
              <span className="min-w-0 flex-1 text-[13px] leading-[1.45] text-ink-muted">
                <span className="line-clamp-2" title={group.title}>
                  {group.title}
                </span>
                <span className="text-ink-subtle">
                  {ordered.map((c) => `p${c.page_start}`).join(", ")}
                  {ordered[0].section ? ` · ${ordered[0].section}` : ""}
                </span>
              </span>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
