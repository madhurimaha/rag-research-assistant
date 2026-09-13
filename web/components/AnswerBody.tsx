"use client";

import type { Citation } from "@/lib/types";

/**
 * Renders a generated answer.
 *
 * The model writes prose but reaches for numbered lists on "list the ..." questions, separating
 * items with blank lines. HTML collapses that whitespace, so rendering the answer as a single
 * paragraph turns a four-item list into one run-on sentence. This does the minimum block-level
 * parsing needed — paragraphs, ordered and unordered lists — rather than pulling in a Markdown
 * renderer, because the citation markers need custom inline handling either way.
 */

export interface ListItem {
  /** The number the model wrote, so rendering never renumbers its list. */
  value?: number;
  text: string;
}

type Block =
  | { kind: "para"; text: string }
  | { kind: "list"; ordered: boolean; items: ListItem[] };

const ORDERED_ITEM = /^\s*(\d+)[.)]\s+(.*)$/;
const UNORDERED_ITEM = /^\s*[-*\u2022]\s+(.*)$/;

export function toBlocks(text: string): Block[] {
  const blocks: Block[] = [];
  let para: string[] = [];
  let list: { ordered: boolean; items: ListItem[] } | null = null;
  // The model writes "loose" lists, separating items with a blank line. A blank line therefore
  // cannot end a list; it only ends a paragraph, and marks that any following plain text starts
  // a new paragraph rather than continuing the previous item.
  let blankPending = false;

  const flushPara = () => {
    if (para.length) {
      blocks.push({ kind: "para", text: para.join(" ").trim() });
      para = [];
    }
  };
  const flushList = () => {
    if (list) {
      blocks.push({ kind: "list", ordered: list.ordered, items: list.items });
      list = null;
    }
  };

  for (const raw of text.split("\n")) {
    const line = raw.trim();

    if (!line) {
      flushPara();
      blankPending = true;
      continue;
    }

    const ordered = line.match(ORDERED_ITEM);
    const unordered = line.match(UNORDERED_ITEM);

    if (ordered || unordered) {
      const isOrdered = Boolean(ordered);
      flushPara();
      if (list && list.ordered !== isOrdered) flushList();
      if (!list) list = { ordered: isOrdered, items: [] };
      list.items.push(
        isOrdered
          ? { value: Number(ordered![1]), text: ordered![2] }
          : { text: unordered![1] },
      );
      blankPending = false;
      continue;
    }

    // A plain line right after an item (no blank between) continues that item.
    if (list && !blankPending) {
      list.items[list.items.length - 1].text += ` ${line}`;
      continue;
    }

    flushList();
    para.push(line);
    blankPending = false;
  }

  flushPara();
  flushList();
  return blocks;
}

/** Bracket groups the model produces: [1], [1,2], [1-3], [1; 2]. */
const CITATION_GROUP = /(\[[\d\s,;\u2013\u2014-]+\])/g;
const BOLD = /(\*\*[^*]+\*\*)/g;

function expandMarkers(inner: string): number[] {
  return inner
    .split(/[,;]/)
    .flatMap((piece) => {
      const range = piece.trim().match(/^(\d+)\s*[\u2013\u2014-]\s*(\d+)$/);
      if (range) {
        const from = Number(range[1]);
        const to = Number(range[2]);
        if (to < from || to - from > 50) return [];
        return Array.from({ length: to - from + 1 }, (_, k) => from + k);
      }
      const n = Number(piece.trim());
      return Number.isInteger(n) ? [n] : [];
    });
}

function Emphasis({ text }: { text: string }) {
  return (
    <>
      {text.split(BOLD).map((part, i) =>
        part.startsWith("**") && part.endsWith("**") && part.length > 4 ? (
          <strong key={i} className="font-semibold">
            {part.slice(2, -2)}
          </strong>
        ) : (
          <span key={i}>{part}</span>
        ),
      )}
    </>
  );
}

/**
 * Inline run: turns [n] markers into buttons.
 *
 * Citations are the primary verification path, so they are real buttons — focusable and
 * keyboard-activated. Styling spans instead would make verification mouse-only.
 */
function Inline({
  text,
  citations,
  onCite,
}: {
  text: string;
  citations: Citation[];
  onCite: (c: Citation) => void;
}) {
  const byMarker = new Map(citations.map((c) => [c.marker, c]));

  return (
    <>
      {text.split(CITATION_GROUP).map((part, i) => {
        const group = part.match(/^\[([\d\s,;\u2013\u2014-]+)\]$/);
        if (!group) return <Emphasis key={i} text={part} />;

        const markers = expandMarkers(group[1]).filter((n) => byMarker.has(n));
        // An out-of-range marker is left as literal text rather than silently dropped.
        if (markers.length === 0) return <span key={i}>{part}</span>;

        return (
          <span key={i} className="whitespace-nowrap">
            {markers.map((marker) => {
              const c = byMarker.get(marker)!;
              return (
                <button
                  key={marker}
                  type="button"
                  onClick={() => onCite(c)}
                  title={`${c.title} — page ${c.page_start}`}
                  className="mx-[1px] inline-flex h-[18px] min-w-[18px] items-center justify-center rounded border border-accent bg-accent-soft px-1 align-[1px] text-[11px] font-semibold text-accent transition-colors hover:bg-accent hover:text-white"
                >
                  {marker}
                  <span className="sr-only">
                    : open source {marker}, {c.title}, page {c.page_start}
                  </span>
                </button>
              );
            })}
          </span>
        );
      })}
    </>
  );
}

export function AnswerBody({
  text,
  citations,
  onCite,
  streaming,
}: {
  text: string;
  citations: Citation[];
  onCite: (c: Citation) => void;
  streaming: boolean;
}) {
  const blocks = toBlocks(text);

  return (
    <div className="space-y-3 text-[15px] leading-relaxed">
      {blocks.map((block, bi) => {
        const isLast = bi === blocks.length - 1;
        const caret = streaming && isLast ? "streaming-caret" : "";

        if (block.kind === "para") {
          return (
            <p key={bi} className={caret}>
              <Inline text={block.text} citations={citations} onCite={onCite} />
            </p>
          );
        }

        const ListTag = block.ordered ? "ol" : "ul";
        return (
          <ListTag
            key={bi}
            start={block.ordered ? block.items[0]?.value : undefined}
            className={`space-y-2 pl-5 ${
              block.ordered ? "list-decimal" : "list-disc"
            } marker:text-ink-subtle`}
          >
            {block.items.map((item, ii) => (
              <li
                key={ii}
                value={item.value}
                className={ii === block.items.length - 1 ? caret : undefined}
              >
                <Inline text={item.text} citations={citations} onCite={onCite} />
              </li>
            ))}
          </ListTag>
        );
      })}
    </div>
  );
}
