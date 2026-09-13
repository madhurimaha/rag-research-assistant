import type { RetrievalConfig, TraceRow } from "./types";

/**
 * Turns a retrieval trace into plain-language observations.
 *
 * The raw trace is a diagnostic surface: `rrf_score = 0.0312` is unnormalized and `rerank_score
 * = 7.72` is unbounded, so neither means anything to a reader who doesn't know `k=60`. This
 * derives the statements a person actually wants — how much of the corpus the answer leaned on,
 * whether the two search arms agreed, and whether reranking changed the outcome — from the same
 * rows, with no model call.
 *
 * Every claim is guarded: with the reranker or the lexical arm switched off, the statements that
 * depend on them are omitted rather than reported as zero.
 */

export interface TraceSummary {
  headline: string;
  points: string[];
}

export function summarizeTrace(
  rows: TraceRow[],
  config?: RetrievalConfig,
): TraceSummary | null {
  if (rows.length === 0) return null;

  const used = rows.filter((r) => r.used_in_context);
  if (used.length === 0) {
    return {
      headline: `Considered ${rows.length} passages, none of which met the bar for inclusion.`,
      points: [],
    };
  }

  const docCount = new Set(used.map((r) => r.doc_key)).size;
  const headline =
    `Drew on ${used.length} of ${rows.length} retrieved ${plural(rows.length, "passage")}, ` +
    `from ${docCount} ${plural(docCount, "document")}.`;

  const points: string[] = [];

  // How the two arms contributed. Only meaningful when the lexical arm actually ran.
  const lexicalRan = config?.use_hybrid !== false && rows.some((r) => r.lexical_rank !== null);
  if (lexicalRan) {
    const both = used.filter((r) => r.vector_rank !== null && r.lexical_rank !== null).length;
    const vectorOnly = used.filter(
      (r) => r.vector_rank !== null && r.lexical_rank === null,
    ).length;
    const lexicalOnly = used.filter(
      (r) => r.vector_rank === null && r.lexical_rank !== null,
    ).length;

    const parts: string[] = [];
    if (both) parts.push(`${both} by both meaning-based and keyword search`);
    if (vectorOnly) parts.push(`${vectorOnly} by meaning-based search alone`);
    if (lexicalOnly) parts.push(`${lexicalOnly} by keyword search alone`);
    if (parts.length) points.push(`Found ${joinList(parts)}.`);

    if (lexicalOnly > 0) {
      points.push(
        `Keyword search contributed ${lexicalOnly} ${plural(lexicalOnly, "passage")} that ` +
          `meaning-based search missed entirely — the reason both run.`,
      );
    }
  }

  // What reranking changed. Skipped unless the stage actually scored something.
  const reranked = rows.filter((r) => r.rerank_score !== null);
  if (config?.use_rerank !== false && reranked.length > 0) {
    const topFused = rows.find((r) => r.rrf_rank === 1);
    const finalFirst = used.find((r) => r.final_rank === 1);

    if (topFused && finalFirst && topFused.chunk_id !== finalFirst.chunk_id) {
      const movedTo = topFused.final_rank;
      // A straight exchange is the common case, and describing it as two separate moves reads as
      // a contradiction ("moved to third, in favour of one ranked third").
      if (movedTo && movedTo === finalFirst.rrf_rank) {
        points.push(
          `The reranker disagreed with combined search about the best source, swapping the ` +
            `first and ${ordinal(movedTo)} results.`,
        );
      } else {
        points.push(
          `The reranker disagreed with combined search about the best source: it promoted the ` +
            `passage combined search placed ${ordinal(finalFirst.rrf_rank)}` +
            (movedTo
              ? `, moving that first choice down to ${ordinal(movedTo)}.`
              : `, dropping that first choice from the final set.`),
        );
      }
    }

    // Passages that fused ranking alone would have excluded.
    const promoted = used.filter((r) => r.rrf_rank > used.length).length;
    if (promoted > 0) {
      points.push(
        `${promoted} of the passages used ${plural(promoted, "was", "were")} outside the top ` +
          `${used.length} on combined search and ${plural(promoted, "was", "were")} promoted ` +
          `by reranking.`,
      );
    }
  }

  if (config && config.carried_forward > 0) {
    points.push(
      `${config.carried_forward} ${plural(config.carried_forward, "passage")} ` +
        `${plural(config.carried_forward, "was", "were")} carried over from the previous ` +
        `answer, because this read as a follow-up question.`,
    );
  }

  return { headline, points };
}

function plural(n: number, singular: string, pluralForm?: string): string {
  if (n === 1) return singular;
  return pluralForm ?? `${singular}s`;
}

function ordinal(n: number): string {
  const names = ["", "first", "second", "third", "fourth", "fifth", "sixth"];
  return names[n] ?? `${n}th`;
}

function joinList(parts: string[]): string {
  if (parts.length === 1) return parts[0];
  if (parts.length === 2) return `${parts[0]} and ${parts[1]}`;
  return `${parts.slice(0, -1).join(", ")}, and ${parts[parts.length - 1]}`;
}
