import type { Citation } from "./types";

/**
 * SSE `citation` events are every chunk sent to the model. After `done`, only
 * `cited_chunk_ids` were actually referenced in the answer.
 */
export function sourcesUsedInAnswer(
  citations: Citation[],
  citedChunkIds: number[] | undefined,
): Citation[] {
  if (!citedChunkIds) return citations;
  const used = new Set(citedChunkIds);
  return citations.filter((c) => used.has(c.chunk_id));
}
