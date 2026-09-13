import { describe, expect, it } from "vitest";
import { summarizeTrace } from "@/lib/trace-summary";
import type { RetrievalConfig, TraceRow } from "@/lib/types";

function row(over: Partial<TraceRow> & { chunk_id: number }): TraceRow {
  return {
    doc_key: "1908.06606",
    title: "A paper",
    page_start: 1,
    section: null,
    snippet: "…",
    vector_rank: null,
    vector_score: null,
    lexical_rank: null,
    lexical_score: null,
    rrf_score: 0.01,
    rrf_rank: 99,
    rerank_score: null,
    final_rank: null,
    used_in_context: false,
    ...over,
  };
}

const config = (over: Partial<RetrievalConfig> = {}): RetrievalConfig => ({
  use_hybrid: true,
  use_rerank: true,
  rerank_depth: 20,
  candidates_per_arm: 50,
  rrf_k: 60,
  hnsw_ef_search: 100,
  top_k: 6,
  contextualized_index: false,
  carried_forward: 0,
  ...over,
});

describe("summarizeTrace", () => {
  it("returns null with no trace rows", () => {
    expect(summarizeTrace([], config())).toBeNull();
  });

  it("reports scope as used-of-retrieved across documents", () => {
    const rows = [
      row({ chunk_id: 1, used_in_context: true, final_rank: 1, rrf_rank: 1, doc_key: "A" }),
      row({ chunk_id: 2, used_in_context: true, final_rank: 2, rrf_rank: 2, doc_key: "B" }),
      row({ chunk_id: 3, rrf_rank: 3 }),
    ];
    expect(summarizeTrace(rows, config())!.headline).toBe(
      "Drew on 2 of 3 retrieved passages, from 2 documents.",
    );
  });

  it("uses singular wording for one passage from one document", () => {
    const rows = [row({ chunk_id: 1, used_in_context: true, final_rank: 1, rrf_rank: 1 })];
    expect(summarizeTrace(rows, config())!.headline).toBe(
      "Drew on 1 of 1 retrieved passage, from 1 document.",
    );
  });

  it("handles an answer where nothing was used", () => {
    const rows = [row({ chunk_id: 1 }), row({ chunk_id: 2 })];
    const s = summarizeTrace(rows, config())!;
    expect(s.headline).toContain("none of which met the bar");
    expect(s.points).toEqual([]);
  });

  it("breaks down which arm found each used passage", () => {
    const rows = [
      row({ chunk_id: 1, used_in_context: true, final_rank: 1, rrf_rank: 1, vector_rank: 1, lexical_rank: 2 }),
      row({ chunk_id: 2, used_in_context: true, final_rank: 2, rrf_rank: 2, vector_rank: 3 }),
      row({ chunk_id: 3, used_in_context: true, final_rank: 3, rrf_rank: 3, lexical_rank: 1 }),
    ];
    const points = summarizeTrace(rows, config())!.points;
    expect(points[0]).toBe(
      "Found 1 by both meaning-based and keyword search, 1 by meaning-based search alone, and 1 by keyword search alone.",
    );
    expect(points[1]).toContain("Keyword search contributed 1 passage that meaning-based search missed");
  });

  /** With the lexical arm off, "0 by keyword search" would be misleading rather than informative. */
  it("omits arm breakdown when hybrid retrieval is disabled", () => {
    const rows = [
      row({ chunk_id: 1, used_in_context: true, final_rank: 1, rrf_rank: 1, vector_rank: 1 }),
    ];
    const points = summarizeTrace(rows, config({ use_hybrid: false }))!.points;
    expect(points.some((p) => p.includes("keyword"))).toBe(false);
  });

  /** A straight exchange reads as a contradiction if described as two independent moves. */
  it("describes a first/third exchange as a swap", () => {
    const rows = [
      row({ chunk_id: 1, used_in_context: true, rrf_rank: 1, final_rank: 3, rerank_score: 5.0, vector_rank: 1 }),
      row({ chunk_id: 2, used_in_context: true, rrf_rank: 3, final_rank: 1, rerank_score: 7.7, vector_rank: 4 }),
    ];
    const points = summarizeTrace(rows, config())!.points;
    expect(points.some((p) => p.includes("swapping the first and third results"))).toBe(true);
  });

  it("describes a non-exchange reorder as a promotion and a demotion", () => {
    const rows = [
      row({ chunk_id: 1, used_in_context: true, rrf_rank: 1, final_rank: 4, rerank_score: 3.0 }),
      row({ chunk_id: 2, used_in_context: true, rrf_rank: 2, final_rank: 1, rerank_score: 7.7 }),
    ];
    const points = summarizeTrace(rows, config())!.points;
    expect(points.some((p) =>
      p.includes("promoted the passage combined search placed second") &&
      p.includes("moving that first choice down to fourth"),
    )).toBe(true);
  });

  it("says the top fused passage was dropped when it did not survive", () => {
    const rows = [
      row({ chunk_id: 1, rrf_rank: 1, rerank_score: -2.0 }),
      row({ chunk_id: 2, used_in_context: true, rrf_rank: 2, final_rank: 1, rerank_score: 7.7 }),
    ];
    const points = summarizeTrace(rows, config())!.points;
    expect(points.some((p) => p.includes("dropping that first choice from the final set"))).toBe(
      true,
    );
  });

  it("omits rerank commentary when the reranker did not run", () => {
    const rows = [
      row({ chunk_id: 1, used_in_context: true, rrf_rank: 1, final_rank: 1, vector_rank: 1 }),
      row({ chunk_id: 2, used_in_context: true, rrf_rank: 2, final_rank: 2, vector_rank: 2 }),
    ];
    const points = summarizeTrace(rows, config({ use_rerank: false }))!.points;
    expect(points.some((p) => p.toLowerCase().includes("rerank"))).toBe(false);
  });

  it("counts passages promoted from below the cut", () => {
    const rows = [
      row({ chunk_id: 1, used_in_context: true, rrf_rank: 1, final_rank: 1, rerank_score: 9 }),
      row({ chunk_id: 2, used_in_context: true, rrf_rank: 12, final_rank: 2, rerank_score: 8 }),
    ];
    const points = summarizeTrace(rows, config())!.points;
    expect(points.some((p) => p.includes("1 of the passages used was outside the top 2"))).toBe(true);
  });

  it("mentions carried-forward context only when it happened", () => {
    const rows = [row({ chunk_id: 1, used_in_context: true, rrf_rank: 1, final_rank: 1 })];
    expect(
      summarizeTrace(rows, config({ carried_forward: 3 }))!.points.some((p) =>
        p.includes("3 passages were carried over"),
      ),
    ).toBe(true);
    expect(
      summarizeTrace(rows, config({ carried_forward: 0 }))!.points.some((p) =>
        p.includes("carried over"),
      ),
    ).toBe(false);
  });

  it("works with no config supplied", () => {
    const rows = [row({ chunk_id: 1, used_in_context: true, rrf_rank: 1, final_rank: 1 })];
    expect(() => summarizeTrace(rows)).not.toThrow();
    expect(summarizeTrace(rows)).not.toBeNull();
  });
});
