import { describe, expect, it } from "vitest";
import { sourcesUsedInAnswer } from "./citations";
import type { Citation } from "./types";

function cite(chunk_id: number, marker: number): Citation {
  return {
    marker,
    chunk_id,
    document_id: 1,
    doc_key: "paper",
    title: "A Paper",
    page_start: marker,
    page_end: marker,
    section: null,
    snippet: "passage",
  };
}

const retrieved = [cite(10, 1), cite(20, 2), cite(30, 3)];

describe("sourcesUsedInAnswer", () => {
  it("keeps every retrieved passage while the stream is still open", () => {
    expect(sourcesUsedInAnswer(retrieved, undefined)).toEqual(retrieved);
  });

  it("keeps only chunks the answer cited", () => {
    expect(sourcesUsedInAnswer(retrieved, [30, 10]).map((c) => c.chunk_id)).toEqual([
      10, 30,
    ]);
  });

  it("hides the list when the model cited nothing", () => {
    expect(sourcesUsedInAnswer(retrieved, [])).toEqual([]);
  });
});
