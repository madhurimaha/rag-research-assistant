import { describe, expect, it } from "vitest";
import { toBlocks } from "@/components/AnswerBody";

describe("toBlocks", () => {
  it("keeps a single paragraph intact", () => {
    expect(toBlocks("The QA-CTS task unifies clinical text structuring [1].")).toEqual([
      { kind: "para", text: "The QA-CTS task unifies clinical text structuring [1]." },
    ]);
  });

  it("splits paragraphs on a blank line", () => {
    const blocks = toBlocks("First claim [1].\n\nSecond claim [2].");
    expect(blocks).toHaveLength(2);
    expect(blocks.every((b) => b.kind === "para")).toBe(true);
  });

  it("joins hard-wrapped lines into one paragraph", () => {
    expect(toBlocks("a sentence that was\nwrapped across lines")).toEqual([
      { kind: "para", text: "a sentence that was wrapped across lines" },
    ]);
  });

  /**
   * The regression that mattered: the model writes "loose" lists with a blank line between
   * items. Flushing the list on each blank line produced one <ol> per item, so every item
   * rendered as "1.".
   */
  it("keeps a loose list as one list, preserving the model's numbering", () => {
    const blocks = toBlocks(
      "The datasets are:\n\n1. COSTRA 1.0: Czech sentences [1].\n\n2. OLID: offensive language [2].\n\n3. Waseem: hate speech [3].",
    );
    expect(blocks).toHaveLength(2);
    expect(blocks[0]).toEqual({ kind: "para", text: "The datasets are:" });
    expect(blocks[1]).toEqual({
      kind: "list",
      ordered: true,
      items: [
        { value: 1, text: "COSTRA 1.0: Czech sentences [1]." },
        { value: 2, text: "OLID: offensive language [2]." },
        { value: 3, text: "Waseem: hate speech [3]." },
      ],
    });
  });

  it("handles a tight list with no blank lines", () => {
    const blocks = toBlocks("1. one\n2. two");
    expect(blocks).toEqual([
      {
        kind: "list",
        ordered: true,
        items: [
          { value: 1, text: "one" },
          { value: 2, text: "two" },
        ],
      },
    ]);
  });

  it("continues an item across a wrapped line but not across a blank line", () => {
    const blocks = toBlocks("1. first part\ncontinued here\n\nA new paragraph.");
    expect(blocks).toEqual([
      { kind: "list", ordered: true, items: [{ value: 1, text: "first part continued here" }] },
      { kind: "para", text: "A new paragraph." },
    ]);
  });

  it("recognises bulleted lists and switches type", () => {
    const blocks = toBlocks("- alpha\n- beta\n\n1. one");
    expect(blocks).toEqual([
      { kind: "list", ordered: false, items: [{ text: "alpha" }, { text: "beta" }] },
      { kind: "list", ordered: true, items: [{ value: 1, text: "one" }] },
    ]);
  });

  it("respects numbering the model restarts or skips", () => {
    const blocks = toBlocks("3. three\n\n5. five");
    expect(blocks[0]).toEqual({
      kind: "list",
      ordered: true,
      items: [
        { value: 3, text: "three" },
        { value: 5, text: "five" },
      ],
    });
  });

  it("does not treat a decimal figure as a list marker", () => {
    const blocks = toBlocks("The model reached 88.4 F1 on the task [1].");
    expect(blocks).toEqual([
      { kind: "para", text: "The model reached 88.4 F1 on the task [1]." },
    ]);
  });

  it("returns nothing for empty or whitespace-only text", () => {
    expect(toBlocks("")).toEqual([]);
    expect(toBlocks("\n\n  \n")).toEqual([]);
  });

  it("is stable on partial text while streaming", () => {
    // Prefixes of a real answer must never throw, since the parser re-runs on every token.
    const full = "Datasets:\n\n1. COSTRA [1].\n\n2. OLID [2].";
    for (let i = 1; i <= full.length; i++) {
      expect(() => toBlocks(full.slice(0, i))).not.toThrow();
    }
  });
});
