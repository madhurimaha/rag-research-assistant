"""PDF text extraction and chunking.

Chunking decisions
------------------
* **Page-aware.** Every chunk records the page range it came from, because citations must point
  a reader at a physical page. Chunk boundaries are therefore never allowed to silently span
  the whole document.
* **Paragraph-first, token-bounded.** Split on blank lines, then pack paragraphs up to a token
  target. This keeps sentences and usually whole paragraphs intact, which matters more for
  answer quality than hitting an exact token count.
* **Overlap.** A fixed token overlap carries the tail of the previous chunk into the next, so a
  fact that straddles a boundary is still fully present in at least one chunk.
* **Heading capture.** A best-effort section heading is attached to each chunk. It is cheap
  (regex over short title-like lines) and gives both the reranker and the reader useful context.

Research-paper specifics: reference lists are dropped. They are long, dense with proper nouns,
and act as lexical-arm magnets that crowd out real content without ever answering a question.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import tiktoken

_ENC = tiktoken.get_encoding("cl100k_base")

# Headings we treat as section starts in academic PDFs.
_HEADING_RE = re.compile(
    r"^\s*(?:(?:\d+(?:\.\d+)*\.?\s+)?"
    r"(abstract|introduction|related work|background|method(?:s|ology)?|approach|model|"
    r"experiment(?:s|al setup)?|dataset(?:s)?|result(?:s)?|analysis|discussion|ablation|"
    r"evaluation|conclusion(?:s)?|limitations|future work|acknowledg(?:e)?ments?|references)"
    r")\s*$",
    re.IGNORECASE,
)

_REFERENCES_RE = re.compile(r"^\s*(?:\d+\.?\s+)?references\s*$", re.IGNORECASE)


def n_tokens(text: str) -> int:
    return len(_ENC.encode(text, disallowed_special=()))


@dataclass
class PageText:
    page: int  # 1-indexed
    text: str


@dataclass
class Chunk:
    ordinal: int
    page_start: int
    page_end: int
    section: str | None
    text: str
    n_tokens: int


def extract_pages(pdf_path: str) -> list[PageText]:
    """Extract text per page with PyMuPDF, stopping at the reference list.

    Uses `get_text("blocks")` rather than `"text"`. This matters: `"text"` mode emits a newline
    per *visual line* with no paragraph separators, so a whole page collapses into one block and
    chunking degenerates to one-chunk-per-page. `"blocks"` returns PyMuPDF's layout-analysed
    regions, which recovers real paragraph boundaries and — usefully — isolates short headings
    into their own blocks.

    Blocks are sorted by (column, vertical position) to get correct reading order on the
    two-column layouts typical of academic papers.
    """
    import pymupdf

    pages: list[PageText] = []
    with pymupdf.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            page_width = page.rect.width
            blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]

            # Two-column detection: if blocks cluster on both sides of the midpoint, read the
            # left column top-to-bottom before the right.
            midpoint = page_width / 2
            left = [b for b in blocks if b[0] < midpoint * 0.85]
            right = [b for b in blocks if b[0] >= midpoint * 0.85]
            two_column = len(left) >= 3 and len(right) >= 3

            if two_column:
                blocks = sorted(left, key=lambda b: b[1]) + sorted(right, key=lambda b: b[1])
            else:
                blocks = sorted(blocks, key=lambda b: (b[1], b[0]))

            # Join blocks with blank lines so downstream splitting sees paragraph units.
            text = "\n\n".join(_clean_block(b[4]) for b in blocks)
            pages.append(PageText(page=i, text=text.strip()))

    return _truncate_at_references(pages)


# Page-one furniture that sits above the title: preprint stamps, license grants, venue lines.
_TITLE_BANNER_RE = re.compile(
    r"^(?:arxiv:|provided proper attribution|permission to (?:reproduce|make)|copyright|"
    r"preprint|to appear in|accepted (?:at|to)|under review|submitted to|\d{4}\s+ieee|"
    r"acm isbn|doi:|https?://|\d{4}\s+association for computational linguistics)",
    re.IGNORECASE,
)


def extract_title(pdf_path: str, fallback: str) -> str:
    """Best-effort paper title from page 1, by font size.

    Font size beats reading order here. Taking the first plausible text block instead picks up
    whatever furniture a publisher put above the title — on "Attention Is All You Need" that is
    Google's reproduction-permission notice, which is a perfectly plausible-looking sentence.

    Two filters make size reliable on arXiv PDFs:

    * **Rotated text is skipped.** The arXiv stamp runs vertically up the left margin in a large
      font, so on size alone it outranks the real title. Measured on the 12-paper corpus, it won
      6 of 12 before this filter.
    * **Known banners are skipped** even when horizontal (license grants, venue lines).

    Title lines that wrap are rejoined, ordered by vertical position, since a title set over two
    lines shares one font size.
    """
    import pymupdf

    try:
        with pymupdf.open(pdf_path) as doc:
            if doc.page_count == 0:
                return fallback
            lines: list[tuple[float, float, str]] = []  # (size, y, text)
            for block in doc[0].get_text("dict")["blocks"]:
                if block.get("type") != 0:  # 0 = text
                    continue
                for line in block["lines"]:
                    if tuple(round(v) for v in line["dir"]) != (1, 0):
                        continue
                    text = "".join(span["text"] for span in line["spans"]).strip()
                    if not text or _TITLE_BANNER_RE.match(text):
                        continue
                    size = max(span["size"] for span in line["spans"])
                    lines.append((round(size, 1), line["bbox"][1], text))
    except Exception:  # noqa: BLE001 - a broken PDF must not fail ingestion over a title
        return fallback

    if not lines:
        return fallback

    # Nothing below the abstract can be the title.
    cutoff = min(
        (y for _, y, text in lines if text.lower().lstrip().startswith("abstract")),
        default=float("inf"),
    )
    head = [row for row in lines if row[1] < cutoff] or lines

    largest = max(size for size, _, _ in head)
    parts = [text for size, _, text in sorted(head, key=lambda r: r[1]) if size == largest]
    title = " ".join(" ".join(parts).split())
    return title if 8 <= len(title) <= 300 else fallback


def _clean_block(text: str) -> str:
    """Collapse a block's internal line wrapping into a single paragraph."""
    text = text.replace("\u00ad", "")
    text = re.sub(r"-\n(?=[a-z])", "", text)   # de-hyphenate across line breaks
    text = re.sub(r"\s*\n\s*", " ", text)      # unwrap
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _clean(text: str) -> str:
    text = text.replace("\u00ad", "")                  # soft hyphens
    text = re.sub(r"-\n(?=[a-z])", "", text)           # de-hyphenate line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _truncate_at_references(pages: list[PageText]) -> list[PageText]:
    """Drop everything from the References heading onward."""
    for idx, page in enumerate(pages):
        lines = page.text.split("\n")
        for line_no, line in enumerate(lines):
            if _REFERENCES_RE.match(line):
                head = "\n".join(lines[:line_no]).strip()
                kept = pages[:idx]
                if head:
                    kept = [*kept, PageText(page=page.page, text=head)]
                # Only trust the marker if it appears late; "References" can occur early in
                # a related-work sentence on page 1.
                if idx >= max(1, len(pages) // 3):
                    return kept
    return pages


def _detect_heading(line: str) -> str | None:
    """Identify a section heading.

    Academic PDFs number sections several ways — "3 Method", "III. METHOD", "B. Clinical Named
    Entity Information" — so we accept a leading numeral, roman numeral or letter, then fall back
    to a shape heuristic (short, capitalised, unpunctuated) for headings we don't recognise.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return None

    # Reject figure/table captions and data-like lines that share a heading's shape.
    if re.match(r"^\s*(fig(?:ure)?|table|algorithm|eq(?:uation)?)\b", stripped, re.IGNORECASE):
        return None
    if ":" in stripped or re.search(r"\d\s*(?:cm|mm|%)\b", stripped):
        return None

    match = _HEADING_RE.match(stripped)
    if match:
        return match.group(1).strip().title()

    # Strip a leading enumerator: "3.", "3.1", "III.", "B."
    body = re.sub(r"^\s*(?:\d+(?:\.\d+)*\.?|[IVXLC]{1,5}\.|[A-Z]\.)\s+", "", stripped)
    had_enumerator = body != stripped
    if not body or body.endswith((".", ",", ";", ":")):
        return None

    words = body.split()
    if not (1 <= len(words) <= 9):
        return None
    if not body[:1].isupper():
        return None

    # ALL CAPS headings, or an explicit enumerator, or majority-capitalised title case.
    capitalised = sum(w[:1].isupper() for w in words)
    if body.isupper() or had_enumerator or capitalised >= max(1, len(words) * 2 // 3):
        return body.title() if body.isupper() else body
    return None


def chunk_pages(
    pages: list[PageText], target_tokens: int = 320, overlap_tokens: int = 64
) -> list[Chunk]:
    """Pack page paragraphs into token-bounded, page-attributed chunks."""
    # Flatten to (page, paragraph) units so a chunk can span a page break while still
    # recording the true page range.
    units: list[tuple[int, str, str | None]] = []
    current_section: str | None = None

    for page in pages:
        for block in page.text.split("\n\n"):
            paragraph = block.strip()
            if not paragraph:
                continue

            # Blocks are already unwrapped paragraphs, so a short block is either a heading or
            # page furniture (folios, running heads, equation fragments, figure labels).
            if len(paragraph) <= 80:
                heading = _detect_heading(paragraph)
                if heading:
                    current_section = heading
                    continue

            if len(paragraph) < 40:
                continue

            units.append((page.page, paragraph, current_section))

    chunks: list[Chunk] = []
    buf: list[tuple[int, str, str | None]] = []
    buf_tokens = 0
    ordinal = 0

    def flush() -> None:
        nonlocal buf, buf_tokens, ordinal
        if not buf:
            return
        text = "\n\n".join(p for _, p, _ in buf)
        pages_in = [pg for pg, _, _ in buf]
        section = next((s for _, _, s in buf if s), None)
        chunks.append(
            Chunk(
                ordinal=ordinal,
                page_start=min(pages_in),
                page_end=max(pages_in),
                section=section,
                text=text,
                n_tokens=n_tokens(text),
            )
        )
        ordinal += 1

        # Carry a token-bounded tail into the next chunk.
        if overlap_tokens > 0:
            tail: list[tuple[int, str, str | None]] = []
            acc = 0
            for unit in reversed(buf):
                t = n_tokens(unit[1])
                if acc + t > overlap_tokens:
                    break
                tail.insert(0, unit)
                acc += t
            buf = tail
            buf_tokens = acc
        else:
            buf, buf_tokens = [], 0

    # Split any paragraph that alone exceeds the target, on sentence boundaries. Without this a
    # single run-on block (common where PyMuPDF merges a column) produces a chunk several times
    # the target size, which dilutes its embedding and wastes context budget.
    expanded: list[tuple[int, str, str | None]] = []
    for page_no, paragraph, section in units:
        if n_tokens(paragraph) <= target_tokens:
            expanded.append((page_no, paragraph, section))
            continue
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", paragraph)
        part: list[str] = []
        part_tokens = 0
        for sentence in sentences:
            st = n_tokens(sentence)
            if part and part_tokens + st > target_tokens:
                expanded.append((page_no, " ".join(part), section))
                part, part_tokens = [], 0
            part.append(sentence)
            part_tokens += st
        if part:
            expanded.append((page_no, " ".join(part), section))
    units = expanded

    for unit in units:
        tokens = n_tokens(unit[1])
        # Still-oversized (a single unsplittable sentence) becomes its own chunk.
        if tokens >= target_tokens and not buf:
            buf = [unit]
            buf_tokens = tokens
            flush()
            continue
        if buf_tokens + tokens > target_tokens and buf:
            flush()
        buf.append(unit)
        buf_tokens += tokens

    if buf:
        # Final flush without generating an overlap-only remnant.
        text = "\n\n".join(p for _, p, _ in buf)
        if n_tokens(text) >= 30:
            pages_in = [pg for pg, _, _ in buf]
            chunks.append(
                Chunk(
                    ordinal=ordinal,
                    page_start=min(pages_in),
                    page_end=max(pages_in),
                    section=next((s for _, _, s in buf if s), None),
                    text=text,
                    n_tokens=n_tokens(text),
                )
            )

    return chunks
