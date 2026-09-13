"""Title extraction from page 1.

The fixtures are built with PyMuPDF rather than checked-in PDFs, so each failure mode this
heuristic exists for is reproduced explicitly and the tests carry no binary assets.
"""

from __future__ import annotations

import pymupdf
import pytest

from app.rag.chunking import extract_title

FALLBACK = "1706.03762"


def build_pdf(path, items, page_size=(612, 792)):
    """`items` are (text, size, x, y, rotate) tuples written onto a single page."""
    doc = pymupdf.open()
    page = doc.new_page(width=page_size[0], height=page_size[1])
    for text, size, x, y, rotate in items:
        page.insert_text((x, y), text, fontsize=size, rotate=rotate)
    doc.save(path)
    doc.close()
    return str(path)


def test_plain_title_is_the_largest_text(tmp_path):
    pdf = build_pdf(
        tmp_path / "a.pdf",
        [
            ("Attention Is All You Need", 20, 100, 100, 0),
            ("Ashish Vaswani, Noam Shazeer", 10, 100, 140, 0),
            ("Abstract", 11, 100, 200, 0),
            ("The dominant sequence transduction models are based on", 10, 100, 220, 0),
        ],
    )
    assert extract_title(pdf, FALLBACK) == "Attention Is All You Need"


def test_rotated_arxiv_stamp_is_ignored_even_when_larger(tmp_path):
    """The arXiv stamp runs up the left margin in a large font; on size alone it would win."""
    pdf = build_pdf(
        tmp_path / "b.pdf",
        [
            ("arXiv:1909.13104v2 [cs.CL] 2 Dec 2019", 28, 30, 500, 90),
            ("Attention-based method for categorizing hate speech", 18, 100, 100, 0),
            ("Abstract", 11, 100, 200, 0),
        ],
    )
    assert extract_title(pdf, FALLBACK).startswith("Attention-based method")


def test_license_banner_above_the_title_is_ignored(tmp_path):
    """The bug this was written for: a plausible-looking sentence placed above the title."""
    pdf = build_pdf(
        tmp_path / "c.pdf",
        [
            ("Provided proper attribution is provided, Google hereby grants", 18, 100, 60, 0),
            ("Attention Is All You Need", 18, 100, 110, 0),
            ("Abstract", 11, 100, 200, 0),
        ],
    )
    assert extract_title(pdf, FALLBACK) == "Attention Is All You Need"


def test_wrapped_title_lines_are_joined(tmp_path):
    pdf = build_pdf(
        tmp_path / "d.pdf",
        [
            ("A BERT-Based Transfer Learning Approach", 18, 100, 100, 0),
            ("for Hate Speech Detection in Online Social Media", 18, 100, 126, 0),
            ("Abstract", 11, 100, 200, 0),
        ],
    )
    assert extract_title(pdf, FALLBACK) == (
        "A BERT-Based Transfer Learning Approach "
        "for Hate Speech Detection in Online Social Media"
    )


def test_body_text_below_the_abstract_is_never_the_title(tmp_path):
    """Large display text in the body (a figure caption) must not outrank the real title."""
    pdf = build_pdf(
        tmp_path / "e.pdf",
        [
            ("Short Real Title", 14, 100, 100, 0),
            ("Abstract", 11, 100, 160, 0),
            ("FIGURE 1: A VERY LARGE CAPTION", 30, 100, 400, 0),
        ],
    )
    assert extract_title(pdf, FALLBACK) == "Short Real Title"


@pytest.mark.parametrize(
    "banner",
    [
        "arXiv:2401.00001v1 [cs.CL] 1 Jan 2024",
        "Preprint. Under review.",
        "To appear in ACL 2024",
        "Copyright 2024 ACM",
        "doi:10.1145/1234567",
        "https://example.org/paper",
    ],
)
def test_known_banner_forms_are_skipped(tmp_path, banner):
    pdf = build_pdf(
        tmp_path / "f.pdf",
        [(banner, 20, 60, 60, 0), ("The Actual Paper Title", 20, 100, 110, 0)],
    )
    assert extract_title(pdf, FALLBACK) == "The Actual Paper Title"


def test_falls_back_when_page_has_no_usable_text(tmp_path):
    pdf = build_pdf(tmp_path / "g.pdf", [])
    assert extract_title(pdf, FALLBACK) == FALLBACK


def test_falls_back_on_a_corrupt_file(tmp_path):
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a pdf at all")
    assert extract_title(str(broken), FALLBACK) == FALLBACK


def test_falls_back_when_candidate_is_implausibly_short(tmp_path):
    pdf = build_pdf(tmp_path / "h.pdf", [("v2", 24, 100, 100, 0)])
    assert extract_title(pdf, FALLBACK) == FALLBACK
