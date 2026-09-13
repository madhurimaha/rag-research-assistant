"""Fetch the seed corpus and gold questions from UDA-QA.

Corpus choice
-------------
UDA-QA (NeurIPS 2024) `PaperText` is a Qasper-derived set of NLP research papers with
expert-annotated Q&A. We take the 12 documents carrying the most annotations, which yields
~100 human-labelled questions — roughly 8 per document.

That density is the point. Benchmarks built for broad/shallow coverage give ~1-2 questions per
document, which cannot support per-document failure analysis on a 12-document corpus.

`doc_name` values are arXiv ids, so PDFs come from arxiv.org directly (~10MB) rather than
UDA's 852MB paper_docs.zip.
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

from app.core.paths import CORPUS_DIR, GOLD_DIR

PARQUET_URL = (
    "https://huggingface.co/datasets/qinchuanhui/UDA-QA/"
    "resolve/main/paper_text/test_00000_of_00001.parquet"
)
ARXIV_PDF = "https://arxiv.org/pdf/{doc}"
UA = {"User-Agent": "rag-research-assistant/0.1 (take-home project)"}


def fetch(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=12)
    ap.add_argument("--corpus-dir", default=str(CORPUS_DIR))
    ap.add_argument("--gold-dir", default=str(GOLD_DIR))
    args = ap.parse_args()

    corpus_dir = Path(args.corpus_dir)
    gold_dir = Path(args.gold_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    gold_dir.mkdir(parents=True, exist_ok=True)

    print("→ downloading UDA-QA PaperText annotations …")
    import pandas as pd

    df = pd.read_parquet(io.BytesIO(fetch(PARQUET_URL, timeout=180)))
    print(f"  {len(df)} questions over {df.doc_name.nunique()} documents")

    counts = collections.Counter(df.doc_name)
    top = [doc for doc, _ in counts.most_common(args.n_docs)]
    subset = df[df.doc_name.isin(top)].copy()
    print(f"→ selected top {len(top)} documents → {len(subset)} gold questions")

    manifest = []
    for i, doc in enumerate(top, start=1):
        out = corpus_dir / f"{doc}.pdf"
        n_q = int(counts[doc])
        if out.exists():
            print(f"  [{i:2d}/{len(top)}] {doc}  cached ({n_q} questions)")
        else:
            try:
                data = fetch(ARXIV_PDF.format(doc=doc), timeout=120)
                out.write_bytes(data)
                print(f"  [{i:2d}/{len(top)}] {doc}  {len(data) / 1e6:.1f}MB ({n_q} questions)")
                time.sleep(1.0)  # be polite to arXiv
            except Exception as exc:  # noqa: BLE001
                print(f"  [{i:2d}/{len(top)}] {doc}  FAILED: {exc}", file=sys.stderr)
                continue
        manifest.append({"doc_key": doc, "filename": out.name, "n_gold_questions": n_q})

    (corpus_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Normalise gold questions. UDA stores up to three independent annotator answers; keeping
    # all of them lets correctness scoring credit any valid phrasing.
    gold = []
    for _, row in subset.iterrows():
        answers = [
            str(row[c]).strip()
            for c in ("answer_1", "answer_2", "answer_3")
            if c in row and row[c] is not None and str(row[c]).strip() not in ("", "nan")
        ]
        if not answers:
            continue
        gold.append(
            {
                "q_uid": row["q_uid"],
                "doc_key": row["doc_name"],
                "question": str(row["question"]).strip(),
                "answers": answers,
                "source": "uda-qa",
                "category": "yes_no"
                if answers[0].lower() in ("yes", "no")
                else ("free_form" if len(answers[0].split()) > 12 else "extractive"),
            }
        )

    out_gold = gold_dir / "uda_gold.json"
    out_gold.write_text(json.dumps(gold, indent=2))

    by_cat = collections.Counter(g["category"] for g in gold)
    print(f"\n→ wrote {len(gold)} gold questions to {out_gold}")
    print(f"  categories: {dict(by_cat)}")
    print(f"→ wrote manifest to {corpus_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
