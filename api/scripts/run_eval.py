"""Run the evaluation and regenerate the reports in `eval/results/`.

    python scripts/run_eval.py --retrieval      # ablation sweep, no API key needed
    python scripts/run_eval.py --generation     # answer + score the gold set (judge needs a key)
    python scripts/run_eval.py --all

The retrieval half is deliberately keyless: embeddings and reranking are local ONNX models, so a
reviewer with no API key can still reproduce the headline ablation table. Only the faithfulness
and free-form-correctness judges require a key, and their absence degrades the report rather than
failing it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.eval import harness as H  # noqa: E402
from app.eval import report as R  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", action="store_true", help="run the ablation sweep")
    parser.add_argument("--generation", action="store_true", help="answer and score the gold set")
    parser.add_argument("--all", action="store_true", help="both")
    parser.add_argument("--limit", type=int, default=None, help="first N gold questions (smoke test)")
    parser.add_argument(
        "--answers-out",
        default=str(Path(__file__).resolve().parents[2] / "eval" / "results" / "answers.json"),
        help="where to dump per-question records for the failure analysis",
    )
    args = parser.parse_args()

    if not (args.retrieval or args.generation or args.all):
        parser.error("choose --retrieval, --generation, or --all")

    settings = get_settings()
    gold_n = len(H.load_gold())
    print(f"gold set: {gold_n} questions · provider={settings.llm_provider}")

    if args.retrieval or args.all:
        print("\n== retrieval ablation ==")
        arms = H.run_ablation(settings)
        H.persist_ablation(arms, settings)
        path = R.write_retrieval_report(arms, settings)
        print(f"wrote {path}")

    if args.generation or args.all:
        if not settings.has_llm:
            print("\n== generation == skipped: no LLM configured (LLM_PROVIDER=none)")
            return 0
        print(f"\n== generation ({settings.active_model}) ==")
        records = H.generate_answers(settings, limit=args.limit, checkpoint=args.answers_out)
        print(f"  scoring (judge={settings.active_judge_model})")
        H.score_answers(records, settings)
        H.save_checkpoint(args.answers_out, records)
        summary = H.summarize_generation(records)
        run_id = H.persist_generation(summary, settings)
        path = R.write_generation_report(records, summary, settings)
        print(f"wrote {path} (eval_runs id={run_id}); records -> {args.answers_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
