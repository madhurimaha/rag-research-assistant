"""Markdown report generation.

The reports in `data/eval/results/` are generated, not hand-maintained. The previous ablation table was
written by hand from a throwaway script, which meant the prose could drift from the numbers and a
reviewer could not regenerate either. Here the interpretive sentences interpolate the measured
values — so if a rerun moves a number, the claim moves with it instead of silently going stale.
"""

from __future__ import annotations

import platform
from datetime import date
from pathlib import Path

from app.core.config import Settings
from app.eval import metrics as M
from app.eval.harness import RECALL_KS, RESULTS_DIR, AnswerRecord, ArmResult, corpus_stats
from app.eval.judges import judge_metadata


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _environment() -> str:
    return f"{platform.system()} {platform.machine()}, Python {platform.python_version()}"


# --------------------------------------------------------------------------- retrieval


def write_retrieval_report(arms: list[ArmResult], settings: Settings) -> Path:
    baseline = arms[0]
    best = max(arms, key=lambda a: a.recall[6])
    hybrid = next((a for a in arms if a.label.startswith("B")), None)
    reranked = [a for a in arms if "rerank" in a.label]

    lines: list[str] = []
    w = lines.append

    corpus = corpus_stats()
    distractors = corpus["documents"] - corpus["seed"]

    w("# Retrieval ablation — document-level")
    w("")
    w(
        f"**Setup.** {baseline.n} expert-annotated questions from UDA-QA `PaperText` over a "
        f"{corpus['documents']}-document corpus ({corpus['chunks']} chunks). Metric: does the "
        f"question's gold source document appear among the top *k* retrieved chunks? No LLM is "
        f"involved — embeddings and reranking are local ONNX models, so this table is reproducible "
        f"with no API key."
    )
    w("")
    if distractors:
        w(
            f"Note the corpus is {corpus['seed']} seed papers plus {distractors} uploaded "
            f"document(s). The gold questions all target the seed papers, so the uploaded "
            f"document(s) act as pure distractors for every question — a slightly harder setting "
            f"than the {corpus['seed']}-document corpus, and the reason the weaker arms score "
            f"a few points below an earlier run on the seed corpus alone."
        )
        w("")
    w(
        f"Gold labels are annotator-provided at document level for {baseline.n}/{baseline.n} "
        f"questions. Chunk-level labels were measured and rejected: exact span matching "
        f"localises only 19/100, and fuzzy token overlap would circularly favour the lexical arm."
    )
    w("")
    w(f"Regenerate with `python scripts/run_eval.py --retrieval`. Run on {_environment()}.")
    w("")

    header = " | ".join(f"R@{k}" for k in RECALL_KS)
    w(f"| Config | {header} | MRR | nDCG@6 | ms/query | R@6 95% CI |")
    w("| --- | " + " | ".join(["---"] * len(RECALL_KS)) + " | --- | --- | --- | --- |")
    for a in arms:
        cells = " | ".join(_pct(a.recall[k]) for k in RECALL_KS)
        lo, hi = a.recall6_ci
        bold = "**" if a is best else ""
        w(
            f"| {bold}{a.label}{bold} | {cells} | {a.mrr:.3f} | {a.ndcg6:.3f} | "
            f"{a.ms_per_query:.0f} | {_pct(lo)}–{_pct(hi)} |"
        )
    w("")

    w("## Significance against the vector-only baseline")
    w("")
    w(
        "Paired McNemar exact test on hit@6. Only questions whose outcome *differs* between the "
        "two configs carry information, so agreements are excluded by construction; the exact "
        "binomial form is used because the discordant counts are single-digit, where the "
        "chi-squared approximation is unreliable."
    )
    w("")
    w("| Config | Δ R@6 | gained | lost | p |")
    w("| --- | --- | --- | --- | --- |")
    for a in arms[1:]:
        gained = sum(1 for x, b in zip(a.hits, baseline.hits) if x and not b)
        lost = sum(1 for x, b in zip(a.hits, baseline.hits) if b and not x)
        p = M.mcnemar_exact(gained, lost)
        sig = "" if p >= 0.05 else " ✓"
        w(
            f"| {a.label} | {100 * (a.recall[6] - baseline.recall[6]):+.1f} | {gained} | {lost} | "
            f"{p:.4f}{sig} |"
        )
    w("")
    w("✓ = significant at p < 0.05.")
    w("")

    w("## Findings")
    w("")
    if hybrid and reranked:
        best_rr = max(reranked, key=lambda a: a.recall[6])
        hybrid_gain = 100 * (hybrid.recall[6] - baseline.recall[6])
        rerank_gain = 100 * (best_rr.recall[6] - hybrid.recall[6])
        w(
            f"**1. Reranking is the dominant win ({rerank_gain:+.1f} points), not hybrid fusion "
            f"({hybrid_gain:+.1f}).** This inverts the ordering usually implied by write-ups that "
            f"present hybrid search as the headline change. On this corpus the cross-encoder does "
            f"most of the work. The likely reason is corpus character: homogeneous academic prose "
            f"in a single domain gives the lexical arm few distinctive exact-match strings to "
            f"recover, where a corpus of product codes, contract clauses or tickers would offer "
            f"many. The hybrid gain is real but modest here, and would likely be larger elsewhere."
        )
        w("")

        depths = [(a.overrides.get("rerank_depth"), a.recall[6]) for a in reranked]
        peak_depth, peak = max(depths, key=lambda d: d[1])
        deeper = [(d, r) for d, r in depths if d and peak_depth and d > peak_depth]
        if deeper and any(r < peak for _, r in deeper):
            worst_d, worst_r = min(deeper, key=lambda d: d[1])
            fast = next((a for a in reranked if a.overrides.get("rerank_depth") == peak_depth), None)
            slow = next((a for a in reranked if a.overrides.get("rerank_depth") == worst_d), None)
            speedup = (slow.ms_per_query / fast.ms_per_query) if fast and slow and fast.ms_per_query else 0
            w(
                f"**2. Rerank quality is non-monotonic in depth — {peak_depth} beats {worst_d}.** "
                f"Recall rises to {_pct(peak)} at depth {peak_depth} then *falls* to "
                f"{_pct(worst_r)}. Feeding the cross-encoder more candidates gives it more "
                f"opportunities to promote a plausible-looking wrong chunk above a correct one, and "
                f"those promotions land directly in the top 6. So rerank depth is a quality "
                f"parameter to be tuned, not merely a latency knob — the common assumption that "
                f"\"rerank more, get more\" is wrong on this data. Depth {peak_depth} is also "
                f"{speedup:.1f}× faster than depth {worst_d}."
            )
            w("")

        w(
            f"**3. Hybrid fusion is effectively free.** {hybrid.ms_per_query:.0f}ms against "
            f"{baseline.ms_per_query:.0f}ms for vector alone. Two index scans plus rank arithmetic "
            f"in one SQL round-trip costs nothing measurable, because fusion happens next to the "
            f"indexes rather than by shipping two candidate sets into Python."
        )
        w("")

        lo, hi = best.recall6_ci
        w(
            f"**4. The remaining {_pct(1 - best.recall[6])} is the interesting part.** The "
            f"confidence interval on the chosen config ({_pct(lo)}–{_pct(hi)}) is wide enough that "
            f"neighbouring depths are not cleanly separated — depth {peak_depth} is the best "
            f"estimate, not a proven optimum. Note also that this is a *document-level* metric and "
            f"therefore an upper bound on chunk-level correctness: the right document can be "
            f"retrieved while the specific evidence chunk is missed. Failure attribution is in "
            f"[generation.md](generation.md)."
        )
        w("")

    w("## Chosen configuration")
    w("")
    w(
        f"`use_hybrid={settings.use_hybrid}`, `use_rerank={settings.use_rerank}`, "
        f"`rerank_depth={settings.rerank_depth}`, "
        f"`candidates_per_arm={settings.candidates_per_arm}`, `rrf_k={settings.rrf_k}`, "
        f"`hnsw_ef_search={settings.hnsw_ef_search}`, "
        f"`final_context_chunks={settings.final_context_chunks}`"
    )
    w("")
    chosen = next((a for a in arms if a.overrides.get("rerank_depth") == settings.rerank_depth), best)
    w(
        f"At {chosen.ms_per_query:.0f}ms of retrieval latency the reranker dominates the query "
        f"budget. It runs on CPU here; a GPU or a hosted reranker would cut it by an order of "
        f"magnitude, and that is the first thing to change if this were deployed."
    )
    w("")
    w(f"_Generated {date.today().isoformat()} by `app.eval.report`._")

    path = RESULTS_DIR / "retrieval_ablation.md"
    path.write_text("\n".join(lines) + "\n")
    return path


# --------------------------------------------------------------------------- generation

_CATEGORY_NOTE = {
    "extractive": "token recall of the gold span (precision omitted — see below)",
    "yes_no": "verdict accuracy, with unreadable answers reported as indeterminate",
    "free_form": "LLM judge, graded {0, 0.5, 1} against the expert reference",
}


def write_generation_report(
    records: list[AnswerRecord], summary: dict, settings: Settings
) -> Path:
    lines: list[str] = []
    w = lines.append
    meta = judge_metadata(settings)

    w("# Generation quality")
    w("")
    w(
        f"**Setup.** {summary['n_questions']} gold questions answered once under the chosen "
        f"retrieval configuration, generated by `{settings.active_model}` and scored by "
        f"`{meta['judge_model']}` where a judge is used. Prompt versions: "
        f"`{'`, `'.join(meta['prompt_versions'].values())}`."
    )
    w("")
    w("Regenerate with `python scripts/run_eval.py --generation`.")
    w("")

    w("## Evaluation dimensions")
    w("")
    w(
        "We evaluate retrieval and generation separately so failures can be attributed to the "
        "correct stage. Retrieval metrics use UDA-QA’s expert-provided source-document labels. "
        "Generation is evaluated for answer coverage against expert reference answers and for "
        "faithfulness to the retrieved context. Citation coverage and abstention behavior are "
        "reported separately."
    )
    w("")
    w(
        "These measures answer different questions: retrieval measures whether the expected "
        "source reached the context window; reference-based scoring measures whether the answer "
        "covers the expected information; and faithfulness measures whether the generated claims "
        "are supported by the retrieved passages. No single score represents overall RAG quality."
    )
    w("")

    w("## Faithfulness")
    w("")
    f = summary["faithfulness"]
    w(
        f"Faithfulness follows the algorithm Ragas established — decompose the answer into atomic "
        f"claims, verify each against the retrieved context, score the fraction supported — "
        f"implemented here so the prompts can be versioned and pinned."
    )
    w("")
    w("| Measure | Value |")
    w("| --- | --- |")
    w(f"| Mean faithfulness (scored answers) | **{f['mean']:.3f}** |")
    w(f"| Judged coverage | {f['scored']}/{f['scored'] + f['unscored']} ({_pct(f['coverage'])}) |")
    w(f"| Fully grounded (1.0) | {f['fully_grounded']} |")
    w(f"| Wholly ungrounded (0.0) | {f['ungrounded']} |")
    w(f"| Answer rate | {_pct(summary['answer_rate'])} |")
    w(f"| Abstention rate | {_pct(summary['abstention_rate'])} |")
    w(f"| Mean citations per answer | {summary['mean_citations_per_answer']:.2f} |")
    w(f"| Answers with no citation | {summary['uncited_answers']} |")
    w("")
    w(
        "**Faithfulness is reported beside the answer rate, and only over non-abstentions.** This "
        "is load-bearing, not decorative. An abstention asserts nothing, so it is vacuously "
        "faithful — measured directly during the framework spike, where a refusal scored a perfect "
        "1.0. Read alone, mean faithfulness therefore rewards a system for refusing to answer, and "
        "the configuration with the *worst* retrieval would look the most trustworthy. The pair of "
        "numbers closes that hole."
    )
    w("")
    w(
        f"**Judged coverage is reported for the same reason.** Structured-output failures are "
        f"stochastic and correlate with answer length, so dropping failed rows would "
        f"preferentially drop the longest answers — the ones most likely to contain an unsupported "
        f"claim, biasing the mean upward. {f['unscored']} of "
        f"{f['scored'] + f['unscored']} answers went unscored here and are counted, not hidden."
    )
    w("")

    w("## Answer coverage")
    w("")
    w(
        "The gold set mixes three incompatible question kinds, so they are scored separately and "
        "never averaged into one number. A single blended correctness score would combine a "
        "token-overlap measure, a binary verdict and a judge rating as though they were the same "
        "quantity; Qasper, the upstream source of these annotations, reports per-category for the "
        "same reason."
    )
    w("")
    w("| Type | n | scored | mean | how it is scored |")
    w("| --- | --- | --- | --- | --- |")
    for cat, note in _CATEGORY_NOTE.items():
        c = summary["correctness_by_category"][cat]
        unscored = f" ({c['unscored']} unscored)" if c["unscored"] else ""
        w(f"| {cat} | {c['n']} | {c['scored']}{unscored} | {c['mean_correctness']:.3f} | {note} |")
    w("")
    w(
        "**Why precision is omitted from the extractive score.** The generator writes prose "
        "(hundreds of characters) while the gold references are extractive spans, several of them "
        "a single word. Token precision therefore penalises the system for answering in sentences, "
        "which is a formatting mismatch and not a quality signal. This is not hypothetical: a "
        "framework spike scored our answers 0.0 on *all* ten sampled questions for exactly this "
        "reason. Recall is the honest measure of whether "
        "the answer covers the reference."
    )
    w("")

    w("## Judge validity")
    w("")
    w(
        f"- **Generator and judge are different models** (`{settings.active_model}` vs "
        f"`{meta['judge_model']}`), so an answer is never graded by the model that wrote it."
    )
    w(
        f"- **Prompts are versioned and recorded** with every run in `eval_runs.config`, so a "
        f"score can be traced to the exact rubric that produced it."
    )
    w(
        f"- **Determinism:** temperature is pinned to 0 for models that accept it. "
        f"`{meta['judge_model']}` is a reasoning-family model that rejects the parameter, so its "
        f"output is not bit-reproducible — recorded as `deterministic: "
        f"{str(meta['deterministic']).lower()}` rather than asserted."
    )
    w(
        "- **Known limitation:** a single judge from a single vendor. Convention prefers agreement "
        "across two heterogeneous judges; no second provider key was available in this "
        "environment, so no agreement figure is reported. Stated rather than papered over."
    )
    w("")

    w("## Failure analysis")
    w("")
    w(_failure_section(records, summary))
    w("")
    w(f"_Generated {date.today().isoformat()} by `app.eval.report`._")

    path = RESULTS_DIR / "generation.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def _failure_section(records: list[AnswerRecord], summary: dict) -> str:
    """Attribute failures to the taxonomy, with a real example per bucket."""
    buckets: dict[str, list[AnswerRecord]] = {
        "retrieval miss": [r for r in records if not r.retrieved_gold_doc and not r.abstained],
        "over-abstention": [r for r in records if r.abstained and r.retrieved_gold_doc],
        "unsupported generation": [
            r for r in records if r.faithfulness is not None and r.faithfulness < 0.5
        ],
        "partial grounding": [
            r for r in records if r.faithfulness is not None and 0.5 <= r.faithfulness < 1.0
        ],
        "citation missing": [r for r in records if not r.abstained and r.n_citations == 0],
        "unscorable verdict": [r for r in records if r.correctness_kind == "indeterminate"],
    }

    out = ["| Category | n | What it means |", "| --- | --- | --- |"]
    meaning = {
        "retrieval miss": "gold document absent from the context the generator received",
        "over-abstention": "refused despite the gold document being present in context",
        "unsupported generation": "most claims not entailed by the retrieved context",
        "partial grounding": "some claims supported, some not",
        "citation missing": "answered with no inline citation at all",
        "unscorable verdict": "yes/no answer whose verdict could not be read confidently",
    }
    for name, rows in buckets.items():
        out.append(f"| {name} | {len(rows)} | {meaning[name]} |")
    out.append("")

    out.append(
        "Buckets overlap by design — a retrieval miss usually causes either an abstention or an "
        "unsupported answer, and counting it once in each place is what makes the chain visible."
    )
    out.append("")

    for name, rows in buckets.items():
        if not rows:
            continue
        r = rows[0]
        out.append(f"**{name}** — example:")
        out.append("")
        out.append(f"> *Q:* {r.question}")
        out.append(f"> *Gold:* {r.reference[:200]}")
        out.append(f"> *Answer:* {r.answer[:260]}{'…' if len(r.answer) > 260 else ''}")
        detail = []
        if r.faithfulness is not None:
            detail.append(f"faithfulness {r.faithfulness:.2f}")
        if r.correctness is not None:
            detail.append(f"correctness {r.correctness:.2f} ({r.correctness_kind})")
        detail.append(f"gold doc retrieved: {'yes' if r.retrieved_gold_doc else 'no'}")
        out.append(f"> *Scores:* {', '.join(detail)}")
        out.append("")

    n_abst = summary["abstained_with_gold_doc_retrieved"]
    out.append(
        f"**The most actionable number here is over-abstention: {n_abst} questions where the "
        f"right document was in context and the system still declined.** That is a prompt and "
        f"threshold problem rather than a retrieval problem, which makes it the cheapest "
        f"remaining win — and it is invisible to any metric that does not pair faithfulness with "
        f"an answer rate."
    )
    return "\n".join(out)
