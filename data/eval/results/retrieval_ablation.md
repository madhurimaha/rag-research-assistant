# Retrieval ablation — document-level

**Setup.** 100 expert-annotated questions from UDA-QA `PaperText` over a 13-document corpus (303 chunks). Metric: does the question's gold source document appear among the top *k* retrieved chunks? No LLM is involved — embeddings and reranking are local ONNX models, so this table is reproducible with no API key.

Note the corpus is 12 seed papers plus 1 uploaded document(s). The gold questions all target the seed papers, so the uploaded document(s) act as pure distractors for every question — a slightly harder setting than the 12-document corpus, and the reason the weaker arms score a few points below an earlier run on the seed corpus alone.

Gold labels are annotator-provided at document level for 100/100 questions. Chunk-level labels were measured and rejected: exact span matching localises only 19/100, and fuzzy token overlap would circularly favour the lexical arm.

Regenerate with `python scripts/run_eval.py --retrieval`. Run on Darwin arm64, Python 3.12.11.

| Config | R@1 | R@3 | R@5 | R@6 | R@10 | MRR | nDCG@6 | ms/query | R@6 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A — vector only | 41.0% | 56.0% | 64.0% | 66.0% | 76.0% | 0.553 | 0.592 | 26 | 56.3%–74.5% |
| B — hybrid (RRF) | 46.0% | 65.0% | 68.0% | 71.0% | 80.0% | 0.608 | 0.649 | 32 | 61.5%–79.0% |
| C — hybrid + rerank, depth 10 | 55.0% | 68.0% | 73.0% | 75.0% | 80.0% | 0.657 | 0.683 | 835 | 65.7%–82.5% |
| C — hybrid + rerank, depth 15 | 57.0% | 70.0% | 79.0% | 79.0% | 83.0% | 0.676 | 0.707 | 1209 | 70.0%–85.8% |
| **C — hybrid + rerank, depth 20** | 57.0% | 71.0% | 78.0% | 82.0% | 85.0% | 0.675 | 0.714 | 1737 | 73.3%–88.3% |
| C — hybrid + rerank, depth 30 | 57.0% | 70.0% | 77.0% | 79.0% | 84.0% | 0.673 | 0.709 | 2296 | 70.0%–85.8% |
| C — hybrid + rerank, depth 50 | 57.0% | 70.0% | 78.0% | 78.0% | 85.0% | 0.674 | 0.706 | 4261 | 68.9%–85.0% |

## Significance against the vector-only baseline

Paired McNemar exact test on hit@6. Only questions whose outcome *differs* between the two configs carry information, so agreements are excluded by construction; the exact binomial form is used because the discordant counts are single-digit, where the chi-squared approximation is unreliable.

| Config | Δ R@6 | gained | lost | p |
| --- | --- | --- | --- | --- |
| B — hybrid (RRF) | +5.0 | 8 | 3 | 0.2266 |
| C — hybrid + rerank, depth 10 | +9.0 | 11 | 2 | 0.0225 ✓ |
| C — hybrid + rerank, depth 15 | +13.0 | 15 | 2 | 0.0023 ✓ |
| C — hybrid + rerank, depth 20 | +16.0 | 17 | 1 | 0.0001 ✓ |
| C — hybrid + rerank, depth 30 | +13.0 | 17 | 4 | 0.0072 ✓ |
| C — hybrid + rerank, depth 50 | +12.0 | 17 | 5 | 0.0169 ✓ |

✓ = significant at p < 0.05.

## Findings

**1. Reranking is the dominant win (+11.0 points), not hybrid fusion (+5.0).** This inverts the ordering usually implied by write-ups that present hybrid search as the headline change. On this corpus the cross-encoder does most of the work. The likely reason is corpus character: homogeneous academic prose in a single domain gives the lexical arm few distinctive exact-match strings to recover, where a corpus of product codes, contract clauses or tickers would offer many. The hybrid gain is real but modest here, and would likely be larger elsewhere.

**2. Rerank quality is non-monotonic in depth — 20 beats 50.** Recall rises to 82.0% at depth 20 then *falls* to 78.0%. Feeding the cross-encoder more candidates gives it more opportunities to promote a plausible-looking wrong chunk above a correct one, and those promotions land directly in the top 6. So rerank depth is a quality parameter to be tuned, not merely a latency knob — the common assumption that "rerank more, get more" is wrong on this data. Depth 20 is also 2.5× faster than depth 50.

**3. Hybrid fusion is effectively free.** 32ms against 26ms for vector alone. Two index scans plus rank arithmetic in one SQL round-trip costs nothing measurable, because fusion happens next to the indexes rather than by shipping two candidate sets into Python.

**4. The remaining 18.0% is the interesting part.** The confidence interval on the chosen config (73.3%–88.3%) is wide enough that neighbouring depths are not cleanly separated — depth 20 is the best estimate, not a proven optimum. Note also that this is a *document-level* metric and therefore an upper bound on chunk-level correctness: the right document can be retrieved while the specific evidence chunk is missed. Failure attribution is in [generation.md](generation.md).

## Chosen configuration

`use_hybrid=True`, `use_rerank=True`, `rerank_depth=20`, `candidates_per_arm=50`, `rrf_k=60`, `hnsw_ef_search=100`, `final_context_chunks=6`

At 1737ms of retrieval latency the reranker dominates the query budget. It runs on CPU here; a GPU or a hosted reranker would cut it by an order of magnitude, and that is the first thing to change if this were deployed.

_Generated 2026-09-13 by `app.eval.report`._
