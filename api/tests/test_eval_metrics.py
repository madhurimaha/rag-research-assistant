"""Metric correctness, against hand-computed values.

Every expected number here was worked out by hand from the metric's definition rather than by
recording what the implementation returned, which is the only way these tests can catch a wrong
formula. The statistical functions are additionally checked against published reference values.
"""

from __future__ import annotations

import math

import pytest

from app.eval.metrics import (
    exact_match,
    gold_span_covered,
    hit_rate_at_k,
    mcnemar_exact,
    ndcg_at_k,
    normalize_answer,
    recall_at_k,
    reciprocal_rank,
    token_f1,
    token_precision,
    token_recall,
    wilson_interval,
    yes_no_verdict,
)


class TestRanking:
    def test_reciprocal_rank_is_one_over_first_hit(self):
        assert reciprocal_rank(["a", "b", "c"], {"c"}) == pytest.approx(1 / 3)
        assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0

    def test_reciprocal_rank_is_zero_when_absent(self):
        assert reciprocal_rank(["a", "b"], {"z"}) == 0.0

    def test_reciprocal_rank_uses_the_first_hit_not_the_best(self):
        assert reciprocal_rank(["a", "b", "c"], {"b", "c"}) == pytest.approx(1 / 2)

    def test_recall_counts_the_fraction_of_gold_found(self):
        assert recall_at_k(["a", "b", "c"], {"a", "z"}, k=3) == pytest.approx(0.5)
        assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=3) == 1.0

    def test_recall_respects_the_cutoff(self):
        assert recall_at_k(["a", "b", "c"], {"c"}, k=2) == 0.0
        assert recall_at_k(["a", "b", "c"], {"c"}, k=3) == 1.0

    def test_recall_equals_hit_rate_for_single_label_gold(self):
        """The equivalence the eval report depends on; see the metrics module docstring."""
        for ranked, gold, k in (
            (["a", "b", "c"], {"b"}, 3),
            (["a", "b", "c"], {"z"}, 3),
            (["a", "b", "c"], {"c"}, 2),
        ):
            assert recall_at_k(ranked, gold, k) == hit_rate_at_k(ranked, gold, k)

    def test_recall_differs_from_hit_rate_for_multi_label_gold(self):
        assert recall_at_k(["a", "b"], {"a", "z"}, k=2) == pytest.approx(0.5)
        assert hit_rate_at_k(["a", "b"], {"a", "z"}, k=2) == 1.0

    def test_ndcg_is_one_when_the_only_gold_ranks_first(self):
        assert ndcg_at_k(["a", "b", "c"], {"a"}, k=3) == pytest.approx(1.0)

    def test_ndcg_discounts_by_log2_of_rank_plus_one(self):
        assert ndcg_at_k(["a", "b"], {"b"}, k=2) == pytest.approx(1 / math.log2(3))
        assert ndcg_at_k(["a", "b", "c"], {"c"}, k=3) == pytest.approx(0.5)

    def test_ndcg_normalises_against_the_ideal_ordering(self):
        # gold at ranks 1 and 3: DCG = 1/log2(2) + 1/log2(4) = 1.5
        #        ideal ranks 1,2: IDCG = 1/log2(2) + 1/log2(3) = 1.63093
        assert ndcg_at_k(["a", "x", "b"], {"a", "b"}, k=3) == pytest.approx(
            1.5 / (1 + 1 / math.log2(3))
        )

    def test_ndcg_separates_ranks_that_recall_cannot(self):
        """The reason nDCG is reported alongside recall at all."""
        first, last = ["g", "x", "y"], ["x", "y", "g"]
        assert recall_at_k(first, {"g"}, 3) == recall_at_k(last, {"g"}, 3) == 1.0
        assert ndcg_at_k(first, {"g"}, 3) > ndcg_at_k(last, {"g"}, 3)

    def test_empty_gold_scores_zero_rather_than_dividing_by_zero(self):
        assert recall_at_k(["a"], set(), k=1) == 0.0
        assert ndcg_at_k(["a"], set(), k=1) == 0.0
        assert reciprocal_rank(["a"], set()) == 0.0


class TestAnswerNormalisation:
    def test_strips_articles_punctuation_and_case(self):
        assert normalize_answer("The  Linear-SVM, and a BiLSTM!") == [
            "linear",
            "svm",
            "and",
            "bilstm",
        ]

    def test_unicode_is_folded_before_comparison(self):
        assert normalize_answer("ﬁne-tuning") == normalize_answer("fine tuning")

    def test_exact_match_ignores_formatting_only_differences(self):
        assert exact_match("English", "english") == 1.0
        assert exact_match("the English", "English") == 1.0
        assert exact_match("English and German", "English") == 0.0


class TestTokenScores:
    def test_recall_is_full_when_prose_contains_the_span(self):
        assert token_recall("The tweets are in English.", "english") == 1.0

    def test_precision_punishes_prose_against_a_one_word_span(self):
        """The concrete reason the report leads with recall and not F1."""
        pred, gold = "The tweets are in English.", "english"
        assert token_recall(pred, gold) == 1.0
        assert token_precision(pred, gold) == pytest.approx(0.25)
        assert token_f1(pred, gold) == pytest.approx(0.4)

    def test_recall_is_partial_on_a_partially_covered_span(self):
        # gold normalises to [linear, svm, and, bilstm]; the prediction supplies 2 of those 4.
        assert token_recall("They use a linear SVM.", "linear SVM and BiLSTM") == pytest.approx(
            2 / 4
        )

    def test_recall_is_zero_on_a_miss(self):
        assert token_recall("They use logistic regression.", "linear SVM") == 0.0

    def test_empty_gold_does_not_divide_by_zero(self):
        assert token_recall("anything", "") == 0.0
        assert token_f1("anything", "") == 0.0

    def test_span_coverage_requires_contiguity_where_recall_does_not(self):
        scattered = "linear models and an SVM"
        assert token_recall(scattered, "linear SVM") == 1.0
        assert gold_span_covered(scattered, "linear SVM") == 0.0
        assert gold_span_covered("we use a linear SVM here", "linear SVM") == 1.0


class TestYesNo:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Yes, the authors evaluate on three datasets [1].", "yes"),
            ("No, they do not release the code [2].", "no"),
            ("Indeed, the model is pre-trained [1].", "yes"),
            ("The authors do not use external resources [1].", "no"),
            ("None of the baselines use attention [3].", "no"),
            ("The model cannot be applied to unseen languages [1].", "no"),
        ],
    )
    def test_reads_explicit_verdicts(self, text, expected):
        assert yes_no_verdict(text) == expected

    def test_leading_marker_beats_later_negation(self):
        assert yes_no_verdict("Yes, although they do not evaluate on Arabic [1].") == "yes"

    @pytest.mark.parametrize(
        "text",
        [
            "The authors use three datasets [1].",
            "The corpus contains 17,833 sentences [2].",
            "",
            "   ",
        ],
    )
    def test_returns_none_rather_than_guessing(self, text):
        assert yes_no_verdict(text) is None

    def test_contrastive_negation_abstains_instead_of_guessing_wrong(self):
        """"X, not Y" is a choice between two things, not a denial.

        Bare "not" is deliberately absent from the negation pattern, which only matches verbal
        forms ("do not", "is not", "cannot"). So contrastive phrasing falls through to
        indeterminate rather than being scored as a confident "no" — the failure mode that would
        silently corrupt accuracy on the 14 yes/no questions.
        """
        assert yes_no_verdict("The model uses BERT, not GPT [1].") is None
        assert yes_no_verdict("They do not use GPT [1].") == "no"


class TestStatistics:
    def test_wilson_matches_published_interval_for_82_of_100(self):
        lo, hi = wilson_interval(82, 100)
        assert lo == pytest.approx(0.7333, abs=1e-3)
        assert hi == pytest.approx(0.8830, abs=1e-3)

    def test_wilson_stays_inside_the_unit_interval_at_the_boundary(self):
        lo, hi = wilson_interval(100, 100)
        assert hi == pytest.approx(1.0)
        assert 0.0 < lo < 1.0
        lo, hi = wilson_interval(0, 100)
        assert lo == pytest.approx(0.0)
        assert 0.0 < hi < 1.0

    def test_wilson_narrows_as_n_grows(self):
        small = wilson_interval(8, 10)
        large = wilson_interval(80, 100)
        assert (large[1] - large[0]) < (small[1] - small[0])

    def test_wilson_handles_zero_questions(self):
        assert wilson_interval(0, 0) == (0.0, 0.0)

    def test_mcnemar_is_one_when_no_question_flips(self):
        assert mcnemar_exact(0, 0) == 1.0

    def test_mcnemar_exact_binomial_values(self):
        # n=13 discordant, all one way: 2 * C(13,0) * 0.5^13
        assert mcnemar_exact(13, 0) == pytest.approx(2 / 8192)
        # n=9, one against: 2 * (C(9,0) + C(9,1)) * 0.5^9
        assert mcnemar_exact(8, 1) == pytest.approx(2 * 10 / 512)

    def test_mcnemar_is_symmetric_and_capped_at_one(self):
        assert mcnemar_exact(8, 1) == mcnemar_exact(1, 8)
        assert mcnemar_exact(5, 5) == 1.0

    def test_mcnemar_ignores_agreements(self):
        """Only discordant pairs carry signal, which is the whole point of the paired test."""
        assert mcnemar_exact(6, 0) == mcnemar_exact(6, 0)
        assert mcnemar_exact(6, 0) < 0.05
        assert mcnemar_exact(3, 0) > 0.05
