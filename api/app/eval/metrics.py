"""Evaluation metrics.

Pure functions over plain values: no database, no network, no model. That is deliberate — these
are the numbers the submission is judged on, so they are unit-tested against hand-computed
fixtures rather than trusted because a framework produced them.

Two conventions are worth stating explicitly because they affect how the results table is read.

**Recall@k collapses to Hit Rate here.** Our gold set gives exactly one source document per
question, and with a single relevant item ``|relevant ∩ top-k| / |relevant|`` is 1 when the item
is found and 0 otherwise — identical to "did at least one relevant item appear". Both functions
exist because they are the named conventions and generalise to multi-label gold, but reporting
them as two columns of a single-label table would imply information that isn't there.

**Token recall is reported without precision.** Our answers are prose (~666 chars mean); the gold
references are extractive spans (~75 chars mean, some a single word). Precision therefore
penalises the system for writing sentences rather than spans, which is a formatting mismatch and
not a quality signal. ``token_f1`` is provided for completeness but is not the headline number,
and the reason is recorded in the eval report.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Hashable, Iterable, Sequence

# --------------------------------------------------------------------------- ranking


def reciprocal_rank(ranked: Sequence[Hashable], gold: Iterable[Hashable]) -> float:
    """1/rank of the first relevant item, or 0 if none is present.

    Rank is 1-based, matching the IR convention rather than list indices.
    """
    gold = set(gold)
    for i, item in enumerate(ranked, start=1):
        if item in gold:
            return 1.0 / i
    return 0.0


def recall_at_k(ranked: Sequence[Hashable], gold: Iterable[Hashable], k: int) -> float:
    """Fraction of relevant items appearing in the top k."""
    gold = set(gold)
    if not gold:
        return 0.0
    found = gold & set(ranked[:k])
    return len(found) / len(gold)


def hit_rate_at_k(ranked: Sequence[Hashable], gold: Iterable[Hashable], k: int) -> float:
    """1 if any relevant item appears in the top k, else 0."""
    return 1.0 if set(gold) & set(ranked[:k]) else 0.0


def ndcg_at_k(ranked: Sequence[Hashable], gold: Iterable[Hashable], k: int) -> float:
    """Normalised discounted cumulative gain with binary relevance.

    Unlike recall, this rewards placing the relevant item higher: it is the metric that
    distinguishes "found it at rank 1" from "found it at rank 6", which matters when only the top
    few chunks reach the generator's context window.
    """
    gold = set(gold)
    if not gold:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 1) for i, item in enumerate(ranked[:k], start=1) if item in gold)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, len(gold)) + 1))
    return dcg / ideal if ideal else 0.0


# --------------------------------------------------------------------------- answer text

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = re.compile(r"[^\w\s]")


def normalize_answer(text: str) -> list[str]:
    """SQuAD/Qasper-style normalisation: casefold, strip punctuation and articles, split.

    Using the upstream benchmark's normalisation is the point — it keeps our token-level scores
    comparable to published numbers on this dataset instead of being a bespoke measure.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    text = _PUNCT.sub(" ", text)
    text = _ARTICLES.sub(" ", text)
    return text.split()


def exact_match(prediction: str, gold: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def gold_span_covered(prediction: str, gold: str) -> float:
    """1 if the normalised gold span appears contiguously inside the prediction.

    A stricter companion to ``token_recall``: it rejects the case where every gold token appears
    somewhere in a long answer but never as the actual phrase.
    """
    p, g = normalize_answer(prediction), normalize_answer(gold)
    if not g:
        return 0.0
    return 1.0 if any(p[i : i + len(g)] == g for i in range(len(p) - len(g) + 1)) else 0.0


def _overlap(prediction: str, gold: str) -> tuple[int, int, int]:
    p, g = Counter(normalize_answer(prediction)), Counter(normalize_answer(gold))
    return sum((p & g).values()), sum(p.values()), sum(g.values())


def token_recall(prediction: str, gold: str) -> float:
    """Fraction of gold answer tokens present in the prediction — the headline correctness score.

    Recall-only because the prediction is prose and the gold is a span; see the module docstring.
    """
    common, _, n_gold = _overlap(prediction, gold)
    return common / n_gold if n_gold else 0.0


def token_precision(prediction: str, gold: str) -> float:
    common, n_pred, _ = _overlap(prediction, gold)
    return common / n_pred if n_pred else 0.0


def token_f1(prediction: str, gold: str) -> float:
    """Provided for completeness; misleading on this dataset. See the module docstring."""
    p, r = token_precision(prediction, gold), token_recall(prediction, gold)
    return 2 * p * r / (p + r) if p + r else 0.0


# --------------------------------------------------------------------------- yes/no

_AFFIRM = re.compile(r"^\W*(yes|indeed|correct|true)\b", re.IGNORECASE)
_DENY = re.compile(r"^\W*(no|not|nope|false|incorrect)\b", re.IGNORECASE)
_NEGATION = re.compile(
    r"\b(do(?:es)?\s+not|did\s+not|is\s+not|are\s+not|was\s+not|were\s+not|cannot|can(?:no|')t"
    r"|never|neither|nor|without|none|no\s+\w+)\b",
    re.IGNORECASE,
)


def yes_no_verdict(text: str) -> str | None:
    """Classify a prose answer as "yes"/"no", or ``None`` when it cannot be read confidently.

    The generator answers in sentences, but 14 gold questions expect a yes/no. Rather than force a
    guess, unreadable cases return ``None`` and are reported as an explicit *indeterminate* bucket:
    a heuristic that silently coerces ambiguous prose to "yes" would inflate accuracy by exactly
    the amount it is wrong.

    Leading markers win over sentence-internal negation, because "No, the authors do not..."  and
    "Yes, although they do not..." both open with their verdict.
    """
    head = text.strip()
    if not head:
        return None
    if _AFFIRM.match(head):
        return "yes"
    if _DENY.match(head):
        return "no"
    first = re.split(r"(?<=[.!?])\s", head)[0]
    return "no" if _NEGATION.search(first) else None


# --------------------------------------------------------------------------- statistics


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because it stays inside [0, 1] and keeps sensible
    coverage at the small n and near-boundary proportions this eval produces — a 100-question set
    scoring 82% is exactly where the textbook interval misbehaves.
    """
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    halfwidth = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, centre - halfwidth), min(1.0, centre + halfwidth))


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value for two paired binary classifiers.

    ``only_a`` is the count of questions config A got right and B wrong; ``only_b`` the reverse.
    Agreements carry no information about which config is better and are correctly ignored.

    The exact binomial form is used rather than the chi-squared approximation because the
    discordant counts here are small (single digits), where the approximation is unreliable.
    """
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) * 0.5**n
    return min(1.0, 2 * tail)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
