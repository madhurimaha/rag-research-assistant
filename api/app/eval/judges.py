"""LLM judges for grounding and free-form correctness.

Two judges, both reference-scored against a fixed rubric:

* **Faithfulness** — implements the algorithm Ragas established: decompose the answer into atomic
  standalone statements, then verify each against the retrieved context with an entailment check,
  and score the fraction supported. Implemented here rather than imported so the prompts can be
  versioned and pinned (which the evaluation literature recommends and a library cannot offer),
  and because Ragas pulls the whole LangChain stack that `PLAN.md` §4.1 deliberately avoids. The
  algorithm is cited; the implementation is ours.
* **Free-form correctness** — semantic equivalence against the gold answer, for the 29 gold
  questions whose references are paraphrases rather than extractable spans. Token overlap is
  meaningless on those, so a judge is the only honest option.

Three practices from the evaluation literature are enforced structurally rather than by
convention:

1. **Prompts are versioned.** ``PROMPT_VERSIONS`` is written into ``eval_runs.config`` with the
   judge model, so a score can always be traced to the exact rubric that produced it. Any edit to
   a prompt below must bump its version.
2. **Judging is deterministic where the model allows it.** Temperature is pinned to 0 for models
   that accept it; reasoning-family models that reject the parameter are noted in the metadata.
3. **Failures are recorded, never silently dropped.** A spike measured a ~5% structured-output
   failure rate that correlated with answer length — so discarding failed rows would preferentially
   discard the longest answers, biasing faithfulness upward. Every result therefore carries an
   explicit ``error`` and the harness reports a judged-coverage rate alongside every judged score.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from app.core.config import Settings, get_settings

PROMPT_VERSIONS = {
    "decompose": "decompose/v1",
    "entail": "entail/v1",
    "freeform": "freeform/v1",
}

# --------------------------------------------------------------------------- prompts

_DECOMPOSE_SYSTEM = """\
You split an answer into atomic factual claims.

Rules:
1. Each claim must stand alone: resolve pronouns and references so it is understandable without
   the other claims or the original answer.
2. One assertion per claim. Split conjunctions and lists into separate claims.
3. Copy the answer's own wording and figures. Do not paraphrase, generalise, or add anything.
4. Ignore text that asserts nothing: citation markers, hedges, and meta-commentary about the
   sources.
5. If the answer makes no factual assertion at all, return an empty list.

Return JSON only: {"claims": ["...", "..."]}\
"""

_ENTAIL_SYSTEM = """\
You check whether each claim is supported by the given context.

A claim is supported only if the context states or directly entails it. Judge each claim strictly
and independently:
- Do not use outside knowledge. The context is the only evidence.
- A claim that is plausible, or true in general, but absent from the context is NOT supported.
- A claim that contradicts the context is NOT supported.
- Paraphrase is fine; the claim need not match the context word for word.

Return JSON only, with one entry per claim in the order given:
{"verdicts": [{"index": 1, "supported": true, "reason": "one short sentence"}, ...]}\
"""

_FREEFORM_SYSTEM = """\
You grade a candidate answer against a reference answer written by a domain expert.

Score the candidate on whether it conveys the same substance as the reference:
- 1.0 — states everything the reference states; extra correct detail is fine.
- 0.5 — states part of it, or is correct but omits a substantive part of the reference.
- 0.0 — contradicts the reference, or misses its substance entirely.

Grade substance, not style. Length, wording, and ordering do not matter. The candidate may be far
longer than the reference; that alone is not a deduction. An answer that declines to answer scores
0.0.

Return JSON only: {"score": 1.0, "reason": "one short sentence"}\
"""


# --------------------------------------------------------------------------- results


@dataclass
class FaithfulnessResult:
    """``score`` is None when the judge failed; callers must not coerce that to 0."""

    score: float | None
    n_claims: int = 0
    n_supported: int = 0
    claims: list[dict] = field(default_factory=list)
    error: str | None = None


@dataclass
class FreeformResult:
    score: float | None
    reason: str = ""
    error: str | None = None


def judge_metadata(settings: Settings | None = None) -> dict:
    """Provenance recorded with every run, so a score is traceable to what produced it."""
    settings = settings or get_settings()
    model = settings.active_judge_model
    return {
        "judge_model": model,
        "generation_model": settings.active_model,
        "prompt_versions": dict(PROMPT_VERSIONS),
        "deterministic": not _is_reasoning_model(model),
        "single_vendor": True,
        "limitation": (
            "Single judge from one vendor. Convention prefers agreement across two heterogeneous "
            "judges; no second provider key is available in this environment. Generator and judge "
            "are different models to avoid self-preference bias."
        ),
    }


# --------------------------------------------------------------------------- transport


def _is_reasoning_model(model: str) -> bool:
    return model.startswith(("gpt-5", "o1", "o3", "o4"))


def build_client(settings: Settings | None = None):
    """One client, reused across judge calls; threads in the harness share it safely."""
    settings = settings or get_settings()
    from openai import OpenAI

    if settings.llm_provider == "ollama":
        return OpenAI(base_url=settings.ollama_base_url, api_key="ollama")
    return OpenAI(api_key=settings.openai_api_key)


def _ask_json(client, model: str, system: str, user: str, max_tokens: int = 3000) -> dict:
    """One JSON-mode completion, parsed.

    JSON mode is used rather than parsing free text because the spike's failures were all
    malformed or truncated output. Reasoning models need a generous token ceiling: the budget is
    shared with hidden reasoning tokens, and running out mid-object is exactly how the 5% failure
    rate arose.
    """
    kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
    }
    if _is_reasoning_model(model):
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens
        kwargs["temperature"] = 0.0

    response = client.chat.completions.create(**kwargs)
    choice = response.choices[0]
    if choice.finish_reason == "length":
        raise ValueError("judge output truncated (finish_reason=length)")
    return json.loads(choice.message.content or "")


def _with_retry(fn, attempts: int = 2):
    """Retry once on a transport or parse failure, then surface the error.

    Retrying matters because the failure mode is stochastic truncation rather than a bad request,
    so a second attempt usually succeeds; surfacing it matters because a dropped row biases the
    aggregate.
    """
    last: Exception | None = None
    for _ in range(attempts):
        try:
            return fn(), None
        except Exception as exc:  # noqa: BLE001 - judge failures are data, not crashes
            last = exc
    return None, f"{type(last).__name__}: {str(last)[:200]}"


# --------------------------------------------------------------------------- judges


def judge_faithfulness(
    question: str,
    answer: str,
    context_texts: list[str],
    settings: Settings | None = None,
    client=None,
) -> FaithfulnessResult:
    """Fraction of the answer's atomic claims that the retrieved context supports.

    Note for the report: an abstention decomposes to zero claims and so has no faithfulness score
    at all — ``score`` is None, not 1.0. A measured finding from the framework spike, where an
    abstention scored a perfect 1.0 and would have rewarded a system for refusing to answer.
    """
    settings = settings or get_settings()
    client = client or build_client(settings)
    model = settings.active_judge_model
    context = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(context_texts, start=1))

    decomposed, error = _with_retry(
        lambda: _ask_json(
            client,
            model,
            _DECOMPOSE_SYSTEM,
            f"Question:\n{question}\n\nAnswer:\n{answer}",
        )
    )
    if error:
        return FaithfulnessResult(score=None, error=f"decompose: {error}")

    claims = [str(c).strip() for c in (decomposed or {}).get("claims", []) if str(c).strip()]
    if not claims:
        # No assertions means nothing to ground. Scoring this 1.0 would reward saying nothing.
        return FaithfulnessResult(score=None, n_claims=0, error="no claims extracted")

    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(claims, start=1))
    verified, error = _with_retry(
        lambda: _ask_json(
            client,
            model,
            _ENTAIL_SYSTEM,
            f"Context:\n{context}\n\nClaims:\n{numbered}",
        )
    )
    if error:
        return FaithfulnessResult(score=None, n_claims=len(claims), error=f"entail: {error}")

    verdicts = {int(v["index"]): v for v in (verified or {}).get("verdicts", []) if "index" in v}
    if len(verdicts) != len(claims):
        return FaithfulnessResult(
            score=None,
            n_claims=len(claims),
            error=f"entail: got {len(verdicts)} verdicts for {len(claims)} claims",
        )

    rows = [
        {
            "claim": claim,
            "supported": bool(verdicts[i].get("supported")),
            "reason": str(verdicts[i].get("reason", ""))[:300],
        }
        for i, claim in enumerate(claims, start=1)
    ]
    supported = sum(1 for r in rows if r["supported"])
    return FaithfulnessResult(
        score=supported / len(rows),
        n_claims=len(rows),
        n_supported=supported,
        claims=rows,
    )


def judge_freeform(
    question: str,
    answer: str,
    reference: str,
    settings: Settings | None = None,
    client=None,
) -> FreeformResult:
    """Graded semantic equivalence against the expert reference, on {0, 0.5, 1}."""
    settings = settings or get_settings()
    client = client or build_client(settings)

    parsed, error = _with_retry(
        lambda: _ask_json(
            client,
            settings.active_judge_model,
            _FREEFORM_SYSTEM,
            f"Question:\n{question}\n\nReference answer:\n{reference}\n\nCandidate answer:\n{answer}",
            max_tokens=1500,
        )
    )
    if error:
        return FreeformResult(score=None, error=error)

    raw = (parsed or {}).get("score")
    try:
        score = float(raw)
    except (TypeError, ValueError):
        return FreeformResult(score=None, error=f"unparsable score: {raw!r}")
    if score not in (0.0, 0.5, 1.0):
        score = min((0.0, 0.5, 1.0), key=lambda allowed: abs(allowed - score))
    return FreeformResult(score=score, reason=str((parsed or {}).get("reason", ""))[:300])
