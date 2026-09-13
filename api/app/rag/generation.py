"""Grounded answer generation.

Design points
------------
* **Three provider tiers**, all behind one interface, selected by configuration:

  | `LLM_PROVIDER` | Generates? | Needs a key | Notes |
  | --- | --- | --- | --- |
  | `openai` / `anthropic` | yes | yes | Best instruction-following |
  | `ollama` | yes | **no** | Local open-weights model; free and offline |
  | `none` | no — returns evidence | no | Retrieval-only fallback |

  Ollama implements the OpenAI wire protocol, so it shares the OpenAI streaming client with only
  a base-URL change. That makes genuinely keyless *generation* possible, which matters for anyone
  reviewing this repo without an API account, and for deployments where document text may not
  leave the premises.

* **Degrades instead of failing.** With no provider at all the system runs in *retrieval-only*
  mode: ranked evidence with citations plus an explicit notice, rather than an exception. This is
  not generation and is labelled as such in the UI — but it means a reviewer with nothing
  configured still sees the retrieval pipeline working instead of a stack trace.
* **Citations are structural, not cosmetic.** The model is given numbered sources and required to
  cite them inline as [n]; markers are then parsed out and resolved back to chunk ids, so every
  claim in the UI is clickable through to a page in a PDF.
* **Abstention is a first-class outcome.** The prompt instructs the model to say the corpus does
  not contain the answer rather than to guess, and that outcome is recorded as `abstained` so the
  evaluation can measure it separately from a wrong answer.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from app.core.config import Settings, get_settings
from app.rag.retrieval import Candidate

ABSTAIN_SENTINEL = "INSUFFICIENT_CONTEXT"

_SYSTEM_PROMPT = f"""\
You are a research assistant answering questions about a fixed collection of academic papers.

Rules:
1. Answer using ONLY the numbered sources provided. Do not use outside knowledge.
2. Cite every factual claim inline with the source number in square brackets, e.g. [2].
   Cite the specific source that supports the claim; do not cite sources you did not use.
3. If the sources do not contain enough information to answer, begin your reply with exactly
   {ABSTAIN_SENTINEL} followed by one sentence explaining what is missing. Do not guess, and do
   not fall back on general knowledge. Never mix an answer and this marker: either the sources
   support an answer, or they do not.
4. Be concise and specific. Prefer the paper's own terminology and exact figures.
5. If sources disagree, say so and cite each side.
6. Earlier turns in the conversation tell you what the user is referring to (e.g. what "that"
   means). They are NOT a source of facts: every factual claim must come from the numbered
   sources in the current turn, even if you stated it in an earlier turn.\
"""


@dataclass
class Answer:
    text: str
    abstained: bool
    citations: list[int] = field(default_factory=list)   # chunk ids, in display order
    prompt_tokens: int = 0
    model: str = ""
    retrieval_only: bool = False


def build_context_block(context: list[Candidate]) -> str:
    """Render retrieved chunks as numbered sources.

    Uses `raw_content` rather than the indexed `content`: if contextualisation is enabled, the
    indexed text carries an LLM-written blurb that helps retrieval but must not be presented to
    the generator as if it were the paper's own words.
    """
    parts = []
    for i, cand in enumerate(context, start=1):
        page = (
            f"p{cand.page_start}"
            if cand.page_start == cand.page_end
            else f"pp{cand.page_start}-{cand.page_end}"
        )
        section = f" · {cand.section}" if cand.section else ""
        parts.append(
            f"[{i}] {cand.title} ({cand.doc_key}, {page}{section})\n{cand.raw_content.strip()}"
        )
    return "\n\n".join(parts)


_ANAPHORA = re.compile(
    r"\b(that|this|these|those|it|its|them|they|the former|the latter|the above)\b",
    re.IGNORECASE,
)
_REFORMULATION = re.compile(
    r"\b(explain|rephrase|reword|simpler|simplify|shorter|shorten|summari[sz]e|elaborate|"
    r"expand|clarify|again|instead|why|how come|tl;?dr)\b",
    re.IGNORECASE,
)


def looks_like_followup(question: str) -> bool:
    """Whether a question depends on the previous turn to be understood.

    Used to decide whether to re-supply the previous answer's evidence. Doing that
    unconditionally is actively harmful: it was observed causing a topic switch
    ("What is the capital of France?") to be answered from the *previous* topic's chunks instead
    of correctly abstaining. So carry-forward applies only to questions that cannot stand alone.

    The signal is a short question containing an anaphor ("that", "it") or a reformulation verb
    ("explain", "simpler") — cheap, deterministic, and no extra LLM call. It is a heuristic and
    will mislabel edge cases; the principled fix is LLM query rewriting, which costs a call on
    every turn and is recorded as future work.
    """
    words = question.split()
    if len(words) > 14:
        return False  # long questions generally carry their own searchable content
    return bool(_ANAPHORA.search(question) or _REFORMULATION.search(question))


def count_context_tokens(context: list[Candidate]) -> int:
    """Token count of the rendered context block.

    Surfaced in the UI so a user can see how much evidence the model actually received — part of
    making the pipeline legible rather than magical.
    """
    from app.rag.chunking import n_tokens

    if not context:
        return 0
    return n_tokens(build_context_block(context)) + n_tokens(_SYSTEM_PROMPT)


_CITATION_GROUP = re.compile(r"\[([\d\s,;–—-]+)\]")


def parse_citations(text: str, context: list[Candidate]) -> list[int]:
    """Resolve inline citation markers to chunk ids, in first-appearance order.

    Handles the several shapes models actually emit, not just the one we ask for: `[1][2]`,
    `[1, 2, 6]`, `[1;2]` and ranges like `[1-3]`. Parsing only `[n]` silently dropped grouped
    citations, which made answers look uncited and broke carry-forward on the next turn.

    Out-of-range markers are ignored rather than raising — a model citing `[9]` when six sources
    were supplied is a hallucinated reference, and must not resolve to an unrelated chunk.
    """
    seen: list[int] = []

    def add(idx: int) -> None:
        if 1 <= idx <= len(context):
            chunk_id = context[idx - 1].chunk_id
            if chunk_id not in seen:
                seen.append(chunk_id)

    for match in _CITATION_GROUP.finditer(text):
        body = match.group(1)
        for part in re.split(r"[,;]", body):
            part = part.strip()
            if not part:
                continue
            span = re.fullmatch(r"(\d{1,2})\s*[–—-]\s*(\d{1,2})", part)
            if span:  # a range, e.g. [1-3]
                start, end = int(span.group(1)), int(span.group(2))
                if start <= end and end - start < 20:
                    for idx in range(start, end + 1):
                        add(idx)
            elif part.isdigit():
                add(int(part))
    return seen


def _retrieval_only_answer(context: list[Candidate]) -> Answer:
    """Fallback when no LLM is configured: present the evidence, clearly labelled."""
    if not context:
        return Answer(
            text=(
                "No matching passages were found in the corpus for this question."
            ),
            abstained=True,
            retrieval_only=True,
        )

    lines = [
        "_Generation is disabled because no LLM API key is configured, so this is the "
        "retrieved evidence rather than a written answer. "
        "Set `LLM_PROVIDER` and the matching API key in `.env` to enable grounded answers._",
        "",
        "**Most relevant passages:**",
        "",
    ]
    for i, cand in enumerate(context, start=1):
        snippet = " ".join(cand.raw_content.split())[:340]
        lines.append(f"[{i}] **{cand.title}** (p{cand.page_start}) — {snippet}…")
        lines.append("")

    return Answer(
        text="\n".join(lines).strip(),
        abstained=False,
        citations=[c.chunk_id for c in context],
        retrieval_only=True,
    )


def stream_answer(
    question: str,
    context: list[Candidate],
    settings: Settings | None = None,
    history: list[tuple[str, str]] | None = None,
) -> Iterator[tuple[str, str]]:
    """Yield ``(event, payload)`` tuples: ``("token", text)`` then a final ``("done", "")``.

    Streaming is deliberate: a wall of text appearing after several silent seconds reads as a
    hang, even at identical total latency.

    `history` is the recent turns of the conversation, oldest first. It is passed to the
    *generator* only — retrieval always runs on the literal question.

    That asymmetry is a deliberate scope decision. Sending history to the generator costs nothing
    extra and handles reformulation follow-ups ("explain that more simply", "shorter please").
    Making *retrieval* conversational is a different and larger problem: a query like "what about
    the second one?" embeds to nothing useful, so it requires an extra LLM call per turn to
    rewrite the question into a standalone form — adding latency, cost, and a new failure mode
    when the rewrite is wrong. Documented as a known limitation rather than half-built.
    """
    settings = settings or get_settings()

    if not settings.has_llm:
        answer = _retrieval_only_answer(context)
        yield ("token", answer.text)
        yield ("done", "")
        return

    prompt = (
        f"Sources:\n\n{build_context_block(context)}\n\n"
        f"Question: {question}\n\n"
        "Answer with inline [n] citations."
    )

    if settings.llm_provider == "anthropic":
        yield from _stream_anthropic(prompt, settings, history or [])
    else:
        # Covers both hosted OpenAI and a local Ollama server, which speaks the same protocol.
        yield from _stream_openai(prompt, settings, history or [])


def _history_messages(history: list[tuple[str, str]], max_chars: int = 1200) -> list[dict]:
    """Render prior turns as chat messages, truncated.

    Older assistant answers are clipped: they exist to establish what "that" refers to, not to be
    re-read in full, and full answers would crowd out retrieved evidence in the context window.
    """
    return [
        {"role": role, "content": content[:max_chars]}
        for role, content in history
        if role in ("user", "assistant") and content.strip()
    ]


def _stream_openai(
    prompt: str, settings: Settings, history: list[tuple[str, str]]
) -> Iterator[tuple[str, str]]:
    """Stream from any OpenAI-protocol endpoint.

    Ollama implements `/v1/chat/completions`, so running a local open-weights model needs only a
    different base URL — no separate client, no second streaming implementation. The API key is
    ignored by Ollama but the SDK requires a non-empty string.
    """
    from openai import OpenAI

    if settings.llm_provider == "ollama":
        client = OpenAI(base_url=settings.ollama_base_url, api_key="ollama")
    else:
        client = OpenAI(api_key=settings.openai_api_key)

    model = settings.active_model
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            *_history_messages(history),
            {"role": "user", "content": prompt},
        ],
        stream=True,
        **_openai_limits(model, settings),
    )
    for event in stream:
        delta = event.choices[0].delta.content if event.choices else None
        if delta:
            yield ("token", delta)
    yield ("done", "")


def _openai_limits(model: str, settings: Settings) -> dict:
    """Per-model request parameters.

    The OpenAI surface is not uniform: reasoning-family models (gpt-5, o-series) require
    `max_completion_tokens` and reject `temperature`, while the gpt-4.x family uses `max_tokens`
    and accepts it. Normalising here keeps the call site free of model trivia and means switching
    models is a config change rather than a code change.
    """
    reasoning_family = model.startswith(("gpt-5", "o1", "o3", "o4"))
    if reasoning_family:
        return {"max_completion_tokens": settings.max_output_tokens}
    return {
        "max_tokens": settings.max_output_tokens,
        "temperature": settings.temperature,
    }


def _stream_anthropic(
    prompt: str, settings: Settings, history: list[tuple[str, str]]
) -> Iterator[tuple[str, str]]:
    from anthropic import Anthropic

    client = Anthropic(api_key=settings.anthropic_api_key)
    model = settings.active_model
    with client.messages.stream(
        model=model,
        system=_SYSTEM_PROMPT,
        messages=[
            *_history_messages(history),
            {"role": "user", "content": prompt},
        ],
        temperature=settings.temperature,
        max_tokens=settings.max_output_tokens,
    ) as stream:
        for text in stream.text_stream:
            yield ("token", text)
    yield ("done", "")


def finalize(text: str, context: list[Candidate], settings: Settings | None = None) -> Answer:
    """Post-process a fully streamed answer into a stored record."""
    settings = settings or get_settings()

    # The sentinel is instructed to lead the reply, but models sometimes emit it mid-sentence or
    # bracketed after an answer. Detect it anywhere: a response that admits missing evidence is
    # an abstention regardless of where it says so, and it must not be recorded as a confident
    # answer. Observed in practice on reformulation follow-ups.
    abstained = ABSTAIN_SENTINEL in text
    cleaned = text
    if abstained:
        cleaned = re.sub(rf"\[?{ABSTAIN_SENTINEL}:?\s*", "", text).strip(" :[]\n") or (
            "The provided documents do not contain enough information to answer this question."
        )
    return Answer(
        text=cleaned,
        abstained=abstained,
        citations=parse_citations(cleaned, context),
        model=settings.active_model,
        retrieval_only=not settings.has_llm,
    )
