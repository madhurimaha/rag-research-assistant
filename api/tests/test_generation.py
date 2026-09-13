"""Citation parsing, abstention handling and no-key degradation.

These are pure functions over a fabricated context, so they need neither a database nor an API
key — which is the point: the behaviour a reviewer most needs to trust is the cheapest to test.
"""

from app.core.config import Settings
from app.rag.generation import (
    ABSTAIN_SENTINEL,
    build_context_block,
    finalize,
    looks_like_followup,
    parse_citations,
    stream_answer,
)
from app.rag.retrieval import Candidate


def _candidate(chunk_id: int, ordinal: int = 0) -> Candidate:
    return Candidate(
        chunk_id=chunk_id,
        document_id=1,
        doc_key="1234.5678",
        title="A Paper",
        ordinal=ordinal,
        page_start=3,
        page_end=3,
        section="Method",
        content="indexed text",
        raw_content="The model reaches 91.2 F1 on the test split.",
    )


CONTEXT = [_candidate(101), _candidate(102, 1), _candidate(103, 2)]


class TestParseCitations:
    def test_resolves_markers_to_chunk_ids(self):
        assert parse_citations("Claim [1] and claim [3].", CONTEXT) == [101, 103]

    def test_preserves_first_appearance_order(self):
        assert parse_citations("[3] then [1] then [3] again", CONTEXT) == [103, 101]

    def test_ignores_out_of_range_markers(self):
        # A hallucinated [9] must not raise or resolve to an unrelated chunk.
        assert parse_citations("Supported [2], invented [9].", CONTEXT) == [102]

    def test_no_citations_returns_empty(self):
        assert parse_citations("An uncited assertion.", CONTEXT) == []

    # Models do not reliably emit the [n][n] form we ask for. Each of these shapes was either
    # observed in practice or is a near neighbour of one that was.
    def test_comma_grouped_citations(self):
        assert parse_citations("Claims here [1,2,3].", CONTEXT) == [101, 102, 103]

    def test_comma_grouped_with_spaces(self):
        assert parse_citations("Claims here [1, 3].", CONTEXT) == [101, 103]

    def test_semicolon_grouped_citations(self):
        assert parse_citations("Claims here [2;3].", CONTEXT) == [102, 103]

    def test_range_citations(self):
        assert parse_citations("Claims here [1-3].", CONTEXT) == [101, 102, 103]

    def test_grouped_markers_are_range_checked(self):
        # [2,9] — the 9 is hallucinated and must be dropped without losing the valid 2.
        assert parse_citations("Mixed [2,9].", CONTEXT) == [102]

    def test_ignores_non_citation_brackets(self):
        assert parse_citations("An array [x] and [] and [1].", CONTEXT) == [101]


class TestAbstention:
    def test_sentinel_marks_answer_as_abstained(self):
        answer = finalize(
            f"{ABSTAIN_SENTINEL}: the papers do not report training cost.",
            CONTEXT,
            Settings(llm_provider="none"),
        )
        assert answer.abstained is True
        assert ABSTAIN_SENTINEL not in answer.text
        assert answer.text.startswith("the papers do not report")

    def test_normal_answer_is_not_abstained(self):
        answer = finalize("It reaches 91.2 F1 [1].", CONTEXT, Settings(llm_provider="none"))
        assert answer.abstained is False
        assert answer.citations == [101]

    def test_bare_sentinel_gets_a_readable_fallback(self):
        answer = finalize(ABSTAIN_SENTINEL, CONTEXT, Settings(llm_provider="none"))
        assert answer.abstained is True
        assert "do not contain enough information" in answer.text

    def test_sentinel_after_an_answer_still_counts_as_abstention(self):
        # Observed on follow-ups: the model answers, then admits the sources did not cover it.
        # That is an abstention, not a confident answer, wherever the marker appears.
        answer = finalize(
            f"The task does X [1]. [{ABSTAIN_SENTINEL}: sources do not mention X.]",
            CONTEXT,
            Settings(llm_provider="none"),
        )
        assert answer.abstained is True
        assert ABSTAIN_SENTINEL not in answer.text


class TestNoKeyDegradation:
    def test_returns_evidence_instead_of_raising(self):
        settings = Settings(llm_provider="none")
        assert settings.has_llm is False

        events = list(stream_answer("Any question?", CONTEXT, settings))
        text = "".join(payload for event, payload in events if event == "token")

        assert events[-1][0] == "done"
        assert "no LLM API key is configured" in text
        assert "91.2 F1" in text  # the actual evidence is shown

    def test_empty_context_is_handled(self):
        events = list(stream_answer("Any question?", [], Settings(llm_provider="none")))
        text = "".join(payload for event, payload in events if event == "token")
        assert "No matching passages" in text


class TestFollowupDetection:
    """Gates whether the previous turn's evidence is re-supplied.

    Both directions matter. A missed follow-up loses grounding on "explain that"; a false
    positive re-supplies stale evidence and was observed answering a topic change from the
    *previous* topic instead of abstaining.
    """

    def test_anaphoric_question_is_a_followup(self):
        assert looks_like_followup("Explain that in one sentence.") is True
        assert looks_like_followup("Tell me more about it") is True

    def test_reformulation_request_is_a_followup(self):
        assert looks_like_followup("Can you simplify?") is True
        assert looks_like_followup("Why?") is True

    def test_self_contained_question_is_not(self):
        assert looks_like_followup("What is the capital of France?") is False
        assert looks_like_followup("What is the QA-CTS task?") is False

    def test_long_question_is_not_a_followup(self):
        # Carries its own searchable content even though it contains "that".
        long_q = (
            "Which of the datasets described in that paper were used for the ablation study "
            "reported in the results section?"
        )
        assert looks_like_followup(long_q) is False


class TestContextBlock:
    def test_numbers_sources_from_one(self):
        block = build_context_block(CONTEXT)
        assert block.startswith("[1] A Paper (1234.5678, p3 · Method)")
        assert "[3]" in block

    def test_uses_raw_text_not_indexed_text(self):
        # With contextualisation enabled the indexed text carries an LLM-written blurb; the
        # generator must see the paper's own words instead.
        block = build_context_block(CONTEXT)
        assert "The model reaches 91.2 F1" in block
        assert "indexed text" not in block

    def test_renders_page_ranges(self):
        spanning = _candidate(104)
        spanning.page_end = 5
        assert "pp3-5" in build_context_block([spanning])
