"""Chat endpoints.

`POST /chat/ask` streams the answer over Server-Sent Events. Streaming is deliberate: an answer
that appears all at once after several seconds of silence reads as a hang, even when total
latency is identical.

Event sequence:
    meta      conversation/message ids, retrieval config, timings  (sent before the first token)
    citation  one per source, so the UI can render chips as they become known
    token     answer text deltas
    done      final record: abstention flag, latency, resolved citations
    error     transport-level failure

Sending `meta` first means the client can show which documents are being read while the model is
still writing.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.core.auth import get_current_user
from app.core.config import get_settings
from app.db.pool import connection
from app.rag import generation
from app.rag.retrieval import Candidate, retrieve
from app.schemas.models import (
    AskRequest,
    CitationOut,
    ConversationOut,
    ExplainOut,
    MessageOut,
    TraceRowOut,
)

router = APIRouter(tags=["chat"])

# Exchanges of prior context given to the generator. Kept small deliberately: history competes
# with retrieved evidence for the context window, and evidence is what grounds the answer.
HISTORY_TURNS = 2


_ABSTRACT_RE = re.compile(r"(?i)\babstract\b[\s:.—–\-]*")


def _snippet(text: str, limit: int = 260) -> str:
    """Quote the body of a chunk, not the author block that often leads page 1."""
    compact = " ".join(text.split())
    match = _ABSTRACT_RE.search(compact)
    if match:
        rest = compact[match.end() :].lstrip()
        if len(rest) > 40:
            compact = rest
    return compact[:limit]


def _sse(event: str, data: dict | str) -> str:
    payload = json.dumps(data) if isinstance(data, dict) else json.dumps({"text": data})
    return f"event: {event}\ndata: {payload}\n\n"


def _persist_traces(conn, message_id: int, candidates: list[Candidate]) -> None:
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO retrieval_traces
                (message_id, chunk_id, vector_rank, vector_score, lexical_rank, lexical_score,
                 rrf_score, rrf_rank, rerank_score, final_rank, used_in_context)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (
                    message_id,
                    c.chunk_id,
                    c.vector_rank,
                    c.vector_score,
                    c.lexical_rank,
                    c.lexical_score,
                    c.rrf_score,
                    c.rrf_rank,
                    c.rerank_score,
                    c.final_rank,
                    c.used_in_context,
                )
                for c in candidates
            ],
        )


@router.post("/chat/ask")
def ask(req: AskRequest, user: dict = Depends(get_current_user)) -> StreamingResponse:
    settings = get_settings()
    user_id = user["id"]

    def event_stream() -> Iterator[str]:
        started = time.perf_counter()
        try:
            with connection() as conn:
                # ---- conversation (owned by this user)
                if req.conversation_id:
                    row = conn.execute(
                        "SELECT id FROM conversations WHERE id = %s AND user_id = %s",
                        (req.conversation_id, user_id),
                    ).fetchone()
                    if not row:
                        yield _sse("error", {"message": "conversation not found"})
                        return
                    conversation_id = req.conversation_id
                else:
                    conversation_id = conn.execute(
                        "INSERT INTO conversations (user_id, title) VALUES (%s, %s) RETURNING id",
                        (user_id, req.question[:120]),
                    ).fetchone()["id"]

                # Recent turns, fetched *before* inserting the current question. Passed to the
                # generator only — retrieval runs on the literal question (see generation docs).
                history = [
                    (row["role"], row["content"])
                    for row in reversed(
                        conn.execute(
                            """
                            SELECT role, content FROM messages
                            WHERE conversation_id = %s
                            ORDER BY id DESC
                            LIMIT %s
                            """,
                            (conversation_id, HISTORY_TURNS * 2),
                        ).fetchall()
                    )
                ]

                conn.execute(
                    "INSERT INTO messages (conversation_id, role, content) VALUES (%s,'user',%s)",
                    (conversation_id, req.question),
                )

                # Chunks the previous answer cited, re-supplied so a reformulation follow-up is
                # grounded in the same evidence rather than whatever its literal text retrieves.
                # Gated on the question actually being a follow-up: carrying evidence forward
                # unconditionally makes a topic change answer from the old topic's chunks
                # instead of abstaining.
                prior_chunk_ids: list[int] = []
                if history and generation.looks_like_followup(req.question):
                    prior_chunk_ids = [
                        row["chunk_id"]
                        for row in conn.execute(
                            """
                            SELECT ct.chunk_id
                            FROM citations ct
                            JOIN messages m ON m.id = ct.message_id
                            WHERE m.conversation_id = %s
                              AND m.id = (SELECT max(id) FROM messages
                                          WHERE conversation_id = %s AND role = 'assistant')
                            ORDER BY ct.marker
                            """,
                            (conversation_id, conversation_id),
                        ).fetchall()
                    ]

                # ---- retrieve
                result = retrieve(
                    conn,
                    req.question,
                    settings,
                    use_hybrid=req.use_hybrid,
                    use_rerank=req.use_rerank,
                    carry_forward_chunk_ids=prior_chunk_ids,
                    user_id=user_id,
                )

                prompt_tokens = generation.count_context_tokens(result.context)
                yield _sse(
                    "meta",
                    {
                        "conversation_id": conversation_id,
                        "config": result.config,
                        "timings_ms": {k: round(v, 1) for k, v in result.timings_ms.items()},
                        "n_candidates": len(result.candidates),
                        "prompt_tokens": prompt_tokens,
                        "generation_enabled": settings.has_llm,
                        "model": settings.active_model,
                    },
                )

                for i, cand in enumerate(result.context, start=1):
                    yield _sse(
                        "citation",
                        {
                            "marker": i,
                            "chunk_id": cand.chunk_id,
                            "document_id": cand.document_id,
                            "doc_key": cand.doc_key,
                            "title": cand.title,
                            "page_start": cand.page_start,
                            "page_end": cand.page_end,
                            "section": cand.section,
                            "snippet": _snippet(cand.raw_content),
                        },
                    )

                # ---- generate
                chunks: list[str] = []
                for event, payload in generation.stream_answer(
                    req.question, result.context, settings, history=history
                ):
                    if event == "token":
                        chunks.append(payload)
                        yield _sse("token", payload)

                answer = generation.finalize("".join(chunks), result.context, settings)
                latency_ms = int((time.perf_counter() - started) * 1000)

                # ---- persist
                message_id = conn.execute(
                    """
                    INSERT INTO messages
                        (conversation_id, role, content, abstained, latency_ms, prompt_tokens)
                    VALUES (%s, 'assistant', %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        conversation_id,
                        answer.text,
                        answer.abstained,
                        latency_ms,
                        prompt_tokens,
                    ),
                ).fetchone()["id"]

                marker_by_chunk = {c.chunk_id: i for i, c in enumerate(result.context, start=1)}
                if answer.citations:
                    with conn.cursor() as cur:
                        cur.executemany(
                            """
                            INSERT INTO citations (message_id, chunk_id, marker)
                            VALUES (%s, %s, %s)
                            ON CONFLICT (message_id, marker) DO NOTHING
                            """,
                            [
                                (message_id, chunk_id, marker_by_chunk[chunk_id])
                                for chunk_id in answer.citations
                                if chunk_id in marker_by_chunk
                            ],
                        )

                _persist_traces(conn, message_id, result.candidates)

                yield _sse(
                    "done",
                    {
                        "message_id": message_id,
                        "conversation_id": conversation_id,
                        "abstained": answer.abstained,
                        "retrieval_only": answer.retrieval_only,
                        "model": answer.model,
                        "latency_ms": latency_ms,
                        "prompt_tokens": prompt_tokens,
                        "cited_chunk_ids": answer.citations,
                    },
                )
        except Exception as exc:  # noqa: BLE001
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # prevent proxy buffering from defeating streaming
        },
    )


@router.get("/chat/{message_id}/explain", response_model=ExplainOut)
def explain(message_id: int, user: dict = Depends(get_current_user)) -> ExplainOut:
    """Replay how a stored answer was retrieved.

    Traces are persisted rather than logged so this works for any historical message, not just
    the one currently on screen.
    """
    with connection() as conn:
        message = conn.execute(
            """
            SELECT m.id, m.conversation_id, m.content
            FROM messages m
            JOIN conversations c ON c.id = m.conversation_id
            WHERE m.id = %s AND c.user_id = %s
            """,
            (message_id, user["id"]),
        ).fetchone()
        if not message:
            raise HTTPException(404, "message not found")

        question = conn.execute(
            """
            SELECT content FROM messages
            WHERE conversation_id = %s AND role = 'user' AND id < %s
            ORDER BY id DESC LIMIT 1
            """,
            (message["conversation_id"], message_id),
        ).fetchone()

        rows = conn.execute(
            """
            SELECT t.chunk_id, t.vector_rank, t.vector_score, t.lexical_rank, t.lexical_score,
                   t.rrf_score, t.rrf_rank, t.rerank_score, t.final_rank, t.used_in_context,
                   c.page_start, c.section, c.raw_content, d.doc_key, d.title
            FROM retrieval_traces t
            JOIN chunks c ON c.id = t.chunk_id
            JOIN documents d ON d.id = c.document_id
            WHERE t.message_id = %s
            ORDER BY COALESCE(t.final_rank, 9999), t.rrf_rank
            """,
            (message_id,),
        ).fetchall()

    return ExplainOut(
        message_id=message_id,
        question=question["content"] if question else "",
        config={},
        timings_ms={},
        candidates=[
            TraceRowOut(
                chunk_id=r["chunk_id"],
                doc_key=r["doc_key"],
                title=r["title"],
                page_start=r["page_start"],
                section=r["section"],
                snippet=_snippet(r["raw_content"], 200),
                vector_rank=r["vector_rank"],
                vector_score=r["vector_score"],
                lexical_rank=r["lexical_rank"],
                lexical_score=r["lexical_score"],
                rrf_score=r["rrf_score"],
                rrf_rank=r["rrf_rank"],
                rerank_score=r["rerank_score"],
                final_rank=r["final_rank"],
                used_in_context=r["used_in_context"],
            )
            for r in rows
        ],
    )


@router.get("/conversations", response_model=list[ConversationOut])
def list_conversations(user: dict = Depends(get_current_user)) -> list[ConversationOut]:
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT id, title, created_at FROM conversations
            WHERE user_id = %s
            ORDER BY id DESC LIMIT 50
            """,
            (user["id"],),
        ).fetchall()
    return [ConversationOut(**row) for row in rows]


@router.get("/conversations/{conversation_id}", response_model=ConversationOut)
def get_conversation(
    conversation_id: int, user: dict = Depends(get_current_user)
) -> ConversationOut:
    with connection() as conn:
        convo = conn.execute(
            "SELECT id, title, created_at FROM conversations WHERE id = %s AND user_id = %s",
            (conversation_id, user["id"]),
        ).fetchone()
        if not convo:
            raise HTTPException(404, "conversation not found")

        messages = conn.execute(
            """
            SELECT id, role, content, abstained, latency_ms, created_at
            FROM messages WHERE conversation_id = %s ORDER BY id
            """,
            (conversation_id,),
        ).fetchall()

        citation_rows = conn.execute(
            """
            SELECT ct.message_id, ct.marker, ct.chunk_id, d.id AS document_id,
                   c.page_start, c.page_end, c.section,
                   c.raw_content, d.doc_key, d.title
            FROM citations ct
            JOIN chunks c ON c.id = ct.chunk_id
            JOIN documents d ON d.id = c.document_id
            JOIN messages m ON m.id = ct.message_id
            WHERE m.conversation_id = %s
            ORDER BY ct.message_id, ct.marker
            """,
            (conversation_id,),
        ).fetchall()

    by_message: dict[int, list[CitationOut]] = {}
    for row in citation_rows:
        by_message.setdefault(row["message_id"], []).append(
            CitationOut(
                marker=row["marker"],
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                doc_key=row["doc_key"],
                title=row["title"],
                page_start=row["page_start"],
                page_end=row["page_end"],
                section=row["section"],
                snippet=_snippet(row["raw_content"]),
            )
        )

    return ConversationOut(
        **convo,
        messages=[
            MessageOut(**msg, citations=by_message.get(msg["id"], [])) for msg in messages
        ],
    )
