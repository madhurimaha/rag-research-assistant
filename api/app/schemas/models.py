"""Request and response models.

The API is typed end-to-end so the frontend can be generated against it and so malformed input
fails at the boundary rather than inside the pipeline.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


# ---------------------------------------------------------------- documents
class DocumentOut(BaseModel):
    id: int
    doc_key: str
    title: str
    filename: str
    source: str
    n_pages: int | None
    n_chunks: int
    status: str
    error: str | None
    created_at: datetime
    ingested_at: datetime | None


class IngestResponse(BaseModel):
    document_id: int
    doc_key: str
    n_pages: int
    n_chunks: int
    contextualized: bool
    seconds: float


# ---------------------------------------------------------------- chat
class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    conversation_id: int | None = None
    # Per-request overrides let the UI demonstrate the ablation live.
    use_hybrid: bool | None = None
    use_rerank: bool | None = None


class SignupRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=8, max_length=200)


class LoginRequest(BaseModel):
    email: str
    password: str


class UserOut(BaseModel):
    id: int
    email: str


class AuthOut(BaseModel):
    token: str
    user: UserOut


class CitationOut(BaseModel):
    marker: int
    chunk_id: int
    document_id: int
    doc_key: str
    title: str
    page_start: int
    page_end: int
    section: str | None
    snippet: str


class TraceRowOut(BaseModel):
    """One candidate chunk and how each retrieval stage ranked it.

    This is the payload behind the "why this answer" panel — it makes the ranking process
    inspectable rather than asking the user to trust it.
    """

    chunk_id: int
    doc_key: str
    title: str
    page_start: int
    section: str | None
    snippet: str
    vector_rank: int | None
    vector_score: float | None
    lexical_rank: int | None
    lexical_score: float | None
    rrf_score: float
    rrf_rank: int
    rerank_score: float | None
    final_rank: int | None
    used_in_context: bool


class AnswerOut(BaseModel):
    message_id: int
    conversation_id: int
    answer: str
    abstained: bool
    retrieval_only: bool
    model: str
    citations: list[CitationOut]
    latency_ms: int
    prompt_tokens: int
    config: dict


class ExplainOut(BaseModel):
    message_id: int
    question: str
    config: dict
    timings_ms: dict
    candidates: list[TraceRowOut]


# ---------------------------------------------------------------- conversations
class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    abstained: bool
    latency_ms: int | None
    created_at: datetime
    citations: list[CitationOut] = []


class ConversationOut(BaseModel):
    id: int
    title: str | None
    created_at: datetime
    messages: list[MessageOut] = []


# ---------------------------------------------------------------- system
class HealthOut(BaseModel):
    status: str
    database: bool
    redis: bool
    documents: int
    chunks: int
    llm_provider: str
    generation_enabled: bool
    model: str
    retrieval_config: dict
