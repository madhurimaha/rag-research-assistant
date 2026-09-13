"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.db.pool import close_pool, init_schema

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    init_schema()

    # Warm the local ONNX models at startup rather than on the first user request: loading them
    # costs a few seconds, which would otherwise land on whoever asks the first question.
    try:
        from app.rag.embedder import Embedder, Reranker

        Embedder.instance().embed_query("warmup")
        if settings.use_rerank:
            Reranker.instance().score("warmup", ["warmup"])
        log.info("local models warmed (embedder + reranker)")
    except Exception as exc:  # noqa: BLE001
        log.warning("model warmup failed: %s", exc)

    log.info(
        "ready | provider=%s generation=%s hybrid=%s rerank=%s contextualize=%s",
        settings.llm_provider,
        settings.has_llm,
        settings.use_hybrid,
        settings.use_rerank,
        settings.contextualize,
    )
    yield
    close_pool()


app = FastAPI(
    title="RAG Research Assistant",
    version="0.1.0",
    description=(
        "Hybrid-retrieval RAG over a research-paper corpus. Retrieval, reranking and embeddings "
        "run locally; answer generation uses a configurable provider and degrades to "
        "retrieval-only when none is set."
    ),
    lifespan=lifespan,
)

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _settings.cors_origins.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.routers import auth, chat, documents, system  # noqa: E402

app.include_router(system.router)
app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(chat.router)
