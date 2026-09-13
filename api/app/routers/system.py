"""Health and configuration introspection.

`/health` reports which capabilities are actually live rather than just returning 200. The
frontend uses `generation_enabled` to show an honest banner when running in retrieval-only mode,
so a missing API key is a visible, explained state instead of a confusing empty answer.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.core.config import get_settings
from app.db.pool import connection
from app.schemas.models import HealthOut

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    settings = get_settings()

    database = False
    documents = chunks = 0
    try:
        with connection() as conn:
            row = conn.execute(
                """
                SELECT (SELECT count(*) FROM documents WHERE status = 'ready') AS documents,
                       (SELECT count(*) FROM chunks WHERE embedding IS NOT NULL) AS chunks
                """
            ).fetchone()
            documents, chunks = row["documents"], row["chunks"]
            database = True
    except Exception:  # noqa: BLE001
        pass

    redis_ok = False
    if settings.redis_url:
        try:
            import redis

            redis_ok = bool(redis.Redis.from_url(settings.redis_url, socket_timeout=1).ping())
        except Exception:  # noqa: BLE001
            redis_ok = False

    return HealthOut(
        status="ok" if database else "degraded",
        database=database,
        redis=redis_ok,
        documents=documents,
        chunks=chunks,
        llm_provider=settings.llm_provider,
        generation_enabled=settings.has_llm,
        model=settings.active_model,
        retrieval_config={
            "use_hybrid": settings.use_hybrid,
            "use_rerank": settings.use_rerank,
            "rerank_depth": settings.rerank_depth,
            "candidates_per_arm": settings.candidates_per_arm,
            "rrf_k": settings.rrf_k,
            "hnsw_ef_search": settings.hnsw_ef_search,
            "final_context_chunks": settings.final_context_chunks,
            "contextualize": settings.contextualize,
            "embedding_model": settings.embedding_model,
            "rerank_model": settings.rerank_model,
        },
    )
