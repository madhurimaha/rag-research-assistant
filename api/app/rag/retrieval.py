"""Hybrid retrieval: vector arm + lexical arm, fused with Reciprocal Rank Fusion.

Why two arms
------------
They fail in different, complementary ways:

  * The **vector arm** matches meaning. A query about "hallucination" finds a passage about
    "unfaithful generation" with no shared vocabulary. But an embedding is lossy compression —
    it blurs exact tokens, so product codes, proper nouns, version strings and specific figures
    often fall out of the top-k.
  * The **lexical arm** matches literal text, so it recovers precisely those: "Section 4.2",
    "bge-reranker-v2-m3", "2,714".

Real questions contain both kinds of signal, so we run both and merge.

Why RRF rather than weighted score blending
-------------------------------------------
Cosine distance lives on roughly [0, 2]; `ts_rank_cd` is an unbounded relevance number. Blending
them directly requires normalising two incompatible scales, which turns into a hunt for magic
constants that then need retuning whenever the corpus changes. RRF ignores scores entirely and
uses only *rank position*:

    score(chunk) = Σ_arms 1 / (k + rank_in_arm)      (k = 60 by convention)

It is parameter-light, robust, and expressible in one SQL statement — so fusion happens in the
database, next to the indexes, rather than by shipping two large result sets into Python.

The trace
---------
Every candidate's per-arm rank, fused score and rerank score is returned alongside the results.
That is what makes the pipeline explainable rather than a black box, and it is why fusion is
written out explicitly instead of being delegated to a library.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.config import Settings, get_settings
from app.rag.embedder import Embedder, Reranker


@dataclass
class Candidate:
    """A retrieved chunk plus the full provenance of how it was ranked."""

    chunk_id: int
    document_id: int
    doc_key: str
    title: str
    ordinal: int
    page_start: int
    page_end: int
    section: str | None
    content: str
    raw_content: str

    vector_rank: int | None = None
    vector_score: float | None = None
    lexical_rank: int | None = None
    lexical_score: float | None = None
    rrf_score: float = 0.0
    rrf_rank: int = 0
    rerank_score: float | None = None
    final_rank: int | None = None
    used_in_context: bool = False
    carried_forward: bool = False  # evidence reused from the previous turn, not retrieved now


@dataclass
class RetrievalResult:
    query: str
    candidates: list[Candidate] = field(default_factory=list)  # all considered, ranked
    context: list[Candidate] = field(default_factory=list)     # subset sent to the generator
    timings_ms: dict[str, float] = field(default_factory=dict)
    config: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------- SQL
#
# Two CTEs, one per arm, each independently ranked; then a FULL OUTER JOIN so a chunk found by
# only one arm still survives, and the RRF sum in the final projection.
#
# Notes:
#  * `1 - (embedding <=> q)` converts cosine distance to similarity for display only; ordering
#    uses the distance operator directly so the HNSW index is used.
#  * websearch_to_tsquery tolerates natural-language input (quotes, OR, -negation) instead of
#    raising on punctuation the way to_tsquery does.
#  * COALESCE(..., 0) implements "absent from this arm contributes nothing".

_HYBRID_SQL = """
WITH vector_arm AS (
    SELECT id AS chunk_id,
           ROW_NUMBER() OVER (ORDER BY embedding <=> %(qvec)s::vector) AS rank,
           1 - (embedding <=> %(qvec)s::vector) AS score
    FROM chunks
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> %(qvec)s::vector
    LIMIT %(per_arm)s
),
lexical_arm AS (
    SELECT id AS chunk_id,
           ROW_NUMBER() OVER (ORDER BY ts_rank_cd(tsv, query) DESC) AS rank,
           ts_rank_cd(tsv, query) AS score
    FROM chunks, websearch_to_tsquery('english', %(qtext)s) AS query
    WHERE tsv @@ query
    ORDER BY ts_rank_cd(tsv, query) DESC
    LIMIT %(per_arm)s
),
fused AS (
    SELECT COALESCE(v.chunk_id, l.chunk_id) AS chunk_id,
           v.rank  AS vector_rank,
           v.score AS vector_score,
           l.rank  AS lexical_rank,
           l.score AS lexical_score,
           COALESCE(1.0 / (%(rrf_k)s + v.rank), 0.0)
         + COALESCE(1.0 / (%(rrf_k)s + l.rank), 0.0) AS rrf_score
    FROM vector_arm v
    FULL OUTER JOIN lexical_arm l ON v.chunk_id = l.chunk_id
)
SELECT f.chunk_id, f.vector_rank, f.vector_score, f.lexical_rank, f.lexical_score, f.rrf_score,
       c.document_id, c.ordinal, c.page_start, c.page_end, c.section, c.content, c.raw_content,
       d.doc_key, d.title
FROM fused f
JOIN chunks c ON c.id = f.chunk_id
JOIN documents d ON d.id = c.document_id
ORDER BY f.rrf_score DESC
LIMIT %(limit)s;
"""

# Vector-only baseline, used as the ablation floor. Same projection shape so downstream code
# does not branch.
_VECTOR_ONLY_SQL = """
SELECT c.id AS chunk_id,
       ROW_NUMBER() OVER (ORDER BY c.embedding <=> %(qvec)s::vector) AS vector_rank,
       1 - (c.embedding <=> %(qvec)s::vector) AS vector_score,
       NULL::bigint AS lexical_rank,
       NULL::real   AS lexical_score,
       1.0 / (%(rrf_k)s + ROW_NUMBER() OVER (ORDER BY c.embedding <=> %(qvec)s::vector))
           AS rrf_score,
       c.document_id, c.ordinal, c.page_start, c.page_end, c.section, c.content, c.raw_content,
       d.doc_key, d.title
FROM chunks c
JOIN documents d ON d.id = c.document_id
WHERE c.embedding IS NOT NULL
ORDER BY c.embedding <=> %(qvec)s::vector
LIMIT %(limit)s;
"""


_BY_ID_SQL = """
SELECT c.id AS chunk_id, c.document_id, c.ordinal, c.page_start, c.page_end, c.section,
       c.content, c.raw_content, d.doc_key, d.title
FROM chunks c
JOIN documents d ON d.id = c.document_id
WHERE c.id = ANY(%(ids)s)
"""


def fetch_candidates(conn, chunk_ids: list[int]) -> list[Candidate]:
    """Load specific chunks as candidates, bypassing ranking.

    Used to carry a previous turn's cited evidence into a follow-up question. Ranking fields stay
    empty because these chunks were not retrieved for the current query — the trace should not
    imply they were.
    """
    if not chunk_ids:
        return []
    rows = conn.execute(_BY_ID_SQL, {"ids": chunk_ids}).fetchall()
    by_id = {
        r["chunk_id"]: Candidate(
            chunk_id=r["chunk_id"],
            document_id=r["document_id"],
            doc_key=r["doc_key"],
            title=r["title"],
            ordinal=r["ordinal"],
            page_start=r["page_start"],
            page_end=r["page_end"],
            section=r["section"],
            content=r["content"],
            raw_content=r["raw_content"],
            carried_forward=True,
        )
        for r in rows
    }
    return [by_id[cid] for cid in chunk_ids if cid in by_id]


def retrieve(
    conn,
    query: str,
    settings: Settings | None = None,
    *,
    use_hybrid: bool | None = None,
    use_rerank: bool | None = None,
    top_k: int | None = None,
    carry_forward_chunk_ids: list[int] | None = None,
) -> RetrievalResult:
    """Run retrieval and return ranked candidates plus the trace.

    The per-call overrides exist so the evaluation harness can sweep ablation configurations
    against one warm process and one warm index.
    """
    import time

    settings = settings or get_settings()
    use_hybrid = settings.use_hybrid if use_hybrid is None else use_hybrid
    use_rerank = settings.use_rerank if use_rerank is None else use_rerank
    top_k = settings.final_context_chunks if top_k is None else top_k

    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    qvec = Embedder.instance().embed_query(query)
    timings["embed_query"] = (time.perf_counter() - t0) * 1000

    # ef_search is set per-transaction rather than globally: a background reindex and a
    # user-facing query want different recall/latency trade-offs.
    # SET does not accept bound parameters, so the value is coerced to int and range-checked
    # before interpolation — never string-formatted from untrusted input.
    ef_search = max(1, min(1000, int(settings.hnsw_ef_search)))
    conn.execute(f"SET LOCAL hnsw.ef_search = {ef_search}")

    params = {
        "qvec": str(qvec),
        "qtext": query,
        "per_arm": settings.candidates_per_arm,
        "rrf_k": settings.rrf_k,
        "limit": settings.candidates_per_arm,
    }

    t0 = time.perf_counter()
    sql = _HYBRID_SQL if use_hybrid else _VECTOR_ONLY_SQL
    rows = conn.execute(sql, params).fetchall()
    timings["sql"] = (time.perf_counter() - t0) * 1000

    candidates = [
        Candidate(
            chunk_id=r["chunk_id"],
            document_id=r["document_id"],
            doc_key=r["doc_key"],
            title=r["title"],
            ordinal=r["ordinal"],
            page_start=r["page_start"],
            page_end=r["page_end"],
            section=r["section"],
            content=r["content"],
            raw_content=r["raw_content"],
            vector_rank=r["vector_rank"],
            vector_score=float(r["vector_score"]) if r["vector_score"] is not None else None,
            lexical_rank=r["lexical_rank"],
            lexical_score=float(r["lexical_score"]) if r["lexical_score"] is not None else None,
            rrf_score=float(r["rrf_score"]),
            rrf_rank=i + 1,
        )
        for i, r in enumerate(rows)
    ]

    # ---- stage 2: cross-encoder rerank
    #
    # Only the top `rerank_depth` fused candidates are rescored. Cross-encoder cost is linear in
    # candidate count (~47ms/doc on CPU here), so reranking all 50 would add ~2.4s to every
    # query. Reranked chunks are placed above un-reranked ones, which keeps the fused tail
    # available for the trace without letting it displace a rescored result.
    if use_rerank and candidates:
        t0 = time.perf_counter()
        depth = min(settings.rerank_depth, len(candidates))
        head, tail = candidates[:depth], candidates[depth:]
        scores = Reranker.instance().score(query, [c.content for c in head])
        for cand, score in zip(head, scores, strict=True):
            cand.rerank_score = score
        head.sort(key=lambda c: c.rerank_score, reverse=True)
        candidates = head + tail
        timings["rerank"] = (time.perf_counter() - t0) * 1000

    for i, cand in enumerate(candidates):
        cand.final_rank = i + 1

    context = candidates[:top_k]

    # ---- carry forward the previous turn's cited evidence
    #
    # A reformulation follow-up ("explain that more simply") retrieves poorly, because the
    # literal query contains no searchable content. Without this, the generator would hold the
    # topic in conversation history while the sources in front of it discuss something else —
    # which either breaks grounding or forces a spurious abstention.
    #
    # Re-supplying the chunks the previous answer cited keeps the follow-up grounded in the same
    # evidence, at no extra LLM call. They are appended, never displacing a freshly retrieved
    # chunk, and are flagged in the trace so the explainability panel shows them as reused.
    if carry_forward_chunk_ids:
        present = {c.chunk_id for c in context}
        extra = [
            c
            for c in fetch_candidates(conn, carry_forward_chunk_ids)
            if c.chunk_id not in present
        ]
        context = context + extra[: settings.carry_forward_limit]

    for cand in context:
        cand.used_in_context = True

    # Carried-forward chunks belong in the trace too, so the panel accounts for every chunk the
    # generator saw.
    traced = candidates + [c for c in context if c.carried_forward]

    return RetrievalResult(
        query=query,
        candidates=traced,
        context=context,
        timings_ms=timings,
        config={
            "use_hybrid": use_hybrid,
            "use_rerank": use_rerank,
            "rerank_depth": settings.rerank_depth if use_rerank else None,
            "candidates_per_arm": settings.candidates_per_arm,
            "rrf_k": settings.rrf_k,
            "hnsw_ef_search": settings.hnsw_ef_search,
            "top_k": top_k,
            "contextualized_index": settings.contextualize,
            "carried_forward": sum(1 for c in context if c.carried_forward),
        },
    )
