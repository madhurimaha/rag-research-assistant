"""Embeddings and reranking — both local ONNX models, no API key required.

Why local:
  * An embedding model is a small BERT-family *encoder* (~100MB), not an LLM. It emits one
    fixed-size vector per input and generates no text, so CPU inference is fast enough:
    ~13ms/text on an M1 Pro, i.e. the whole 12-document corpus in under a second.
  * Document text never leaves the machine, which is a hard requirement in regulated settings
    and a sensible default everywhere else.
  * Hosted embeddings (text-embedding-3-large, Cohere, Voyage) score marginally better on hard
    benchmarks; at this corpus size that difference is a rounding error.

Both classes are lazily loaded singletons: model init costs a few seconds and must not happen
per-request. Swapping in a hosted provider means implementing the same two methods.
"""

from __future__ import annotations

import threading

from app.core.config import get_settings


class Embedder:
    """Local ONNX bi-encoder. Query and passage are embedded independently."""

    _instance: Embedder | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        from fastembed import TextEmbedding

        settings = get_settings()
        self.model_name = settings.embedding_model
        self.dim = settings.embedding_dim
        self._model = TextEmbedding(self.model_name)

    @classmethod
    def instance(cls) -> Embedder:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        # bge models are trained with an asymmetric query prefix; using it improves
        # query/passage alignment at zero cost.
        prefixed = f"Represent this sentence for searching relevant passages: {text}"
        return next(iter(self._model.embed([prefixed]))).tolist()


class Reranker:
    """Local ONNX cross-encoder.

    A bi-encoder embeds query and document separately, so it never sees them together. A
    cross-encoder scores the *pair*, which is markedly more accurate but too slow to run over a
    whole corpus. Hence the standard two-stage shape: retrieve ~50 candidates cheaply, then
    rerank those down to the handful that reach the prompt.
    """

    _instance: Reranker | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        settings = get_settings()
        self.model_name = settings.rerank_model
        self._model = TextCrossEncoder(self.model_name)

    @classmethod
    def instance(cls) -> Reranker:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def score(self, query: str, documents: list[str]) -> list[float]:
        """Relevance score per document. Higher is better; scale is model-specific."""
        if not documents:
            return []
        return [float(s) for s in self._model.rerank(query, documents)]
