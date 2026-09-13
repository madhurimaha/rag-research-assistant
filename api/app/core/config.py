"""Application settings.

Every tunable that affects retrieval quality is exposed here rather than hard-coded, so the
evaluation harness can sweep configurations (the ablation table) without touching pipeline code.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"), env_file_encoding="utf-8", extra="ignore"
    )

    # ---------------------------------------------------------------- infra
    database_url: str = "postgresql://rag:rag@localhost:5435/rag"
    redis_url: str | None = "redis://localhost:6380/0"
    cors_origins: str = "http://localhost:3000"

    # ---------------------------------------------------------------- LLM provider
    # Provider is pluggable across three tiers:
    #   "openai"/"anthropic" — hosted, best quality, needs a key
    #   "ollama"             — local open-weights model, real generation, no key, no cost
    #   "none"               — retrieval-only mode; the app still runs and shows evidence
    llm_provider: Literal["openai", "anthropic", "ollama", "none"] = "none"
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    # Ollama exposes an OpenAI-compatible /v1 endpoint, so it reuses the OpenAI client with a
    # different base URL rather than needing its own transport code.
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "qwen3:8b"

    generation_model: str = ""  # empty -> provider default
    context_model: str = ""     # cheap model for chunk contextualisation
    judge_model: str = ""       # eval judge; ideally a different family than generation

    # ---------------------------------------------------------------- embeddings
    # Small BERT-family encoder run locally via ONNX. No API key, no network at query time.
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # ---------------------------------------------------------------- chunking
    chunk_target_tokens: int = 320
    chunk_overlap_tokens: int = 64

    # Index-time enrichment (Anthropic-style contextual retrieval). Flagged because it is the
    # one pipeline stage that needs an LLM, and turning it off must leave a working system.
    contextualize: bool = False

    # ---------------------------------------------------------------- retrieval
    candidates_per_arm: int = 50   # top-k from each of the vector and lexical arms
    rrf_k: int = 60                # RRF damping constant
    hnsw_ef_search: int = 100      # set per-transaction, not globally
    final_context_chunks: int = 6  # chunks passed to the generator
    # Chunks re-supplied from the previous turn's citations so reformulation follow-ups stay
    # grounded. Bounded because history must not crowd out freshly retrieved evidence.
    carry_forward_limit: int = 3

    use_hybrid: bool = True        # False -> vector-only (ablation baseline)
    use_rerank: bool = True
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    # Cross-encoder cost is linear in candidates (~47ms/doc on CPU), so the reranker sees only
    # the top slice of the fused list. Fusion still considers all `candidates_per_arm` results;
    # this bounds only the expensive second stage.
    rerank_depth: int = 20

    # ---------------------------------------------------------------- generation
    max_output_tokens: int = 900
    temperature: float = 0.0

    @property
    def has_llm(self) -> bool:
        if self.llm_provider == "openai":
            return bool(self.openai_api_key)
        if self.llm_provider == "anthropic":
            return bool(self.anthropic_api_key)
        if self.llm_provider == "ollama":
            return True  # local server; no credential to check
        return False

    @property
    def active_model(self) -> str:
        if self.generation_model:
            return self.generation_model
        return {
            "openai": "gpt-4.1-mini",
            "anthropic": "claude-haiku-4-5",
            "ollama": self.ollama_model,
        }.get(self.llm_provider, "")

    @property
    def active_context_model(self) -> str:
        """Cheap model for index-time chunk contextualisation."""
        if self.context_model:
            return self.context_model
        return {
            "openai": "gpt-4.1-nano",
            "anthropic": "claude-haiku-4-5",
            "ollama": self.ollama_model,
        }.get(self.llm_provider, "")

    @property
    def active_judge_model(self) -> str:
        """Model for LLM-judge eval metrics.

        Deliberately a different model than the generator: scoring an answer with the model
        that wrote it invites self-preference bias. A different family would be better still,
        which is noted as a limitation in the eval writeup when only one provider is available.
        """
        if self.judge_model:
            return self.judge_model
        return {
            "openai": "gpt-5-mini",
            "anthropic": "claude-sonnet-4-5",
            "ollama": self.ollama_model,
        }.get(self.llm_provider, "")


@lru_cache
def get_settings() -> Settings:
    return Settings()
