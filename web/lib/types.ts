/** Mirrors the FastAPI response models in `api/app/schemas/models.py`. */

export interface Document {
  id: number;
  doc_key: string;
  title: string;
  filename: string;
  source: "seed" | "upload";
  n_pages: number | null;
  n_chunks: number;
  status: "pending" | "parsing" | "embedding" | "ready" | "failed";
  error: string | null;
  created_at: string;
  ingested_at: string | null;
}

export interface Citation {
  marker: number;
  chunk_id: number;
  document_id: number;
  doc_key: string;
  title: string;
  page_start: number;
  page_end: number;
  section: string | null;
  snippet: string;
}

/** Per-candidate provenance: how each retrieval stage ranked a chunk. */
export interface TraceRow {
  chunk_id: number;
  doc_key: string;
  title: string;
  page_start: number;
  section: string | null;
  snippet: string;
  vector_rank: number | null;
  vector_score: number | null;
  lexical_rank: number | null;
  lexical_score: number | null;
  rrf_score: number;
  rrf_rank: number;
  rerank_score: number | null;
  final_rank: number | null;
  used_in_context: boolean;
}

export interface RetrievalConfig {
  use_hybrid: boolean;
  use_rerank: boolean;
  rerank_depth: number | null;
  candidates_per_arm: number;
  rrf_k: number;
  hnsw_ef_search: number;
  top_k: number;
  contextualized_index: boolean;
  carried_forward: number;
}

export interface StreamMeta {
  conversation_id: number;
  config: RetrievalConfig;
  timings_ms: Record<string, number>;
  n_candidates: number;
  prompt_tokens: number;
  generation_enabled: boolean;
  model: string;
}

export interface StreamDone {
  message_id: number;
  conversation_id: number;
  abstained: boolean;
  retrieval_only: boolean;
  model: string;
  latency_ms: number;
  prompt_tokens: number;
  cited_chunk_ids: number[];
}

export interface Health {
  status: string;
  database: boolean;
  redis: boolean;
  documents: number;
  chunks: number;
  llm_provider: string;
  generation_enabled: boolean;
  model: string;
  retrieval_config: Record<string, unknown>;
}

export interface Turn {
  id: string;
  question: string;
  answer: string;
  citations: Citation[];
  meta?: StreamMeta;
  done?: StreamDone;
  streaming: boolean;
  error?: string;
}
