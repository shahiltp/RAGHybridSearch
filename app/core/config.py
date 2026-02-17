from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # LLM / Embeddings (OpenAI)
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    llm_model: str = Field("gpt-4o-mini", alias="LLM_MODEL")  # fast + cheap default
    embedding_model: str = Field("text-embedding-3-large", alias="EMBEDDING_MODEL")

    # Postgres / pgvector
    pg_dsn: str = Field(..., alias="PG_DSN")  # e.g. postgresql+psycopg://user:pass@localhost:5432/db

    # Retrieval parameters
    semantic_top_k: int = Field(30, alias="SEMANTIC_TOP_K")
    bm25_top_k: int = Field(30, alias="BM25_TOP_K")
    fused_top_n: int = Field(20, alias="FUSED_TOP_N")

    # RRF fusion parameters
    rrf_k: int = Field(60, alias="RRF_K")
    w_semantic: float = Field(1.0, alias="W_SEMANTIC")
    w_bm25: float = Field(1.2, alias="W_BM25")

    # Final context size
    final_top_k: int = Field(5, alias="FINAL_TOP_K")

    # Reranking / optional providers (Cohere)
    rerank_provider: Optional[str] = Field(None, alias="RERANK_PROVIDER")
    cohere_api_key: Optional[str] = Field(None, alias="COHERE_API_KEY")
    cohere_rerank_model: Optional[str] = Field(None, alias="COHERE_RERANK_MODEL")

    # Ingestion chunking (optional overrides)
    chunk_tokens: Optional[int] = Field(None, alias="CHUNK_TOKENS")
    chunk_overlap_tokens: Optional[int] = Field(None, alias="CHUNK_OVERLAP_TOKENS")


settings = Settings()
