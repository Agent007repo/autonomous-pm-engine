"""
src/config/settings.py

Single source of truth for all runtime configuration.
Values are loaded from environment variables or a .env file.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    All configuration for the Autonomous PM Engine.
    Reads from environment variables; falls back to .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_model: str = Field("gpt-4o", description="OpenAI model name")
    openai_temperature: float = Field(0.2, ge=0.0, le=2.0)
    openai_max_tokens: int = Field(4096, ge=256, le=16384)

    # ── Neo4j ─────────────────────────────────────────────────────────────────
    neo4j_uri: str = Field("bolt://localhost:7687")
    neo4j_user: str = Field("neo4j")
    neo4j_password: str = Field(..., description="Neo4j password")

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    chroma_host: str = Field("localhost")
    chroma_port: int = Field(8001, ge=1, le=65535)
    chroma_collection: str = Field("pm_engine")
    chroma_mode: Literal["server", "local"] = Field("server")

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = Field("BAAI/bge-m3")
    embedding_device: Literal["cpu", "cuda", "mps"] = Field("cpu")

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = Field(512, ge=64, le=4096)
    chunk_overlap: int = Field(64, ge=0, le=512)
    semantic_split_threshold: float = Field(0.65, ge=0.0, le=1.0)

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k_retrieval: int = Field(10, ge=1, le=100)
    hybrid_alpha: float = Field(
        0.7,
        ge=0.0,
        le=1.0,
        description="Weight for dense vs sparse in hybrid retrieval. "
                    "1.0 = pure dense, 0.0 = pure sparse (BM25).",
    )

    # ── Agent behaviour ───────────────────────────────────────────────────────
    max_critique_rounds: int = Field(3, ge=1, le=10)
    engineering_gate_threshold: float = Field(7.0, ge=0.0, le=10.0)

    # ── Output ────────────────────────────────────────────────────────────────
    output_dir: str = Field("outputs/")
    prd_version: str = Field("1.0")

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field("INFO")
    log_file: str = Field("logs/pm_engine.log")

    # ── Validators ────────────────────────────────────────────────────────────
    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_size(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 512)
        if v >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({v}) must be less than chunk_size ({chunk_size})"
            )
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached singleton Settings instance.
    Call get_settings() anywhere in the codebase; no need to re-instantiate.
    """
    return Settings()
