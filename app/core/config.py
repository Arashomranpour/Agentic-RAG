from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration comes from environment variables (or a .env file)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Agentic RAG API"
    environment: str = "development"
    log_level: str = "INFO"

    # Storage
    database_url: str = "sqlite:///./data/app.db"
    checkpoint_path: str = "./data/checkpoints.db"

    # Security
    admin_token: str = ""  # empty = admin endpoints disabled
    rate_limit_per_minute: int = 60
    max_upload_mb: int = 10

    # LLM
    llm_backend: Literal["groq", "fake"] = "groq"
    groq_api_key: str | None = None
    groq_model: str = "openai/gpt-oss-120b"
    serper_api_key: str | None = None  # enables the web-search tool

    # Retrieval
    embedding_backend: Literal["fastembed", "fake"] = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k: int = 4

    # Agent
    max_rewrites: int = 1
    history_messages: int = 10


@lru_cache
def get_settings() -> Settings:
    return Settings()
