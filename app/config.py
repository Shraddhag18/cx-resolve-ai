from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    faiss_index_path: str = "data/faiss_index"
    docs_dir: str = "data/sample_docs"
    retriever_top_k: int = 5
    max_tokens: int = 1024
    app_name: str = "CX Resolve AI"
    app_version: str = "1.0.0"
    app_api_key: str = "cx-resolve-2026-secret"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
