from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    faiss_index_path: str = "data/faiss_index"
    docs_dir: str = "data/sample_docs"
    retriever_top_k: int = 5
    max_tokens: int = 1024
    app_name: str = "CX Resolve AI"
    app_version: str = "1.0.0"

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
