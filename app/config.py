import os
from functools import lru_cache
from dotenv import load_dotenv

# Load .env file locally if present; no-op on Railway where env vars are injected directly
load_dotenv()


class Settings:
    def __init__(self):
        self.openai_api_key: str = os.environ["OPENAI_API_KEY"]
        self.openai_model: str = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.openai_embedding_model: str = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.faiss_index_path: str = os.environ.get("FAISS_INDEX_PATH", "data/faiss_index")
        self.docs_dir: str = os.environ.get("DOCS_DIR", "data/sample_docs")
        self.retriever_top_k: int = int(os.environ.get("RETRIEVER_TOP_K", "5"))
        self.max_tokens: int = int(os.environ.get("MAX_TOKENS", "1024"))
        self.app_name: str = os.environ.get("APP_NAME", "CX Resolve AI")
        self.app_version: str = os.environ.get("APP_VERSION", "1.0.0")
        self.app_api_key: str = os.environ.get("APP_API_KEY", "cx-resolve-2026-secret")


@lru_cache()
def get_settings() -> Settings:
    return Settings()
