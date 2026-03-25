from langchain_openai import OpenAIEmbeddings
from app.config import get_settings

_embeddings_instance = None


def get_embeddings() -> OpenAIEmbeddings:
    global _embeddings_instance
    if _embeddings_instance is None:
        settings = get_settings()
        _embeddings_instance = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            openai_api_key=settings.openai_api_key,
        )
    return _embeddings_instance
