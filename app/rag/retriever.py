from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from app.models import CitedSource


def retrieve(vectorstore: FAISS, query: str, top_k: int = 5) -> tuple[list[Document], list[float]]:
    """Retrieve top_k documents most relevant to query, with similarity scores."""
    results = vectorstore.similarity_search_with_score(query, k=top_k)
    docs = [r[0] for r in results]
    scores = [float(r[1]) for r in results]
    return docs, scores


def format_cited_sources(docs: list[Document], scores: list[float]) -> list[CitedSource]:
    """Convert retrieved documents into CitedSource objects."""
    sources = []
    seen_titles = set()
    for doc, score in zip(docs, scores):
        title = doc.metadata.get("title", "Unknown")
        doc_id = doc.metadata.get("doc_id", "unknown")

        # Deduplicate by title while keeping best score
        if title in seen_titles:
            continue
        seen_titles.add(title)

        excerpt = doc.page_content[:300].strip()
        if len(doc.page_content) > 300:
            excerpt += "..."

        # FAISS L2 distance → convert to 0–1 similarity
        relevance = max(0.0, 1.0 - score / 2.0)

        sources.append(
            CitedSource(
                doc_id=doc_id,
                title=title,
                excerpt=excerpt,
                relevance_score=round(relevance, 4),
            )
        )
    return sources
