from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from app.models import CitedSource


def retrieve(vectorstore: FAISS, query: str, top_k: int = 5) -> tuple[list[Document], list[float]]:
    """Retrieve top_k documents most relevant to query, with similarity scores."""
    results = vectorstore.similarity_search_with_score(query, k=top_k)
    docs = [r[0] for r in results]
    scores = [float(r[1]) for r in results]
    return docs, scores


MIN_RELEVANCE = 0.45


def format_cited_sources(docs: list[Document], scores: list[float]) -> list[CitedSource]:
    """Convert retrieved documents into CitedSource objects, filtering low-relevance results."""
    sources = []
    seen_titles = set()
    for doc, score in zip(docs, scores):
        title = doc.metadata.get("title", "Unknown")
        doc_id = doc.metadata.get("doc_id", "unknown")

        # FAISS L2 distance → convert to 0–1 similarity
        relevance = max(0.0, 1.0 - score / 2.0)

        # Skip low-relevance and duplicate sources
        if relevance < MIN_RELEVANCE or title in seen_titles:
            continue
        seen_titles.add(title)

        # Strip document header line from excerpt (first non-empty line if it's a title)
        content = doc.page_content.strip()
        lines = content.splitlines()
        if lines and lines[0].strip() and not lines[0].strip().endswith("?"):
            content = "\n".join(lines[1:]).strip()

        excerpt = content[:250].strip()
        if len(content) > 250:
            excerpt += "..."

        sources.append(
            CitedSource(
                doc_id=doc_id,
                title=title,
                excerpt=excerpt,
                relevance_score=round(relevance, 4),
            )
        )
    return sources
