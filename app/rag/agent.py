"""Agentic RAG pipeline: query decomposition, iterative retrieval, and rewrite-on-failure."""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from app.rag.retriever import retrieve, MIN_RELEVANCE
from app.config import get_settings
from app.models import AgentStep


def _llm():
    s = get_settings()
    return ChatOpenAI(model=s.openai_model, openai_api_key=s.openai_api_key, temperature=0)


def _decompose(question: str, llm) -> list[str]:
    """Break a complex question into focused sub-queries for retrieval."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a query planning assistant for a customer support knowledge base. "
            "Given a customer question, output 1 to 3 focused search queries that together cover the question. "
            "If the question is simple and self-contained, output just 1 query. "
            "Output ONLY the queries — one per line, no numbering, bullets, or explanation."
        )),
        ("human", "{question}"),
    ])
    result = (prompt | llm).invoke({"question": question})
    queries = [q.strip() for q in result.content.strip().splitlines() if q.strip()]
    return queries[:3]


def _rewrite(question: str, llm) -> str:
    """Rewrite a query to improve keyword overlap with support documents."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Rewrite this customer support question to maximize keyword overlap with a support knowledge base. "
            "Make it specific and keyword-rich. Output ONLY the rewritten question."
        )),
        ("human", "{question}"),
    ])
    result = (prompt | llm).invoke({"question": question})
    return result.content.strip()


def agentic_retrieve(
    vectorstore: FAISS,
    question: str,
    top_k: int = 5,
) -> tuple[list[Document], list[float], list[AgentStep]]:
    """
    Agentic retrieval pipeline:
    1. Decompose question into sub-queries
    2. Retrieve for each sub-query, deduplicating results
    3. If max relevance < threshold, rewrite and retry
    4. Merge, sort by relevance, return top_k with agent trace
    """
    llm = _llm()
    steps: list[AgentStep] = []
    all_docs: list[Document] = []
    all_scores: list[float] = []
    seen: set[str] = set()

    # Step 1: Query decomposition
    sub_queries = _decompose(question, llm)
    steps.append(AgentStep(
        step="Query Planning",
        detail=f"Split into {len(sub_queries)} sub-quer{'y' if len(sub_queries) == 1 else 'ies'}: "
               + " | ".join(f'"{q}"' for q in sub_queries),
    ))

    # Step 2: Retrieve for each sub-query
    for q in sub_queries:
        docs, scores = retrieve(vectorstore, q, top_k=top_k)
        added = 0
        for doc, score in zip(docs, scores):
            key = doc.page_content[:80]
            if key not in seen:
                seen.add(key)
                all_docs.append(doc)
                all_scores.append(score)
                added += 1
        steps.append(AgentStep(
            step="Retrieval",
            detail=f'Query "{q}" → {added} unique chunk{"s" if added != 1 else ""} retrieved',
        ))

    # Step 3: Check if any result clears the relevance threshold
    relevances = [max(0.0, 1.0 - s / 2.0) for s in all_scores]
    max_rel = max(relevances) if relevances else 0.0

    if max_rel < MIN_RELEVANCE:
        rewritten = _rewrite(question, llm)
        steps.append(AgentStep(
            step="Query Rewrite",
            detail=f"Best relevance was {round(max_rel * 100)}% — below threshold. "
                   f'Rewrote to: "{rewritten}"',
        ))
        docs, scores = retrieve(vectorstore, rewritten, top_k=top_k)
        added = 0
        for doc, score in zip(docs, scores):
            key = doc.page_content[:80]
            if key not in seen:
                seen.add(key)
                all_docs.append(doc)
                all_scores.append(score)
                added += 1
        steps.append(AgentStep(
            step="Retry Retrieval",
            detail=f"Retrieved {added} additional chunk{'s' if added != 1 else ''} after rewrite",
        ))

    # Step 4: Sort by relevance and cap at top_k
    paired = sorted(zip(all_scores, all_docs), key=lambda x: x[0])
    final_scores = [p[0] for p in paired[:top_k]]
    final_docs = [p[1] for p in paired[:top_k]]

    best = round(max(max(0.0, 1.0 - s / 2.0) for s in final_scores) * 100) if final_scores else 0
    steps.append(AgentStep(
        step="Context Assembly",
        detail=f"Assembled {len(final_docs)} chunk{'s' if len(final_docs) != 1 else ''}, "
               f"best relevance: {best}% — ready for answer generation",
    ))

    return final_docs, final_scores, steps
