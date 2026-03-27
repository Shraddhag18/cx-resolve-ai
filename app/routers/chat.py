import time
from fastapi import APIRouter, HTTPException, Security
from app.models import QueryRequest, QueryResponse
from app.rag.indexer import get_or_build_index
from app.rag.agent import agentic_retrieve
from app.rag.retriever import format_cited_sources
from app.rag.chain import answer_with_citations
from app.routers.dashboard import record_query
from app.auth import verify_api_key
from langchain_community.vectorstores import FAISS

router = APIRouter(prefix="/api/v1", tags=["chat"], dependencies=[Security(verify_api_key)])

_vectorstore: FAISS | None = None


def get_vectorstore() -> FAISS:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = get_or_build_index()
    return _vectorstore


@router.post("/query", response_model=QueryResponse, summary="Submit a support query")
async def query_endpoint(request: QueryRequest):
    """
    Submit a customer support question.
    Runs an agentic RAG pipeline: query decomposition → iterative retrieval →
    rewrite-on-failure → cited answer generation.
    """
    start = time.perf_counter()

    try:
        vs = get_vectorstore()
        docs, scores, agent_steps = agentic_retrieve(vs, request.question, top_k=request.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {str(e)}")

    if not docs:
        raise HTTPException(status_code=404, detail="No relevant documents found.")

    try:
        answer, confidence = answer_with_citations(request.question, docs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    sources = format_cited_sources(docs, scores)
    resolved = confidence >= 0.65

    ticket_id = record_query(question=request.question, resolved=resolved, latency_ms=latency_ms, confidence=confidence)

    return QueryResponse(
        ticket_id=ticket_id,
        question=request.question,
        answer=answer,
        sources=sources,
        resolved=resolved,
        confidence=confidence,
        latency_ms=latency_ms,
        agent_steps=agent_steps,
    )


@router.post("/ingest", summary="Re-index documents from disk")
async def ingest_endpoint():
    """Trigger a re-index of all documents in the docs directory."""
    global _vectorstore
    from app.config import get_settings
    from app.rag.indexer import build_index

    settings = get_settings()
    try:
        _vectorstore = build_index(settings.docs_dir, settings.faiss_index_path)
        count = _vectorstore.index.ntotal
        return {"documents_indexed": count, "index_path": settings.faiss_index_path, "message": "Index rebuilt successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
