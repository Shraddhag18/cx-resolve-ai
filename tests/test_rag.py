"""Unit tests for RAG components (no OpenAI calls required)."""
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from app.rag.retriever import format_cited_sources
from app.rag.chain import build_context


def make_doc(content: str, title: str = "Test Doc", doc_id: str = "test") -> Document:
    return Document(page_content=content, metadata={"title": title, "doc_id": doc_id})


def test_format_cited_sources_deduplicates():
    docs = [
        make_doc("Content A", title="Billing FAQ", doc_id="billing"),
        make_doc("Content B", title="Billing FAQ", doc_id="billing"),  # duplicate
        make_doc("Content C", title="Account Guide", doc_id="account"),
    ]
    scores = [0.1, 0.2, 0.3]
    sources = format_cited_sources(docs, scores)
    titles = [s.title for s in sources]
    assert len(sources) == 2
    assert titles.count("Billing FAQ") == 1


def test_format_cited_sources_relevance_range():
    docs = [make_doc("Some content")]
    scores = [0.0]
    sources = format_cited_sources(docs, scores)
    assert 0.0 <= sources[0].relevance_score <= 1.0


def test_format_cited_sources_excerpt_truncation():
    long_content = "x" * 500
    docs = [make_doc(long_content)]
    scores = [0.1]
    sources = format_cited_sources(docs, scores)
    assert sources[0].excerpt.endswith("...")
    assert len(sources[0].excerpt) <= 303 + 3  # 300 chars + "..."


def test_build_context_formats_correctly():
    docs = [
        make_doc("Answer to billing question.", title="Billing FAQ"),
        make_doc("Answer to account question.", title="Account Guide"),
    ]
    context = build_context(docs)
    assert "[Billing FAQ]" in context
    assert "[Account Guide]" in context
    assert "---" in context


def test_build_context_single_doc():
    docs = [make_doc("Single doc content.", title="Solo")]
    context = build_context(docs)
    assert "[Solo]" in context
    assert "---" not in context
