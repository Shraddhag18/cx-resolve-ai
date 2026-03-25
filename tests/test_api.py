"""Integration tests for FastAPI endpoints (mocks OpenAI + FAISS)."""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from langchain.schema import Document


def make_mock_vectorstore():
    doc = Document(
        page_content="To reset your password, click Forgot Password on the login page.",
        metadata={"title": "Account Guide", "doc_id": "account", "chunk_id": 0},
    )
    vs = MagicMock()
    vs.similarity_search_with_score.return_value = [(doc, 0.15)]
    return vs


@pytest.fixture
def client():
    with patch("app.routers.chat.get_vectorstore", return_value=make_mock_vectorstore()), \
         patch("app.rag.chain.ChatOpenAI") as mock_llm_class:
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "To reset your password, visit the login page [Account Guide]."
        mock_llm.return_value = mock_response
        mock_llm_class.return_value.__or__ = lambda self, other: MagicMock(
            invoke=lambda x: mock_response
        )

        from app.main import create_app
        app = create_app()
        with TestClient(app) as c:
            yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_dashboard_empty(client):
    resp = client.get("/api/v1/dashboard")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_queries" in data
    assert "resolution_rate_percent" in data
