# CX Resolve AI

A production-grade **Tier-1 customer support chatbot** powered by **OpenAI**, **LangChain**, and **FAISS** vector search. Answers customer questions in under 1 second with cited sources from a 10K+ document knowledge base.

## Features

- **RAG Pipeline** — LangChain retrieval-augmented generation with inline citations on every response
- **FAISS Semantic Search** — Sub-second vector retrieval across large document corpora
- **Citation Traceability** — Every answer references source documents by title, enabling full traceability
- **FastAPI Backend** — Async REST API with OpenAPI docs at `/docs`
- **Weekly Dashboard** — `/api/v1/dashboard` tracks resolution rate, latency, and top topics
- **Easy Ingestion** — Add `.txt` files to `data/sample_docs/` and run the ingest script

## Architecture

```
Customer Query
      │
      ▼
FastAPI /api/v1/query
      │
      ├──► FAISS Retriever (top-k semantic search)
      │         └── OpenAI text-embedding-3-small
      │
      └──► LangChain RAG Chain
                ├── Retrieved context (with titles)
                ├── ChatOpenAI (gpt-4o-mini)
                └── Cited answer + confidence score
```

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY
```

### 3. Build the FAISS index
```bash
python scripts/ingest.py
```

### 4. Start the server
```bash
python -m app.main
```

API docs available at: `http://localhost:8000/docs`

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/query` | Submit a support question |
| `POST` | `/api/v1/ingest` | Rebuild the FAISS index |
| `GET` | `/api/v1/dashboard` | Weekly resolution stats |
| `GET` | `/health` | Health check |

### Example Query

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I reset my password?"}'
```

```json
{
  "question": "How do I reset my password?",
  "answer": "To reset your password, click 'Forgot Password' on the login page [Account Management Guide]. A reset link will be sent to your registered email and expires after 1 hour.",
  "sources": [
    {
      "doc_id": "account",
      "title": "Account Management Guide",
      "excerpt": "To reset your password, click 'Forgot Password' on the login page...",
      "relevance_score": 0.9312
    }
  ],
  "resolved": true,
  "confidence": 0.8,
  "latency_ms": 423.5
}
```

## Adding Documents

Place `.txt` files in `data/sample_docs/` (any subdirectory), then run:

```bash
python scripts/ingest.py
```

Or call the ingest endpoint:
```bash
curl -X POST http://localhost:8000/api/v1/ingest
```

## Running Tests

```bash
pytest tests/ -v
```

## Tech Stack

- **Python 3.11+**
- **FastAPI** — async REST framework
- **LangChain** — RAG orchestration and citation pipelines
- **OpenAI API** — embeddings (`text-embedding-3-small`) + LLM (`gpt-4o-mini`)
- **FAISS** — high-performance vector similarity search
- **Pydantic v2** — request/response validation
