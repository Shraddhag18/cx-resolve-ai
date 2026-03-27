# CX Resolve AI

A production-grade **Tier-1 customer support chatbot** built on an **Agentic RAG pipeline** — powered by OpenAI, LangChain, and FAISS vector search.

Upload any support document (PDF, DOCX, or TXT), ask questions in natural language, and get cited answers in seconds. Every response includes a full agent trace showing exactly how the answer was found.

---

## What It Does

A customer types a support question → the agentic pipeline decomposes it into sub-queries → searches your uploaded documents semantically → GPT-4o-mini generates a cited answer → returns whether the issue is resolved or needs human escalation.

**Key capabilities:**
- Upload PDF, DOCX, or TXT files through the UI — indexed into FAISS automatically
- Agentic query planning: breaks complex questions into focused sub-queries
- Iterative retrieval with automatic query rewriting if initial results are poor
- Every answer cites the source document(s) used
- Real-time agent trace shows each step the pipeline took
- Weekly resolution dashboard tracking tickets, latency, and confidence

---

## Architecture

```
User Question
      │
      ▼
  Query Planning (LLM)
  Decompose into 1–3 sub-queries
      │
      ▼
  FAISS Retrieval (per sub-query)
  OpenAI text-embedding-3-small
      │
      ▼
  Relevance Check
  If max relevance < 45% → rewrite query → retry
      │
      ▼
  Context Assembly
  Merge, deduplicate, sort by relevance
      │
      ▼
  Answer Generation (GPT-4o-mini)
  Cited answer + confidence score
      │
      ▼
  Response: answer + sources + agent_steps + ticket_id
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI (async) |
| RAG Orchestration | LangChain |
| Vector Store | FAISS (faiss-cpu) |
| LLM + Embeddings | OpenAI GPT-4o-mini + text-embedding-3-small |
| Validation | Pydantic v2 |
| Document Parsing | pypdf, python-docx |
| Frontend | Vanilla HTML/CSS/JS (served by FastAPI) |
| Tests | pytest + pytest-asyncio |

---

## Project Structure

```
cx-resolve-ai/
├── app/
│   ├── auth.py              # API key authentication
│   ├── config.py            # Pydantic settings (reads from .env)
│   ├── main.py              # FastAPI app, UI serving
│   ├── models.py            # Request/response schemas
│   ├── rag/
│   │   ├── agent.py         # Agentic pipeline (decompose → retrieve → rewrite)
│   │   ├── chain.py         # LLM chain with citation prompt
│   │   ├── embedder.py      # OpenAI embeddings singleton
│   │   ├── indexer.py       # FAISS build/load
│   │   ├── parser.py        # PDF, DOCX, TXT parser
│   │   └── retriever.py     # Vector search + source formatting
│   └── routers/
│       ├── chat.py          # POST /query endpoint
│       ├── dashboard.py     # GET /dashboard endpoint
│       └── upload.py        # POST /upload, GET/DELETE /documents
├── data/
│   └── sample_docs/         # Drop your documents here
├── static/
│   └── index.html           # Chat UI + Documents tab + Dashboard
├── tests/
│   ├── test_api.py
│   └── test_rag.py
├── scripts/
│   └── ingest.py            # Standalone re-index script
├── .env.example
└── requirements.txt
```

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- An [OpenAI API key](https://platform.openai.com/api-keys)

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/Shraddhag18/cx-resolve-ai.git
cd cx-resolve-ai
```

---

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

---

### Step 3 — Set up environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your values:

```env
OPENAI_API_KEY=sk-...          # Required — your OpenAI API key
APP_API_KEY=your-secret-key    # Required — protects all /api/v1/* endpoints

# Optional — change these if needed
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
FAISS_INDEX_PATH=data/faiss_index
DOCS_DIR=data/sample_docs
RETRIEVER_TOP_K=5
MAX_TOKENS=1024
```

> The `APP_API_KEY` is used internally by the UI — you set it once and the app handles it automatically. You never have to enter it manually.

---

### Step 4 — Start the server

```bash
python -m app.main
```

The app will:
1. Load (or build) the FAISS index from `data/sample_docs/`
2. Start the FastAPI server on `http://localhost:8000`

---

### Step 5 — Open the UI

Visit **http://localhost:8000** in your browser.

Three tabs are available:

| Tab | What it does |
|-----|-------------|
| **Chat** | Ask support questions, see cited answers + agent trace |
| **Documents** | Upload PDF/DOCX/TXT, view and delete indexed files |
| **Dashboard** | Weekly resolution stats, tickets, and top topics |

---

## Uploading Documents

### Via the UI (recommended)

1. Go to the **Documents** tab
2. Drag and drop a file, or click to browse
3. Supported formats: `.pdf`, `.docx`, `.txt` (max 10 MB)
4. The document is parsed, chunked, embedded, and indexed automatically

### Via the API

```bash
curl -X POST http://localhost:8000/api/v1/upload \
  -H "X-API-Key: your-secret-key" \
  -F "file=@your-document.pdf"
```

### Via the ingest script (bulk)

Drop multiple files into `data/sample_docs/` then run:

```bash
python scripts/ingest.py
```

---

## API Reference

All `/api/v1/*` endpoints require the header:
```
X-API-Key: your-secret-key
```

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/query` | Submit a support question |
| `POST` | `/api/v1/upload` | Upload a document and re-index |
| `GET` | `/api/v1/documents` | List all indexed documents |
| `DELETE` | `/api/v1/documents/{filename}` | Delete a document and re-index |
| `POST` | `/api/v1/ingest` | Rebuild FAISS index from disk |
| `GET` | `/api/v1/dashboard` | Weekly resolution stats |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Interactive Swagger UI |

### Example — Submit a query

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{"question": "How do I reset my password?"}'
```

**Response:**

```json
{
  "ticket_id": "a3f9c21b",
  "question": "How do I reset my password?",
  "answer": "To reset your password, click 'Forgot Password' on the login page and enter your registered email. You will receive a reset link within 5 minutes [Account Management Guide].",
  "sources": [
    {
      "doc_id": "account",
      "title": "Account Management Guide",
      "excerpt": "How do I reset my password? To reset your password, click Forgot Password...",
      "relevance_score": 0.6001
    }
  ],
  "resolved": true,
  "confidence": 0.65,
  "latency_ms": 1823.4,
  "agent_steps": [
    { "step": "Query Planning", "detail": "Split into 1 sub-query: \"reset password instructions\"" },
    { "step": "Retrieval", "detail": "Query \"reset password instructions\" → 5 unique chunks retrieved" },
    { "step": "Context Assembly", "detail": "Assembled 5 chunks, best relevance: 60%" }
  ]
}
```

---

## Running Tests

```bash
pytest tests/ -v
```

Expected output: **8 tests passing**, 0 warnings.

---

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(required)* | Your OpenAI API key |
| `APP_API_KEY` | *(required)* | Secret key to protect API endpoints |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI chat model |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `FAISS_INDEX_PATH` | `data/faiss_index` | Path to save/load FAISS index |
| `DOCS_DIR` | `data/sample_docs` | Directory scanned for documents |
| `RETRIEVER_TOP_K` | `5` | Number of chunks retrieved per query |
| `MAX_TOKENS` | `1024` | Max tokens in LLM response |

---

## How the Agentic Pipeline Works

Unlike a basic RAG system that runs one search query, this pipeline acts like an agent:

1. **Query Planning** — The LLM analyzes your question and breaks it into 1–3 focused sub-queries (e.g. "How do I reset my password and enable 2FA?" → `["reset password", "enable two-factor authentication"]`)

2. **Retrieval** — FAISS searches the index for each sub-query independently, deduplicating results across queries

3. **Relevance Check** — If the best match scores below 45% relevance, the query is automatically rewritten to be more keyword-rich and retrieval is retried

4. **Context Assembly** — All results are merged, sorted by relevance score, and capped at `top_k` chunks

5. **Answer Generation** — GPT-4o-mini generates a cited answer using only the retrieved context

Every step is returned in the `agent_steps` field so you can see exactly how the answer was derived.

---

## Sample Questions to Try

Once running, try these in the Chat tab (using the included sample documents):

- `How do I reset my password?`
- `Why was I charged twice?`
- `How do I enable two-factor authentication?`
- `The app is not loading, what should I do?`
- `How do I cancel my subscription?`
- `How do I delete my account?`
