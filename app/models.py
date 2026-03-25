from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, description="Customer support question")
    top_k: Optional[int] = Field(default=5, ge=1, le=20, description="Number of documents to retrieve")


class CitedSource(BaseModel):
    doc_id: str
    title: str
    excerpt: str
    relevance_score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[CitedSource]
    resolved: bool
    confidence: float
    latency_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class IngestRequest(BaseModel):
    directory: Optional[str] = None


class IngestResponse(BaseModel):
    documents_indexed: int
    index_path: str
    message: str


class TicketResolution(BaseModel):
    ticket_id: str
    question: str
    resolved: bool
    timestamp: datetime


class DashboardStats(BaseModel):
    week_start: str
    week_end: str
    total_queries: int
    resolved_queries: int
    resolution_rate_percent: float
    avg_latency_ms: float
    avg_confidence: float
    top_topics: List[str]
    recent_tickets: List[TicketResolution]
