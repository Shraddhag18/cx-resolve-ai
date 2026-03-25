from fastapi import APIRouter, Security
from app.auth import verify_api_key
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from app.models import DashboardStats, TicketResolution
import uuid

router = APIRouter(prefix="/api/v1", tags=["dashboard"], dependencies=[Security(verify_api_key)])

# In-memory store for demo; replace with PostgreSQL in production
_query_log: list[dict] = []


def record_query(question: str, resolved: bool, latency_ms: float, confidence: float):
    _query_log.append({
        "ticket_id": str(uuid.uuid4())[:8],
        "question": question,
        "resolved": resolved,
        "latency_ms": latency_ms,
        "confidence": confidence,
        "timestamp": datetime.now(timezone.utc),
    })


@router.get("/dashboard", response_model=DashboardStats, summary="Weekly resolution dashboard")
async def dashboard():
    """
    Returns weekly resolution stats: ticket volume, resolution rate,
    avg latency, confidence scores, and recent tickets.
    Mirrors the weekly dashboards used to track 40% manual ticket reduction.
    """
    now = datetime.now(timezone.utc)
    week_start = now - timedelta(days=now.weekday())
    week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)

    weekly = [e for e in _query_log if e["timestamp"] >= week_start]

    total = len(weekly)
    resolved = sum(1 for e in weekly if e["resolved"])
    resolution_rate = round((resolved / total * 100), 1) if total else 0.0
    avg_latency = round(sum(e["latency_ms"] for e in weekly) / total, 2) if total else 0.0
    avg_confidence = round(sum(e["confidence"] for e in weekly) / total, 4) if total else 0.0

    # Naive topic extraction: first 3 words of each question
    topic_counts: dict[str, int] = defaultdict(int)
    for e in weekly:
        words = e["question"].split()
        topic = " ".join(words[:3]).lower().rstrip("?")
        topic_counts[topic] += 1
    top_topics = sorted(topic_counts, key=topic_counts.get, reverse=True)[:5]

    recent = [
        TicketResolution(
            ticket_id=e["ticket_id"],
            question=e["question"],
            resolved=e["resolved"],
            timestamp=e["timestamp"],
        )
        for e in sorted(weekly, key=lambda x: x["timestamp"], reverse=True)[:10]
    ]

    return DashboardStats(
        week_start=week_start.strftime("%Y-%m-%d"),
        week_end=(week_start + timedelta(days=6)).strftime("%Y-%m-%d"),
        total_queries=total,
        resolved_queries=resolved,
        resolution_rate_percent=resolution_rate,
        avg_latency_ms=avg_latency,
        avg_confidence=avg_confidence,
        top_topics=top_topics,
        recent_tickets=recent,
    )
