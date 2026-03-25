from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from app.config import get_settings

SYSTEM_PROMPT = """You are CX Resolve, a Tier-1 customer support AI assistant.
Answer the customer's question using ONLY the provided support documents.
Always cite which document(s) you used by referencing their title in brackets, e.g. [Billing FAQ].
If the documents do not contain enough information to fully answer, say so clearly and suggest escalation.
Be concise, empathetic, and professional."""

HUMAN_PROMPT = """Support Documents:
{context}

Customer Question: {question}

Provide a clear, cited answer:"""


def build_context(docs: list[Document]) -> str:
    parts = []
    for doc in docs:
        title = doc.metadata.get("title", "Document")
        parts.append(f"[{title}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def answer_with_citations(question: str, docs: list[Document]) -> tuple[str, float]:
    """
    Run the LangChain RAG chain to produce a cited answer.
    Returns (answer_text, confidence_score).
    """
    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.openai_model,
        openai_api_key=settings.openai_api_key,
        max_tokens=settings.max_tokens,
        temperature=0.2,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ])

    context = build_context(docs)
    chain = prompt | llm

    response = chain.invoke({"context": context, "question": question})
    answer = response.content.strip()

    # Heuristic confidence: based on number of citation markers in response
    citation_count = answer.count("[")
    confidence = min(0.95, 0.5 + citation_count * 0.15)

    return answer, round(confidence, 4)
