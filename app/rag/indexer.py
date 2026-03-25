import os
import glob
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from app.rag.embedder import get_embeddings
from app.config import get_settings


def load_documents(docs_dir: str) -> list[Document]:
    """Load all .txt files from a directory as LangChain Documents."""
    documents = []
    pattern = os.path.join(docs_dir, "**", "*.txt")
    for filepath in glob.glob(pattern, recursive=True):
        title = Path(filepath).stem.replace("_", " ").title()
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if content:
            documents.append(
                Document(
                    page_content=content,
                    metadata={"source": filepath, "title": title, "doc_id": Path(filepath).stem},
                )
            )
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


def build_index(docs_dir: str, index_path: str) -> FAISS:
    """Load documents, chunk them, embed, and save FAISS index."""
    documents = load_documents(docs_dir)
    if not documents:
        raise ValueError(f"No .txt documents found in {docs_dir}")

    chunks = chunk_documents(documents)
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(index_path, exist_ok=True)
    vectorstore.save_local(index_path)
    return vectorstore


def load_index(index_path: str) -> FAISS:
    """Load a saved FAISS index from disk."""
    embeddings = get_embeddings()
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)


def get_or_build_index() -> FAISS:
    """Load existing index or build a new one from sample docs."""
    settings = get_settings()
    index_path = settings.faiss_index_path

    if os.path.exists(os.path.join(index_path, "index.faiss")):
        return load_index(index_path)

    return build_index(settings.docs_dir, index_path)
