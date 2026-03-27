"""Document upload, listing, and deletion endpoints."""
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Security
from app.auth import verify_api_key
from app.config import get_settings
from app.rag.parser import SUPPORTED_EXTENSIONS
import app.routers.chat as chat_router

router = APIRouter(prefix="/api/v1", tags=["documents"], dependencies=[Security(verify_api_key)])

MAX_SIZE_BYTES = 10 * 1024 * 1024 * 1024  # 10 GB


@router.post("/upload", summary="Upload a support document and re-index")
async def upload_document(file: UploadFile = File(...)):
    settings = get_settings()
    docs_dir = Path(settings.docs_dir)

    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{suffix}' not supported. Accepted: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    content = await file.read()
    if len(content) > MAX_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 10 GB limit.")

    dest = docs_dir / file.filename
    dest.write_bytes(content)

    try:
        from app.rag.indexer import build_index
        chat_router._vectorstore = build_index(str(docs_dir), settings.faiss_index_path)
        count = chat_router._vectorstore.index.ntotal
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")

    return {
        "filename": file.filename,
        "size_kb": round(len(content) / 1024, 1),
        "vectors_indexed": count,
        "message": "Document uploaded and indexed successfully.",
    }


@router.get("/documents", summary="List all indexed documents")
async def list_documents():
    settings = get_settings()
    docs_dir = Path(settings.docs_dir)
    docs = [
        {
            "filename": f.name,
            "size_kb": round(f.stat().st_size / 1024, 1),
            "type": f.suffix.lower().lstrip("."),
        }
        for f in sorted(docs_dir.iterdir())
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return {"documents": docs, "total": len(docs)}


@router.delete("/documents/{filename}", summary="Delete a document and re-index")
async def delete_document(filename: str):
    settings = get_settings()
    docs_dir = Path(settings.docs_dir)
    target = docs_dir / filename

    if not target.exists():
        raise HTTPException(status_code=404, detail="Document not found.")
    if target.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Invalid file type.")

    target.unlink()

    try:
        from app.rag.indexer import build_index
        chat_router._vectorstore = build_index(str(docs_dir), settings.faiss_index_path)
        count = chat_router._vectorstore.index.ntotal
    except Exception:
        chat_router._vectorstore = None
        count = 0

    return {"message": f"'{filename}' deleted and index rebuilt.", "vectors_remaining": count}
