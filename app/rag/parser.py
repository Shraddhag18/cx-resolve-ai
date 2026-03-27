"""Multi-format document parser: supports .txt, .pdf, .docx."""
from pathlib import Path
from langchain_core.documents import Document


def _parse_txt(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()


def _parse_pdf(filepath: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(filepath)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def _parse_docx(filepath: str) -> str:
    from docx import Document as DocxDoc
    doc = DocxDoc(filepath)
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip()).strip()


_PARSERS = {".txt": _parse_txt, ".pdf": _parse_pdf, ".docx": _parse_docx}
SUPPORTED_EXTENSIONS = set(_PARSERS.keys())


def parse_document(filepath: str) -> Document:
    """Parse a file into a LangChain Document regardless of format."""
    path = Path(filepath)
    suffix = path.suffix.lower()
    if suffix not in _PARSERS:
        raise ValueError(f"Unsupported file type '{suffix}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}")
    content = _PARSERS[suffix](filepath)
    title = path.stem.replace("_", " ").replace("-", " ").title()
    return Document(
        page_content=content,
        metadata={"source": filepath, "title": title, "doc_id": path.stem},
    )
