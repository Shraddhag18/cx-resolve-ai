"""
Standalone ingestion script.
Run: python scripts/ingest.py [--docs-dir path] [--index-path path]
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from app.rag.indexer import build_index
from app.config import get_settings


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from documents")
    parser.add_argument("--docs-dir", default=None)
    parser.add_argument("--index-path", default=None)
    args = parser.parse_args()

    settings = get_settings()
    docs_dir = args.docs_dir or settings.docs_dir
    index_path = args.index_path or settings.faiss_index_path

    print(f"Loading documents from: {docs_dir}")
    vectorstore = build_index(docs_dir, index_path)
    total = vectorstore.index.ntotal
    print(f"Index built successfully: {total} vectors saved to {index_path}")


if __name__ == "__main__":
    main()
