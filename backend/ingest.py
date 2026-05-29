"""
PDF ingestion script for the multilingual RAG pipeline.

Usage:
    python backend\ingest.py path\to\document.pdf
    python backend\ingest.py path\to\document.pdf --collection my_collection
"""

import argparse
import re
import sys
from pathlib import Path

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb


def ingest_pdf(
    pdf_path: str,
    collection_name: str = "documents",
    vector_store_path: str = "vector_store",
    chunk_size: int = 250,
    chunk_overlap: int = 50,
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
) -> int:
    """
    Read a PDF, chunk it, embed it, and store it in a persistent ChromaDB collection.

    Returns the number of chunks stored.
    """
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # --- Step 1: Read and clean the PDF text ---
    print(f"Reading {pdf_file.name} ...")
    reader = PdfReader(str(pdf_file))
    full_text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        # Keep ASCII plus Hindi (Devanagari) characters; drop control chars and box-drawing junk
        page_text = re.sub(r"[^\x00-\x7F\u0900-\u097F]+", " ", page_text)
        page_text = re.sub(r"\s+", " ", page_text)
        full_text += page_text + "\n"

    if not full_text.strip():
        raise ValueError(f"No text extracted from {pdf_path}. PDF may be scanned or empty.")

    # --- Step 2: Chunk ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_text(full_text)
    print(f"Chunks created: {len(chunks)}")

    # --- Step 3: Embed ---
    print("Loading multilingual embedding model ...")
    model = SentenceTransformer(model_name)
    print("Embedding chunks ...")
    embeddings = model.encode(chunks, show_progress_bar=True)

    # --- Step 4: Persist to ChromaDB ---
    client = chromadb.PersistentClient(path=vector_store_path)
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        print(f"Resetting existing collection: {collection_name}")
        client.delete_collection(collection_name)
    collection = client.create_collection(collection_name)

    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        embeddings=embeddings.tolist(),
        documents=chunks,
        metadatas=[{"source": pdf_file.name, "chunk_index": i} for i in range(len(chunks))],
    )

    count = collection.count()
    print(f"Stored {count} chunks in collection '{collection_name}' at '{vector_store_path}/'.")
    return count


def main():
    parser = argparse.ArgumentParser(description="Ingest a PDF into the RAG vector store.")
    parser.add_argument("pdf_path", help="Path to the PDF file to ingest")
    parser.add_argument(
        "--collection",
        default="documents",
        help="ChromaDB collection name (default: documents)",
    )
    parser.add_argument(
        "--vector-store",
        default="vector_store",
        help="Path to the ChromaDB persistent store (default: vector_store)",
    )
    args = parser.parse_args()

    try:
        ingest_pdf(
            pdf_path=args.pdf_path,
            collection_name=args.collection,
            vector_store_path=args.vector_store,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
