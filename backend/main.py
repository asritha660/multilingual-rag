"""
FastAPI backend for the Multilingual RAG Assistant.

Exposes two endpoints:
  POST /upload  -> accepts a PDF, processes it into the vector store
  POST /ask     -> accepts a question, returns a grounded answer + sources

Run locally with:
  uvicorn backend.main:app --reload --port 8000
"""

import os
import re
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from rank_bm25 import BM25Okapi
from google import genai
from langdetect import detect, DetectorFactory
import time
from backend import database

DetectorFactory.seed = 0  # deterministic language detection

# ----------------------------------------------------------------------------
# One-time setup
# ----------------------------------------------------------------------------
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# Load models once when the server starts (not per request)
print("Loading models (first run downloads them)...")
embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("Models loaded.")

VECTOR_STORE_PATH = "vector_store"

# ----------------------------------------------------------------------------
# FastAPI app
# ----------------------------------------------------------------------------
app = FastAPI(title="Multilingual RAG API", version="1.0.0")

# Allow the Streamlit frontend (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # for local dev; tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------------
# RAG helper functions (moved from the Streamlit app)
# ----------------------------------------------------------------------------
def extract_clean_text(file_obj):
    reader = PdfReader(file_obj)
    full_text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        page_text = re.sub(r"[^\x00-\x7F\u0900-\u097F]+", " ", page_text)
        page_text = re.sub(r"\s+", " ", page_text)
        full_text += page_text + "\n"
    return full_text


def process_pdf(file_obj):
    text = extract_clean_text(file_obj)
    splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
    chunks = splitter.split_text(text)
    embeddings = embed_model.encode(chunks)

    db = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
    if "documents" in [c.name for c in db.list_collections()]:
        db.delete_collection("documents")
    collection = db.create_collection("documents")
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        embeddings=embeddings.tolist(),
        documents=chunks,
    )
    return len(chunks)


def get_collection():
    db = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
    return db.get_collection("documents")


def retrieve(question, language="en"):
    collection = get_collection()
    q_emb = embed_model.encode(question).tolist()
    all_chunks = collection.get()["documents"]

    vector_chunks = collection.query(
        query_embeddings=[q_emb], n_results=5
    )["documents"][0]

    tokenized = [c.lower().split() for c in all_chunks]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(question.lower().split())
    top_bm25 = sorted(zip(scores, all_chunks), key=lambda x: x[0], reverse=True)[:5]
    keyword_chunks = [c for s, c in top_bm25]

    merged = []
    for ch in vector_chunks + keyword_chunks:
        if ch not in merged:
            merged.append(ch)

    if language == "en":
        pairs = [[question, chunk] for chunk in merged]
        rerank_scores = reranker.predict(pairs)
        reranked = sorted(zip(rerank_scores, merged), key=lambda x: x[0], reverse=True)
        return [chunk for score, chunk in reranked[:4]]
    else:
        return merged[:6]


def generate_answer(question, chunks):
    context = "\n\n".join(chunks)
    prompt = f"""You are a helpful assistant. Answer the user's question using ONLY the context below.
If the answer is not in the context, say you don't know. Answer in the same language as the question.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text


# ----------------------------------------------------------------------------
# Request/response models for /ask
# ----------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    detected_language: str
    sources: list[str]


# ----------------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------------
@app.get("/")
def health_check():
    """Simple endpoint to confirm the API is running."""
    return {"status": "ok", "message": "Multilingual RAG API is running"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Accept a PDF file and process it into the vector store."""
    n_chunks = process_pdf(file.file)
    # Log this document to PostgreSQL
    database.log_document(
        file_name=file.filename,
        language="auto",
        chunk_count=n_chunks,
        metadata={"source": "upload"},
    )
    return {"filename": file.filename, "chunks_stored": n_chunks}


@app.post("/ask", response_model=AskResponse)
def ask_question(req: AskRequest):
    """Accept a question and return a grounded answer with sources."""
    start = time.time()
    detected_lang = detect(req.question)
    chunks = retrieve(req.question, language=detected_lang)
    answer = generate_answer(req.question, chunks)
    elapsed_ms = int((time.time() - start) * 1000)

    # Log this query to PostgreSQL
    database.log_query(
        user_query=req.question,
        detected_language=detected_lang,
        response_time_ms=elapsed_ms,
        retrieved_chunks=len(chunks),
        answer=answer,
    )

    return AskResponse(
        answer=answer,
        detected_language=detected_lang,
        sources=chunks,
    )

@app.get("/history")
def query_history(limit: int = 20):
    """Return the most recent queries from the logs."""
    rows = database.get_recent_queries(limit=limit)
    # Convert datetime objects to strings so they're JSON-serializable
    for r in rows:
        if r.get("created_at"):
            r["created_at"] = r["created_at"].isoformat()
    return {"queries": rows}