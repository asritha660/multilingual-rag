# Multilingual RAG Assistant

A production-grade Retrieval-Augmented Generation pipeline that answers questions about uploaded PDF documents in multiple languages, with grounded answers, cited sources, and a measurable evaluation harness.

Built as a multi-service application: containerized FastAPI backend, decoupled Streamlit frontend, persistent ChromaDB vector store, PostgreSQL for users and query telemetry, and JWT authentication.

**Repository:** https://github.com/asritha660/multilingual-rag

## What this project demonstrates

- **Multilingual retrieval** with shared-embedding-space queries across English, Hindi, and 50+ languages supported by the underlying model.
- **Hybrid retrieval**: dense vector search plus BM25 keyword search, fused and deduplicated.
- **Conditional reranking**: a cross-encoder reranker is applied to English queries (the model is English-only) and skipped for other languages, which then use a wider hybrid top-K.
- **Grounded generation**: answers are produced strictly from retrieved chunks, returned in the same language as the question, with a sources panel for verification.
- **Measured behavior**: a versioned test set, automated retrieval and faithfulness evaluation scripts, and PostgreSQL-captured latency telemetry on every query.

## Measured Results

Evaluated on a 7-question test set (6 English plus 1 Hindi) over a fully fabricated test corpus (`test_document.pdf`). The set includes a deliberate keyword-dilution stress-test question.

| Metric | Value |
|---|---|
| Hit Rate (recall proxy) | **7/7 = 1.00** |
| Average Precision@K | **0.35** |
| Average retrieval latency | **0.17s** |
| End-to-end query latency | **~2.9s** (generation dominates) |
| Faithfulness score | *Full rerun pending; partial run scored 1.0* |

The system achieves saturated recall on this test set; the genuine remaining weakness is precision, not recall. Detailed analysis lives in [REPORT.md](REPORT.md).

## Architecture

```
                                  +---------------------------+
                                  |    Streamlit Frontend     |
                                  |  (login, upload, query)   |
                                  +-------------+-------------+
                                                |
                                       HTTP + JWT bearer
                                                |
                                                v
+---------------------------+   +---------------------------+
|       PostgreSQL          |<--|     FastAPI Backend       |
|  users, documents,        |   |   /register /login        |
|  query_logs (latency)     |   |   /upload  /ask  /history |
+---------------------------+   +-------------+-------------+
                                              |
                          +-------------------+-------------------+
                          |                                       |
                          v                                       v
              +-----------------------+              +-----------------------+
              |   ChromaDB (local)    |              |   Gemini 2.5 Flash    |
              |   persistent vectors  |              |   grounded answer     |
              +-----------------------+              +-----------------------+
                          ^
                          |
              embed (paraphrase-multilingual-MiniLM-L12-v2)
                          ^
                          |
                   chunk (250 chars, 50 overlap)
                          ^
                          |
                    extract + clean (pypdf)
                          ^
                          |
                       PDF upload
```

The FastAPI backend is containerized (Python 3.11-slim, 622 MB compressed image). For development, the backend connects to the host PostgreSQL via `host.docker.internal`.

## Tech Stack

| Layer | Choice |
|---|---|
| Backend API | FastAPI |
| Frontend | Streamlit |
| Authentication | JWT (`python-jose`) with bcrypt password hashing |
| Embeddings | `sentence-transformers`, `paraphrase-multilingual-MiniLM-L12-v2` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (English-only, conditional) |
| Vector DB | ChromaDB (persistent local store) |
| Keyword Search | BM25 via `rank-bm25` |
| Generation | Google Gemini 2.5 Flash (via `google-genai`) |
| Relational DB | PostgreSQL (users, documents, query_logs) |
| Language Detection | `langdetect` (seeded for determinism) |
| PDF Parsing | `pypdf` |
| Chunking | `langchain-text-splitters` RecursiveCharacterTextSplitter |
| Container | Docker (Python 3.11-slim) |
| Runtime | Python 3.11 |

## Project Structure

```
multilingual-rag/
├── backend/                  FastAPI app, retrieval, ingestion, auth, DB
│   ├── main.py               API endpoints (/register, /login, /upload, /ask, /history)
│   ├── ingest.py             PDF ingestion CLI + importable function
│   ├── database.py           PostgreSQL connection and schema
│   └── auth.py               JWT issuance and verification
├── frontend/
│   └── app.py                Streamlit UI calling the backend over HTTP
├── evaluation/
│   ├── test_set.json         Versioned test questions and expected keywords
│   ├── evaluate_retrieval.py Hit Rate, Precision@K, latency (no LLM calls)
│   └── evaluate_faithfulness.py Gemini-as-judge faithfulness scoring
├── Dockerfile                Backend container (python:3.11-slim, port 8000)
├── requirements.txt
├── REPORT.md                 Evaluation methodology, numbers, analysis
└── README.md                 (this file)
```

## Setup

### Prerequisites

- Python 3.11
- PostgreSQL 14+ running locally (or remotely; set `DB_HOST` accordingly)
- A Gemini API key from https://aistudio.google.com/apikey (free tier works)
- Docker Desktop (optional, only for the containerized path)

### Local development (no Docker)

```powershell
git clone https://github.com/asritha660/multilingual-rag.git
cd multilingual-rag
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_key_here
JWT_SECRET_KEY=any_long_random_string
DB_NAME=ragdb
DB_USER=postgres
DB_PASSWORD=your_local_postgres_password
DB_HOST=localhost
DB_PORT=5432
```

Create the database (one-time):

```sql
CREATE DATABASE ragdb;
```

Tables are created automatically on backend startup.

Ingest a document:

```powershell
python backend\ingest.py path\to\your_document.pdf
```

Run the backend:

```powershell
uvicorn backend.main:app --reload --port 8000
```

In a second terminal, run the frontend:

```powershell
streamlit run frontend\app.py
```

The app opens at `http://localhost:8501`. The backend is at `http://localhost:8000`.

### Containerized backend (Docker)

```powershell
docker build -t multilingual-rag-backend .
docker run -d --name rag-backend -p 8000:8000 -e DB_HOST=host.docker.internal multilingual-rag-backend
```

The backend will be reachable at `http://localhost:8000` and will connect to PostgreSQL on the host machine via `host.docker.internal`.

Stop and remove:

```powershell
docker stop rag-backend
docker rm rag-backend
```

### Running the evaluation harness

After ingesting a document:

```powershell
python -m evaluation.evaluate_retrieval
python -m evaluation.evaluate_faithfulness   # uses Gemini quota
```

## Screenshots

*Screenshots to be added.*

## Engineering Notes

A few decisions that turned out to matter in practice:

- **Conditional reranking.** The cross-encoder reranker is English-only. Applying it to non-English queries degraded retrieval, so the pipeline detects language first and skips reranking for non-English queries, falling back to a wider hybrid top-K. This kept Hindi retrieval working without retraining anything.
- **`bcrypt==4.0.1` pinned.** Newer bcrypt versions break `passlib==1.7.4` with a misleading "password cannot be longer than 72 bytes" error. Pinning the older version is the documented workaround.
- **CPU-only torch on cloud deploys.** The default `torch` install pulls CUDA wheels and busts the 1 GB free-tier memory budget on Streamlit Cloud. `--extra-index-url https://download.pytorch.org/whl/cpu` in `requirements.txt` keeps the install lean.
- **`host.docker.internal` for the containerized backend.** Inside a container, `localhost` means the container itself. The Docker-managed DNS name `host.docker.internal` points back to the Windows host where PostgreSQL is running, which is the simplest path before moving everything into `docker-compose`.
- **Free-tier Gemini limits.** `gemini-2.5-flash` allows 20 requests per day and 5 per minute on the free tier. The faithfulness eval script implements 30-second throttling between calls and retry-on-429, so it degrades gracefully when the daily quota is hit mid-run.

## License

This project is for portfolio and learning purposes. The test document `test_document.pdf` is a fictional corpus authored specifically for evaluation and is not based on any real organization.
