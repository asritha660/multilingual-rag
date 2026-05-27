# 📚 Multilingual RAG Assistant

A Retrieval Augmented Generation (RAG) pipeline that lets users upload PDF documents and ask questions about them in any language, with grounded, cited answers.

Built end to end as a hands on project covering AI engineering, vector search, hybrid retrieval, and full stack Python development.

## Features

* 📄 **PDF upload and live ingestion.** Drop in any PDF and the app extracts, cleans, chunks, embeds, and indexes it on the spot.
* 🌍 **Cross lingual retrieval.** Ask in Hindi, English, Spanish, and many more languages. The multilingual embedding model maps queries and source text into a shared meaning space.
* 🔍 **Hybrid search.** Combines semantic vector search with BM25 keyword search to surface both conceptually related and exact term matches.
* 🤖 **Grounded answers with sources.** Every answer is generated only from the retrieved chunks, and includes a "See the sources used" panel to verify it isn't hallucinated.
* 🗣️ **Language aware responses.** Answers come back in the same language as the question.

## Tech Stack

| Layer | Tool |
|-------|------|
| Frontend | Streamlit |
| LLM | Google Gemini 2.5 Flash (via `google-genai`) |
| Embeddings | `sentence-transformers`, model `paraphrase-multilingual-MiniLM-L12-v2` (50+ languages, runs locally) |
| Vector Database | ChromaDB (persistent, local) |
| Keyword Search | BM25 (`rank-bm25`) |
| PDF Parsing | `pypdf` |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| Language | Python 3.11 |

## Architecture

```
User -> Streamlit UI -> PDF upload
                            |
                            v
                extract -> clean -> chunk (250 chars, 50 overlap)
                            |
                            v
                multilingual embeddings -> ChromaDB (persistent)
                            |
                            v
User question -> embed query
                |
        +-------+-------+
        v               v
  Vector search   BM25 keyword search
        +-------+-------+
                v
            merge + dedupe
                v
       Gemini 2.5 Flash -> grounded answer (in query's language)
```

## Setup

**1. Clone and enter the repo**

```bash
git clone https://github.com/asritha660/multilingual-rag.git
cd multilingual-rag
```

**2. Create a virtual environment**

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows
# or
source venv/bin/activate      # macOS / Linux
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Add your Gemini API key**

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_key_here
```

Get a free key at https://aistudio.google.com/apikey (no credit card required).

**5. Run the app**

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Project Structure

```
multilingual-rag/
├── app.py                    # Streamlit web app (main entry point)
├── ingest.py                 # Standalone ingestion script
├── ask.py                    # Standalone question answering script
├── read_pdf.py               # PDF text extraction (learning step)
├── chunk_pdf.py              # Chunking demonstration
├── embed_chunks.py           # Embedding demonstration
├── search.py                 # Pure vector retrieval demonstration
├── requirements.txt          # Python dependencies
├── .streamlit/config.toml    # Streamlit configuration
├── .gitignore                # Excludes .env, venv/, vector_store/
└── sample.pdf                # Sample document for testing
```

## Future Improvements

* Reranking model for sharper top 1 precision
* Streaming responses
* Multi document support (currently one document at a time)
* FastAPI backend separation
* Authentication and user specific document spaces
* Evaluation harness (Precision@K, faithfulness, hallucination rate)
* Docker and CI/CD deployment pipeline

## Notes

This project was built as a hands on exploration of production RAG patterns. Every layer (chunking, embeddings, retrieval, generation) was implemented step by step and tuned through observed failure modes. For example, pure vector search initially failed to surface chunks containing specific terms like "Qdrant" even though they semantically matched the query, which motivated adding BM25 keyword search alongside vector search to form the hybrid retrieval pipeline.
