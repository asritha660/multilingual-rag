import os
import re
from dotenv import load_dotenv
import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from rank_bm25 import BM25Okapi
from google import genai
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0   # makes detection deterministic

# --- One-time setup ---
load_dotenv()
# Read the key from Streamlit secrets if deployed, otherwise from .env locally
api_key = st.secrets.get("GEMINI_API_KEY") if hasattr(st, "secrets") else None
if not api_key:
    api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
# --- Load the embedding model ONCE (cached) ---
@st.cache_resource
def load_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

embed_model = load_model()

# --- Helper: read + clean text from an uploaded PDF ---
def extract_clean_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    full_text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        page_text = re.sub(r"[^\x00-\x7F\u0900-\u097F]+", " ", page_text)
        page_text = re.sub(r"\s+", " ", page_text)
        full_text += page_text + "\n"
    return full_text

# --- Helper: process a PDF into the vector database (REPLACES old content) ---
def process_pdf(uploaded_file):
    text = extract_clean_text(uploaded_file)
    splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
    chunks = splitter.split_text(text)
    embeddings = embed_model.encode(chunks)

    client = chromadb.PersistentClient(path="vector_store")
    # Wipe any previous document so we only answer about the new one
    if "documents" in [c.name for c in client.list_collections()]:
        client.delete_collection("documents")
    collection = client.create_collection("documents")
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        embeddings=embeddings.tolist(),
        documents=chunks,
    )
    return len(chunks)

# --- Helper: connect to the existing database ---
def get_collection():
    client = chromadb.PersistentClient(path="vector_store")
    return client.get_collection("documents")

# --- Hybrid retrieval ---
def retrieve(question):
    collection = get_collection()
    q_emb = embed_model.encode(question).tolist()
    all_chunks = collection.get()["documents"]

    vector_chunks = collection.query(query_embeddings=[q_emb], n_results=5)["documents"][0]

    tokenized = [c.lower().split() for c in all_chunks]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(question.lower().split())
    top_bm25 = sorted(zip(scores, all_chunks), key=lambda x: x[0], reverse=True)[:5]
    keyword_chunks = [c for s, c in top_bm25]

    merged = []
    for ch in vector_chunks + keyword_chunks:
        if ch not in merged:
            merged.append(ch)
    return merged

# --- Generation ---
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

# ================= THE WEB PAGE =================

st.title("📚 Multilingual RAG Assistant")
st.write("Upload a PDF, then ask questions about it — in any language!")

# --- Upload section ---
uploaded = st.file_uploader("Upload a PDF", type="pdf")
if uploaded is not None:
    if st.button("Process this PDF"):
        with st.spinner("Reading, chunking, and embedding your document..."):
            n = process_pdf(uploaded)
        st.success(f"Done! Stored {n} chunks. You can now ask questions below.")

st.divider()

# --- Question section ---
question = st.text_input("Your question:", placeholder="e.g. What is this document about?")
if st.button("Ask"):
    if not question:
        st.warning("Please type a question first.")
    else:
        try:
            # Detect the language of the question
            detected_lang = detect(question)
            lang_names = {
                "en": "English", "hi": "Hindi", "es": "Spanish", "fr": "French",
                "de": "German", "te": "Telugu", "ta": "Tamil", "bn": "Bengali",
                "ar": "Arabic", "zh-cn": "Chinese", "ja": "Japanese", "ru": "Russian",
                "pt": "Portuguese", "it": "Italian",
            }
            lang_display = lang_names.get(detected_lang, detected_lang.upper())
            st.caption(f"🌐 Detected language: **{lang_display}** (`{detected_lang}`)")

            with st.spinner("Searching and thinking..."):
                chunks = retrieve(question)
                answer = generate_answer(question, chunks)
            st.subheader("Answer")
            st.write(answer)
            with st.expander("📄 See the sources used"):
                for i, ch in enumerate(chunks):
                    st.markdown(f"**Source {i+1}:** {ch}")
        except Exception:
            st.error("No document found. Please upload and process a PDF first.")
