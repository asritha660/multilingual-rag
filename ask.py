import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai

# --- Step 0: Load the secret API key from the .env file ---
load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# --- Step 1: Connect to the existing vector database ---
client = chromadb.PersistentClient(path="vector_store")
collection = client.get_collection("documents")

# --- Step 2: Load the embedding model (for the question only) ---
embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# --- Step 3: The question ---
question = "What vector database does this project recommend?"
question_embedding = embed_model.encode(question).tolist()

# --- Step 4: Retrieve the most relevant chunks ---
results = collection.query(query_embeddings=[question_embedding], n_results=6)
retrieved_chunks = results["documents"][0]

# DEBUG: show what we actually retrieved
# --- Step 4: HYBRID retrieval = vector search + keyword (BM25) search ---
from rank_bm25 import BM25Okapi

# Get ALL chunks from the database (for the keyword search to scan)
all_data = collection.get()
all_chunks = all_data["documents"]

# 4a. Vector search: top 5 by meaning
vector_results = collection.query(query_embeddings=[question_embedding], n_results=5)
vector_chunks = vector_results["documents"][0]

# 4b. Keyword search: top 5 by exact-word relevance (BM25)
tokenized_corpus = [c.lower().split() for c in all_chunks]
bm25 = BM25Okapi(tokenized_corpus)
tokenized_query = question.lower().split()
bm25_scores = bm25.get_scores(tokenized_query)
# Pair each chunk with its keyword score, sort, take top 5
top_bm25 = sorted(zip(bm25_scores, all_chunks), key=lambda x: x[0], reverse=True)[:5]
keyword_chunks = [chunk for score, chunk in top_bm25]

# 4c. Combine both lists, removing duplicates while preserving order
retrieved_chunks = []
for ch in vector_chunks + keyword_chunks:
    if ch not in retrieved_chunks:
        retrieved_chunks.append(ch)

# DEBUG: show what we actually retrieved
print("\n=== RETRIEVED CONTEXT (hybrid: vector + keyword) ===")
for i, ch in enumerate(retrieved_chunks):
    print(f"\n--- retrieved chunk {i} ---\n{ch[:200]}...")

# Join the retrieved chunks into one block of context
context = "\n\n".join(retrieved_chunks)

# --- Step 5: Build the prompt for the LLM ---
prompt = f"""You are a helpful assistant. Answer the user's question using ONLY the context below.
If the answer is not in the context, say you don't know. Answer in the same language as the question.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

# --- Step 6: Ask Gemini to generate the answer ---
llm = genai.GenerativeModel("gemini-2.5-flash")
response = llm.generate_content(prompt)

# --- Step 7: Show it ---
print(f"\nQUESTION: {question}\n")
print("=== ANSWER ===")
print(response.text)