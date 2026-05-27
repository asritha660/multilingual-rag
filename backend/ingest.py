from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# --- Step 1: Read and chunk the PDF ---
import re

reader = PdfReader("sample.pdf")
full_text = ""
for page in reader.pages:
    page_text = page.extract_text()
    # Clean: drop weird box-drawing/control chars, collapse extra whitespace
    page_text = re.sub(r"[^\x00-\x7F\u0900-\u097F]+", " ", page_text)  # keep ASCII + Hindi
    page_text = re.sub(r"\s+", " ", page_text)
    full_text += page_text + "\n"
splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
chunks = splitter.split_text(full_text)
print("Chunks created:", len(chunks))

# --- Step 2: Embed the chunks ---
print("Loading model and embedding...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embeddings = model.encode(chunks)

# --- Step 3: Connect to a PERSISTENT ChromaDB (saves to a folder on disk) ---
client = chromadb.PersistentClient(path="vector_store")
# A "collection" is like a table. Reset it each run so we start clean.
client.delete_collection("documents") if "documents" in [c.name for c in client.list_collections()] else None
collection = client.create_collection("documents")

# --- Step 4: Store each chunk with its vector and an ID ---
collection.add(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    embeddings=embeddings.tolist(),
    documents=chunks,
)

print(f"Stored {collection.count()} chunks in the vector database.")
print("Done. Vectors saved to the 'vector_store' folder on disk.")