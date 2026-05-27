from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# --- Step 1: Read and chunk the PDF (same as before) ---
reader = PdfReader("sample.pdf")
full_text = ""
for page in reader.pages:
    full_text += page.extract_text() + "\n"

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_text(full_text)
print("Number of chunks:", len(chunks))

# --- Step 2: Load the multilingual embedding model ---
# The first time you run this, it DOWNLOADS the model (a one-time ~120MB download).
print("Loading embedding model (first run downloads it, please wait)...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# --- Step 3: Turn every chunk into a vector of numbers ---
embeddings = model.encode(chunks)

# --- Step 4: Inspect what we got ---
print("\nWe created", len(embeddings), "embeddings.")
print("Each embedding is a list of", len(embeddings[0]), "numbers.")
print("\nHere are the first 8 numbers of the first chunk's embedding:")
print(embeddings[0][:8])