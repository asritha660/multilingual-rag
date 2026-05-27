from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, util

# --- Step 1: Read and chunk the PDF ---
reader = PdfReader("sample.pdf")
full_text = ""
for page in reader.pages:
    full_text += page.extract_text() + "\n"

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_text(full_text)

# --- Step 2: Load model and embed all chunks ---
print("Loading model...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
chunk_embeddings = model.encode(chunks)

# --- Step 3: Ask a question and embed IT with the same model ---
question = "इस प्रोजेक्ट के लिए कौन सा डेटाबेस अच्छा है?"
question_embedding = model.encode(question)

# --- Step 4: Find the chunks closest in meaning to the question ---
scores = util.cos_sim(question_embedding, chunk_embeddings)[0]

# Pair each chunk with its score, then sort best-first
results = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)

# --- Step 5: Show the top 3 matches ---
print(f"\nQUESTION: {question}\n")
print("=== TOP 3 MOST RELEVANT CHUNKS ===")
for score, chunk in results[:3]:
    print(f"\n[similarity score: {score:.3f}]")
    print(chunk)