from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Step 1: Read ALL text from the PDF, page by page ---
reader = PdfReader("sample.pdf")
full_text = ""
for page in reader.pages:
    full_text += page.extract_text() + "\n"

print("Total characters extracted:", len(full_text))

# --- Step 2: Split that text into overlapping chunks ---
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # each chunk is up to ~500 characters
    chunk_overlap=100,   # consecutive chunks share ~100 characters
)
chunks = splitter.split_text(full_text)

print("Number of chunks created:", len(chunks))

# --- Step 3: Show us the first 3 chunks so we can see what they look like ---
for i in range(3):
    print(f"\n--- CHUNK {i} ---")
    print(chunks[i])