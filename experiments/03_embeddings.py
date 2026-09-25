from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ---------- Step 1: Load PDF ----------
pdf_path = "data/document.pdf"
reader = PdfReader(pdf_path)

all_text = ""
for page in reader.pages:
    page_text = page.extract_text()
    all_text += page_text + "\n"

# ---------- Step 2: Chunking ----------
def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks

chunks = chunk_text(all_text, chunk_size=800, overlap=100)
print(f"Total chunks: {len(chunks)}")

# ---------- Step 3: Embeddings ----------
print("\nLoading embedding model... (pehli baar thoda time lega, model download hoga)")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Generating embeddings for all chunks...")
embeddings = model.encode(chunks, show_progress_bar=True)

print(f"\nEmbeddings shape: {embeddings.shape}")
print(f"Yeh matlab: {embeddings.shape[0]} chunks, har ek {embeddings.shape[1]}-dimensional vector")

# Ek single embedding ka example dekhein
print(f"\n--- Chunk 0 ka embedding (pehle 10 numbers) ---")
print(embeddings[0][:10])