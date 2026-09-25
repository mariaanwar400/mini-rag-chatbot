from pypdf import PdfReader

# ---------- Step 1 se code (reuse) ----------
pdf_path = "data/document.pdf"
reader = PdfReader(pdf_path)

all_text = ""
for page in reader.pages:
    page_text = page.extract_text()
    all_text += page_text + "\n"

print(f"Total characters: {len(all_text)}")

# ---------- Step 2: Chunking Logic ----------
def chunk_text(text, chunk_size=800, overlap=100):
    """
    Text ko fixed-size overlapping chunks mein todta hai.
    chunk_size: har chunk ke characters
    overlap: consecutive chunks ke beech overlapping characters
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        # Agla chunk overlap ke saath shuru hoga
        start = end - overlap

    return chunks

# Chunking apply karein
chunks = chunk_text(all_text, chunk_size=800, overlap=100)

print(f"Total chunks created: {len(chunks)}")
print("\n--- Chunk 0 (first chunk) ---\n")
print(chunks[0])
print("\n--- Chunk 1 (second chunk, notice overlap with chunk 0's end) ---\n")
print(chunks[1])