import re
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ---------- Step 1: Load PDF ----------
pdf_path = "data/document.pdf"
reader = PdfReader(pdf_path)

all_text = ""
for page in reader.pages:
    page_text = page.extract_text()
    all_text += page_text + "\n"

print(f"Total characters: {len(all_text)}")

# ---------- Step 2: Sentence-Aware Chunking ----------
def split_into_sentences(text):
    """
    Text ko sentences mein todta hai (period, exclamation, question mark ke baad,
    lekin common abbreviations jaise 'No.' 'Art.' ko galti se split nahi karta).
    """
    # Simple lekin effective sentence-boundary regex
    sentence_endings = re.compile(r'(?<!\b[A-Z])(?<=[.!?])\s+(?=[A-Z(])')
    sentences = sentence_endings.split(text)
    return [s.strip() for s in sentences if s.strip()]

def chunk_by_sentences(text, target_chunk_size=1000, overlap_sentences=2):
    sentences = split_into_sentences(text)
    print(f"Total sentences found: {len(sentences)}")

    chunks = []
    current_chunk_sentences = []
    current_length = 0

    for sentence in sentences:
        current_chunk_sentences.append(sentence)
        current_length += len(sentence)

        if current_length >= target_chunk_size:
            chunk = " ".join(current_chunk_sentences)

            # Duplicate se bachne ke liye: sirf tab add karo jab yeh pichle chunk se same na ho
            if not chunks or chunk != chunks[-1]:
                chunks.append(chunk)

            # Overlap ke liye pichle N sentences rakhein, lekin sirf tab
            # jab current group mein overlap se zyada sentences hon
            if len(current_chunk_sentences) > overlap_sentences:
                current_chunk_sentences = current_chunk_sentences[-overlap_sentences:]
            else:
                current_chunk_sentences = []  # poora consume ho gaya, fresh start karo

            current_length = sum(len(s) for s in current_chunk_sentences)

    if current_chunk_sentences:
        final_chunk = " ".join(current_chunk_sentences)
        if not chunks or final_chunk != chunks[-1]:
            chunks.append(final_chunk)

    return chunks

chunks = chunk_by_sentences(all_text, target_chunk_size=1000, overlap_sentences=2)
print(f"Total chunks created: {len(chunks)}")

print("\n--- Chunk 0 preview ---")
print(chunks[0][:300])
print("\n--- Chunk 1 preview (ab clean sentence se start hona chahiye) ---")
print(chunks[1][:300])

# ---------- Step 3: Embeddings ----------
print("\nLoading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Generating embeddings...")
embeddings = model.encode(chunks, show_progress_bar=True)
embeddings = np.array(embeddings).astype('float32')

# ---------- Step 4: FAISS Vector Store ----------
dimension = embeddings.shape[1]
print(f"\nCreating FAISS index with dimension: {dimension}")

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print(f"Total vectors in index: {index.ntotal}")

# ---------- Test Query ----------
query = "What are high-risk AI systems?"
print(f"\n--- Test Query: '{query}' ---")

query_embedding = model.encode([query]).astype('float32')
k = 3
distances, indices = index.search(query_embedding, k)

print(f"\nTop {k} most relevant chunks:")
for rank, idx in enumerate(indices[0]):
    print(f"\n[Rank {rank+1}] Distance: {distances[0][rank]:.4f}")
    print(f"Chunk index: {idx}")
    print(f"Content preview: {chunks[idx][:300]}")

# ---------- Diagnostic: Find the "purpose" chunk automatically ----------
print("\n--- Diagnostic: 'purpose of this Regulation' wala chunk dhoondte hain ---")
for i, chunk in enumerate(chunks):
    if "purpose of this Regulation" in chunk:
        purpose_chunk_embedding = embeddings[i].reshape(1, -1)
        distance_to_purpose_chunk = np.linalg.norm(query_embedding - purpose_chunk_embedding)
        all_distances, all_indices = index.search(query_embedding, len(chunks))
        rank_of_purpose_chunk = np.where(all_indices[0] == i)[0][0]
        print(f"Found at chunk index {i}, distance: {distance_to_purpose_chunk:.4f}, rank: #{rank_of_purpose_chunk + 1}")
        break