import os
import re
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from groq import Groq

# ---------- Load API Key from .env ----------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY nahi mili! .env file check karein.")

client = Groq(api_key=groq_api_key)

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
    sentence_endings = re.compile(r'(?<!\b[A-Z])(?<=[.!?])\s+(?=[A-Z(])')
    sentences = sentence_endings.split(text)
    return [s.strip() for s in sentences if s.strip()]

def chunk_by_sentences(text, target_chunk_size=1000, overlap_sentences=2):
    sentences = split_into_sentences(text)
    chunks = []
    current_chunk_sentences = []
    current_length = 0

    for sentence in sentences:
        current_chunk_sentences.append(sentence)
        current_length += len(sentence)

        if current_length >= target_chunk_size:
            chunk = " ".join(current_chunk_sentences)
            if not chunks or chunk != chunks[-1]:
                chunks.append(chunk)

            if len(current_chunk_sentences) > overlap_sentences:
                current_chunk_sentences = current_chunk_sentences[-overlap_sentences:]
            else:
                current_chunk_sentences = []
            current_length = sum(len(s) for s in current_chunk_sentences)

    if current_chunk_sentences:
        final_chunk = " ".join(current_chunk_sentences)
        if not chunks or final_chunk != chunks[-1]:
            chunks.append(final_chunk)

    return chunks

chunks = chunk_by_sentences(all_text, target_chunk_size=1000, overlap_sentences=2)
print(f"Total chunks: {len(chunks)}")

# ---------- Step 3: Embeddings ----------
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Generating embeddings...")
embeddings = model.encode(chunks, show_progress_bar=True)
embeddings = np.array(embeddings).astype('float32')

# ---------- Step 4: FAISS Vector Store ----------
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print(f"Total vectors in index: {index.ntotal}")

# ---------- Step 5: Retrieval Function ----------
def retrieve_chunks(query, k=3):
    query_embedding = model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, k)
    retrieved = [chunks[idx] for idx in indices[0]]
    return retrieved, distances[0]

# ---------- Step 5: Generation Function ----------
def generate_answer(query, retrieved_chunks):
    context = "\n\n---\n\n".join(retrieved_chunks)

    prompt = f"""You are a helpful assistant answering questions based ONLY on the provided context from a legal document (EU AI Act).

Context:
{context}

Question: {query}

Instructions:
- Answer using ONLY the information in the context above.
- If the context does not contain enough information to answer, say "I don't have enough information in the provided document to answer this question."
- Be concise and accurate.

Answer:"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500
    )

    return response.choices[0].message.content

# ---------- Test: Full RAG Pipeline ----------
query = "What are high-risk AI systems?"
print(f"\n{'='*60}")
print(f"QUERY: {query}")
print(f"{'='*60}")

retrieved_chunks, distances = retrieve_chunks(query, k=3)

print("\n--- Retrieved Chunks (Sources) ---")
for i, (chunk, dist) in enumerate(zip(retrieved_chunks, distances)):
    print(f"\n[Source {i+1}] (distance: {dist:.4f})")
    print(chunk[:200] + "...")

print("\n--- Generating Answer via Groq LLaMA-3 ---")
answer = generate_answer(query, retrieved_chunks)

print(f"\n{'='*60}")
print("FINAL ANSWER:")
print(f"{'='*60}")
print(answer)