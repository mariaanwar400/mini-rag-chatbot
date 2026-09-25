from src.config import (
    PDF_PATH, FAISS_INDEX_PATH, CHUNKS_PATH,
    CHUNK_TARGET_SIZE, CHUNK_OVERLAP_SENTENCES, EMBEDDING_MODEL_NAME
)
from src.document_loader import load_pdf_text
from src.chunker import chunk_by_sentences
from src.embedder import load_embedding_model, generate_embeddings
from src.vector_store import build_faiss_index, save_index_and_chunks

print("Step 1: Loading PDF...")
text = load_pdf_text(PDF_PATH)
print(f"Total characters: {len(text)}")

print("\nStep 2: Chunking...")
chunks = chunk_by_sentences(text, CHUNK_TARGET_SIZE, CHUNK_OVERLAP_SENTENCES)
print(f"Total chunks: {len(chunks)}")

print("\nStep 3: Loading embedding model...")
model = load_embedding_model(EMBEDDING_MODEL_NAME)

print("\nStep 4: Generating embeddings...")
embeddings = generate_embeddings(model, chunks)

print("\nStep 5: Building FAISS index...")
index = build_faiss_index(embeddings)
print(f"Total vectors indexed: {index.ntotal}")

print("\nStep 6: Saving index and chunks to disk...")
save_index_and_chunks(index, chunks, FAISS_INDEX_PATH, CHUNKS_PATH)
print(f"Saved to: {FAISS_INDEX_PATH} and {CHUNKS_PATH}")

print("\n✅ Indexing complete! Ab Streamlit app isse reuse kar sakta hai.")