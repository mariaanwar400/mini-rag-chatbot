from src.config import FAISS_INDEX_PATH, CHUNKS_PATH, EMBEDDING_MODEL_NAME, TOP_K_CHUNKS
from src.vector_store import load_index_and_chunks
from src.embedder import load_embedding_model
from src.rag_pipeline import retrieve_chunks, generate_answer

print("Loading saved index and chunks from disk (fast, no re-computation)...")
index, chunks = load_index_and_chunks(FAISS_INDEX_PATH, CHUNKS_PATH)
print(f"Loaded {index.ntotal} vectors and {len(chunks)} chunks.")

print("\nLoading embedding model...")
model = load_embedding_model(EMBEDDING_MODEL_NAME)

query = "What are high-risk AI systems?"
print(f"\nQuery: {query}")

retrieved, distances = retrieve_chunks(query, model, index, chunks, k=TOP_K_CHUNKS)

print("\nGenerating answer...")
answer = generate_answer(query, retrieved)

print("\n" + "="*60)
print("ANSWER:")
print("="*60)
print(answer)