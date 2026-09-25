import os
import pickle
import faiss

def build_faiss_index(embeddings):
    """Embeddings se naya FAISS index banata hai."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def save_index_and_chunks(index, chunks, faiss_path, chunks_path):
    """FAISS index aur chunks ko disk pe save karta hai, taake dobara build na karna paड़े."""
    os.makedirs(os.path.dirname(faiss_path), exist_ok=True)
    faiss.write_index(index, faiss_path)
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)


def load_index_and_chunks(faiss_path, chunks_path):
    """Pehle se saved FAISS index aur chunks disk se load karta hai."""
    index = faiss.read_index(faiss_path)
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def index_exists(faiss_path, chunks_path):
    """Check karta hai ke saved index files exist karti hain ya nahi."""
    return os.path.exists(faiss_path) and os.path.exists(chunks_path)