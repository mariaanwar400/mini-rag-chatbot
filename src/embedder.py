import numpy as np
from sentence_transformers import SentenceTransformer

def load_embedding_model(model_name):
    """Embedding model load karta hai (ek baar call karna, phir reuse karna)."""
    return SentenceTransformer(model_name)


def generate_embeddings(model, texts):
    """List of texts (chunks ya query) ke embeddings generate karta hai."""
    embeddings = model.encode(texts, show_progress_bar=True)
    return np.array(embeddings).astype('float32')