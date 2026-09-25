import os
from dotenv import load_dotenv

load_dotenv()

# ---------- Paths ----------
PDF_PATH = "data/document.pdf"
STORAGE_DIR = "storage"
FAISS_INDEX_PATH = os.path.join(STORAGE_DIR, "faiss_index.bin")
CHUNKS_PATH = os.path.join(STORAGE_DIR, "chunks.pkl")

# ---------- Chunking Settings ----------
CHUNK_TARGET_SIZE = 1000
CHUNK_OVERLAP_SENTENCES = 2

# ---------- Embedding Model ----------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ---------- LLM Settings ----------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "openai/gpt-oss-120b"
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 500

# ---------- Retrieval Settings ----------
TOP_K_CHUNKS = 3