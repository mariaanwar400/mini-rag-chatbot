# 🤖 EU AI Act — RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about the EU AI Act (Regulation (EU) 2024/1689) using semantic search and an LLM — with support for uploading and chatting with any custom PDF document.

🔗 **Live Demo**: [mariaanwar-rag-chatbot.streamlit.app](https://mariaanwar-rag-chatbot.streamlit.app)

---

## 📌 Overview

This project demonstrates a complete RAG pipeline built from scratch — document ingestion, chunking, embedding, vector search, and grounded LLM generation — wrapped in an interactive Streamlit chat interface with source attribution.

Instead of relying on an LLM's general knowledge (which can hallucinate or go out of date), this chatbot retrieves the most relevant passages from the actual document and instructs the LLM to answer **only** from that retrieved context — making answers verifiable and traceable back to their source.

## ✨ Features

- 💬 Conversational chat interface with message history
- 📄 Upload any PDF and chat with it (dynamically re-indexes the document)
- 🔍 Source attribution — view the exact retrieved passages behind every answer
- ⚡ Fast retrieval using FAISS vector search
- 🔒 Secure API key handling via environment variables / Streamlit Secrets

## 🏗️ Architecture

The system follows the standard three-stage RAG pipeline:
```
INDEXING (one-time, per document)
PDF → Text Extraction (pypdf) → Sentence-Aware Chunking → Embeddings (sentence-transformers) → FAISS Vector Store

RETRIEVAL (per query)
User Query → Query Embedding → FAISS Similarity Search → Top-K Relevant Chunks

GENERATION (per query)
Retrieved Chunks + Query → Prompt Template → Groq LLM (LLaMA/GPT-OSS) → Grounded Answer
```

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Text Extraction | pypdf |
| Chunking | Custom sentence-aware chunker (regex-based sentence splitting + overlap) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector Store | FAISS (`IndexFlatL2`) |
| LLM | Groq API (`openai/gpt-oss-120b`) |
| UI | Streamlit |
| Deployment | Streamlit Community Cloud |

## 🧠 Key Technical Decisions

**Sentence-aware chunking over naive character splitting**
An initial character-count-based chunking approach was frequently cutting chunks mid-sentence (and even mid-word), which measurably degraded retrieval quality — a chunk containing the exact answer to a test query ranked as low as #278 out of ~860 chunks. Switching to sentence-boundary-aware chunking (grouping whole sentences up to a target size, with sentence-level overlap) improved that same chunk's rank to #12, without changing the embedding model.

**Embedding model choice**
`all-MiniLM-L6-v2` (384 dimensions) was chosen over the larger `all-mpnet-base-v2` (768 dimensions) after testing showed the larger model did not improve retrieval on this document — the actual bottleneck was chunk quality, not model capacity. MiniLM is also significantly faster on CPU, which matters for free-tier deployment.

**Index persistence**
The FAISS index and chunks are built once (`src/build_index.py`) and saved to disk (`storage/`), rather than being rebuilt on every app run — critical for a responsive Streamlit experience.

## ⚠️ Known Limitations

- Retrieval performs very well on specific, factual queries (e.g. *"What are high-risk AI systems?"*) but is noticeably weaker on abstract, paraphrased queries (e.g. *"What is the purpose of this Regulation?"*), where the correct chunk can rank outside the top 3. This is a known trade-off of small, general-purpose embedding models on dense, repetitive legal text.
- Chat history is session-based (not persisted across app restarts).
- No conversation-aware retrieval yet — each question is treated independently (no follow-up context).

## 🚀 Running Locally

**1. Clone the repository**
```bash
git clone https://github.com/mariaanwar400/mini-rag-chatbot.git
cd mini-rag-chatbot
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up your Groq API key**

Create a `.env` file in the project root:

GROQ_API_KEY=your_groq_api_key_here

**5. Build the vector index**
```bash
python -m src.build_index
```

**6. Run the app**
```bash
streamlit run app.py
```

## 📁 Project Structure
```
mini-rag-chatbot/
├── data/
├── storage/
├── src/
│ ├── config.py
│ ├── document_loader.py
│ ├── chunker.py
│ ├── embedder.py
│ ├── vector_store.py
│ ├── rag_pipeline.py
│ └── build_index.py
├── experiments/
├── app.py
└── requirements.txt
```

## 👩‍💻 Author

**Maria Anwar**
[LinkedIn](https://linkedin.com/in/mariaanwar400) · [GitHub](https://github.com/mariaanwar400)