import streamlit as st
import os
from src.config import (
    PDF_PATH, FAISS_INDEX_PATH, CHUNKS_PATH, EMBEDDING_MODEL_NAME,
    TOP_K_CHUNKS, CHUNK_TARGET_SIZE, CHUNK_OVERLAP_SENTENCES
)
from src.vector_store import load_index_and_chunks, index_exists, build_faiss_index, save_index_and_chunks
from src.embedder import load_embedding_model, generate_embeddings
from src.document_loader import load_pdf_text
from src.chunker import chunk_by_sentences
from src.rag_pipeline import retrieve_chunks, generate_answer

# ---------- Page Config ----------
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🤖 RAG Chatbot")
st.caption("Upload a PDF and ask questions about it, answered using Retrieval-Augmented Generation.")

# ---------- Cached Resource Loading ----------
@st.cache_resource
def load_resources():
    index, chunks = load_index_and_chunks(FAISS_INDEX_PATH, CHUNKS_PATH)
    model = load_embedding_model(EMBEDDING_MODEL_NAME)
    return index, chunks, model

# ---------- Function: Build New Index from Uploaded PDF ----------
def build_index_from_upload(uploaded_file):
    # Uploaded file ko data/ folder mein save karein
    os.makedirs(os.path.dirname(PDF_PATH), exist_ok=True)
    with open(PDF_PATH, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Poora indexing pipeline chalayein
    text = load_pdf_text(PDF_PATH)
    chunks = chunk_by_sentences(text, CHUNK_TARGET_SIZE, CHUNK_OVERLAP_SENTENCES)

    model = load_embedding_model(EMBEDDING_MODEL_NAME)
    embeddings = generate_embeddings(model, chunks)

    index = build_faiss_index(embeddings)
    save_index_and_chunks(index, chunks, FAISS_INDEX_PATH, CHUNKS_PATH)

# ---------- Sidebar: PDF Upload ----------
with st.sidebar:
    st.header("📁 Document")
    uploaded_file = st.file_uploader("Upload a PDF to chat with", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Process Document"):
            with st.spinner("Processing PDF... this may take a minute."):
                build_index_from_upload(uploaded_file)
                st.cache_resource.clear()  # Purana cached index/model clear karein
                st.session_state.messages = []  # Chat history reset karein
            st.success("Document processed! Ab aap sawal poochh sakti hain.")
            st.rerun()

    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ---------- Check Index Exists ----------
if not index_exists(FAISS_INDEX_PATH, CHUNKS_PATH):
    st.info("👈 Please upload a PDF from the sidebar to get started.")
    st.stop()

index, chunks, model = load_resources()

# ---------- Initialize Conversation History ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- Display Past Messages ----------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📄 View Sources"):
                for i, (chunk, dist) in enumerate(message["sources"]):
                    st.markdown(f"**Source {i+1}** (similarity distance: {dist:.4f})")
                    st.text(chunk[:400] + "...")
                    st.divider()

# ---------- Chat Input ----------
user_query = st.chat_input("Ask a question about the document...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching document and generating answer..."):
            retrieved_chunks, distances = retrieve_chunks(user_query, model, index, chunks, k=TOP_K_CHUNKS)
            answer = generate_answer(user_query, retrieved_chunks)

        st.markdown(answer)

        with st.expander("📄 View Sources"):
            for i, (chunk, dist) in enumerate(zip(retrieved_chunks, distances)):
                st.markdown(f"**Source {i+1}** (similarity distance: {dist:.4f})")
                st.text(chunk[:400] + "...")
                st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": list(zip(retrieved_chunks, distances))
    })