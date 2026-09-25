from groq import Groq
from src.config import GROQ_API_KEY, GROQ_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS

client = Groq(api_key=GROQ_API_KEY)


def retrieve_chunks(query, model, index, chunks, k=3):
    """Query ke liye top-k relevant chunks retrieve karta hai."""
    query_embedding = model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, k)
    retrieved = [chunks[idx] for idx in indices[0]]
    return retrieved, distances[0]


def generate_answer(query, retrieved_chunks):
    """Retrieved chunks ko context bana kar Groq LLM se answer generate karta hai."""
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
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS
    )

    return response.choices[0].message.content