import re

def split_into_sentences(text):
    """Text ko sentences mein todta hai, proper boundaries pe."""
    sentence_endings = re.compile(r'(?<!\b[A-Z])(?<=[.!?])\s+(?=[A-Z(])')
    sentences = sentence_endings.split(text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_by_sentences(text, target_chunk_size=1000, overlap_sentences=2):
    """
    Sentences ko group karke chunks banata hai, mid-sentence cut kiye bina.
    Duplicate chunks bhi avoid karta hai.
    """
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