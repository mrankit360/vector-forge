
def chunk_text(text, chunk_words=250, overlap_words=30):
    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")

    if overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("overlap_words must be between 0 and chunk_words")

    words = text.split()

    if not words:
        return []

    if len(words) <= chunk_words:
        return [text]

    step = chunk_words - overlap_words
    chunks = []

    for start in range(0, len(words), step):
        chunk = words[start:start + chunk_words]
        if chunk:
            chunks.append(" ".join(chunk))

        if start + chunk_words >= len(words):
            break

    return chunks
