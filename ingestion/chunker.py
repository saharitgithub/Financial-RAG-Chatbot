import re

def chunk_text(text, size=500, overlap=100):
    """
    Split text into overlapping chunks.
    Smaller size (500) ensures formulas stay intact within a chunk.
    Overlap (100) ensures context is not lost between chunks.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + size
        chunk = text[start:end]

        # Avoid cutting mid-word
        if end < len(text):
            last_space = chunk.rfind(" ")
            if last_space != -1:
                chunk = chunk[:last_space]

        chunk = chunk.strip()

        if chunk:
            chunks.append(chunk)

        start += size - overlap

    return chunks