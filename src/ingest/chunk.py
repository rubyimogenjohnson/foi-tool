"""
chunk.py — split cleaned FOI text into overlapping chunks.

all-MiniLM-L6-v2 has a 256-token limit (~1 000 chars of English text).
We chunk at 800 chars (with 150-char overlap) so the prepended title
still fits comfortably within the model's window.

Strategy: split on paragraph boundaries, merge short paragraphs up to
CHUNK_SIZE, then backtrack by OVERLAP chars when starting the next chunk.
"""

import re

CHUNK_SIZE = 800   # characters
OVERLAP = 150      # characters carried into the next chunk


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    """
    Split *text* into overlapping chunks that respect paragraph boundaries.
    Returns at least one chunk even if the text is shorter than chunk_size.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    if not paragraphs:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)

        if current and current_len + para_len > chunk_size:
            chunks.append("\n\n".join(current))

            # Backtrack: keep tail paragraphs that fit within the overlap budget
            tail: list[str] = []
            tail_len = 0
            for p in reversed(current):
                if tail_len + len(p) <= overlap:
                    tail.insert(0, p)
                    tail_len += len(p)
                else:
                    break
            current, current_len = tail, tail_len

        current.append(para)
        current_len += para_len

    if current:
        chunks.append("\n\n".join(current))

    return chunks if chunks else [text]
