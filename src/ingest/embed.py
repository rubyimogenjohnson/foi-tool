"""
embed.py — sentence-transformers embedding wrapper.

Model: all-MiniLM-L6-v2
  - 384-dimensional embeddings
  - 256-token max sequence length
  - Fast and well-suited for semantic similarity / retrieval tasks
  - Downloaded automatically from HuggingFace on first use (~80 MB)
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Return the shared model instance, loading it on first call."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    """Return a list of embedding vectors (one per input text)."""
    return get_model().encode(texts, show_progress_bar=False).tolist()
