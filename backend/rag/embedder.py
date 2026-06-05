from sentence_transformers import SentenceTransformer
import numpy as np

_MODEL_NAME = "all-MiniLM-L6-v2"

def get_embedder() -> SentenceTransformer:
    """Load and return the sentence-transformers model."""
    return SentenceTransformer(_MODEL_NAME)

def embed_texts(embedder: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    """Embed a list of texts, return list of float vectors."""
    vectors = embedder.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    return vectors.tolist()

def embed_query(embedder: SentenceTransformer, query: str) -> list[float]:
    """Embed a single query string."""
    vec = embedder.encode([query], normalize_embeddings=True)
    return vec[0].tolist()
